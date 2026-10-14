"""
Evaluation Graders and Report Generator for Computer Hardware Agent

Provides deterministic metrics and grounding checks:
1. Intent & constraint extraction accuracy (budget, optimization, form-factor, capacity)
2. Tool and argument selection accuracy
3. Budget compliance (checks all returned products stay <= specified budget)
4. Form-factor correctness (UDIMM vs SODIMM compatibility verification)
5. Product ranking correctness (deterministic Value Score vs Lowest Price vs Performance)
6. Grounding grader (verifies factual claims in final answer match tool outputs)
7. Aggregate reporting (task success rate, latency, failure rate, and cost estimation)
"""
import time
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from app.agent.service import (
    analyze_task_intent,
    calculate_value_score,
    build_product_comparison,
)


@dataclass
class CheckResult:
    """Individual assertion check inside a graded evaluation case."""
    name: str
    passed: bool
    expected: Any
    actual: Any
    details: str = ""


@dataclass
class GradeResult:
    """Evaluation result for a single test case."""
    case_id: str
    category: str
    task: str
    passed: bool
    score: float  # 0.0 to 1.0
    checks: List[CheckResult] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None
    grounding_score: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class EvalReport:
    """Aggregate evaluation summary across all test cases."""
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    overall_accuracy: float = 0.0
    failure_rate: float = 0.0
    avg_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    category_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    grounding_avg_score: float = 0.0
    total_estimated_cost_usd: float = 0.0
    results: List[GradeResult] = field(default_factory=list)
    timestamp: str = ""


# ---------------------------------------------------------------------------
# 1. Intent & Constraint Extraction Grader
# ---------------------------------------------------------------------------
def grade_intent_extraction(case: Dict[str, Any]) -> GradeResult:
    """
    Evaluates analyze_task_intent output against expected intent and constraints.
    Checks:
    - Optimization goal (Raw performance, Lowest price, Highest customer rating, Best value)
    - Budget string representation
    - Numeric budget value
    - Form factor (UDIMM, SODIMM, Any)
    - Capacity (e.g. 16GB, 32GB)
    """
    t0 = time.perf_counter()
    task = case["task"]
    expected = case["expected"]
    checks: List[CheckResult] = []

    try:
        actual = analyze_task_intent(task)
    except Exception as e:
        return GradeResult(
            case_id=case["id"],
            category=case["category"],
            task=task,
            passed=False,
            score=0.0,
            error=f"Exception in analyze_task_intent: {str(e)}",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    latency_ms = (time.perf_counter() - t0) * 1000

    # Optimization
    if expected.get("optimization") is not None:
        p = actual.get("optimization") == expected["optimization"]
        checks.append(CheckResult(
            name="optimization",
            passed=p,
            expected=expected["optimization"],
            actual=actual.get("optimization"),
            details=f"Expected '{expected['optimization']}', got '{actual.get('optimization')}'",
        ))

    # Budget String
    if expected.get("budget") is not None:
        p = actual.get("budget") == expected["budget"]
        checks.append(CheckResult(
            name="budget_str",
            passed=p,
            expected=expected["budget"],
            actual=actual.get("budget"),
            details=f"Expected '{expected['budget']}', got '{actual.get('budget')}'",
        ))

    # Budget Numeric Value
    if "budget_val" in expected:
        exp_val = expected["budget_val"]
        act_val = actual.get("budget_val")
        if exp_val is None:
            p = act_val is None
        elif act_val is None:
            p = False
        else:
            p = abs(float(act_val) - float(exp_val)) < 0.01

        checks.append(CheckResult(
            name="budget_val",
            passed=p,
            expected=exp_val,
            actual=act_val,
            details=f"Expected budget_val={exp_val}, got {act_val}",
        ))

    # Form Factor
    if expected.get("form_factor") is not None:
        p = actual.get("form_factor") == expected["form_factor"]
        checks.append(CheckResult(
            name="form_factor",
            passed=p,
            expected=expected["form_factor"],
            actual=actual.get("form_factor"),
            details=f"Expected form_factor='{expected['form_factor']}', got '{actual.get('form_factor')}'",
        ))

    # Target Capacity
    if "capacity" in expected and expected["capacity"] is not None:
        p = actual.get("capacity") == expected["capacity"]
        checks.append(CheckResult(
            name="capacity",
            passed=p,
            expected=expected["capacity"],
            actual=actual.get("capacity"),
            details=f"Expected capacity='{expected['capacity']}', got '{actual.get('capacity')}'",
        ))

    passed_count = sum(1 for c in checks if c.passed)
    score = passed_count / len(checks) if checks else 1.0
    passed = all(c.passed for c in checks)

    return GradeResult(
        case_id=case["id"],
        category=case["category"],
        task=task,
        passed=passed,
        score=score,
        checks=checks,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# 2. Tool & Argument Selection Grader
# ---------------------------------------------------------------------------
def grade_tool_selection(case: Dict[str, Any], agent_result: Dict[str, Any]) -> GradeResult:
    """
    Evaluates whether the agent selected the proper tools and populated valid arguments.
    Checks:
    - Permitted tools called
    - Forbidden tools NOT called
    - Parameter accuracy (e.g. max_price, capacity_gb, form_factor)
    """
    task = case["task"]
    expected = case["expected"]
    checks: List[CheckResult] = []

    tools_called = agent_result.get("tools_called") or []
    if agent_result.get("tool_called") and agent_result.get("tool_called") not in tools_called:
        tools_called.append(agent_result["tool_called"])

    # Permitted tools
    permitted = expected.get("permitted_tools", [])
    if permitted:
        p = len(tools_called) > 0 and all(t in permitted for t in tools_called)
        checks.append(CheckResult(
            name="permitted_tools",
            passed=p,
            expected=f"Subset of {permitted}",
            actual=tools_called,
            details=f"Tools called {tools_called} {'are permitted' if p else 'contain unpermitted tool'}",
        ))

    # Forbidden tools
    for tool_name in expected.get("forbidden_tools", []):
        p = tool_name not in tools_called
        checks.append(CheckResult(
            name=f"forbidden_tool_{tool_name}",
            passed=p,
            expected=f"NOT {tool_name}",
            actual=tools_called,
            details=f"Forbidden tool '{tool_name}' {'correctly avoided' if p else 'unexpectedly called'}",
        ))

    # Argument validation
    expected_args = expected.get("tool_args") or {}
    actual_args = agent_result.get("tool_arguments") or {}

    for arg_k, arg_v in expected_args.items():
        act_v = actual_args.get(arg_k)
        if isinstance(arg_v, float) and isinstance(act_v, (int, float)):
            p = abs(float(act_v) - arg_v) < 0.01
        elif isinstance(arg_v, str) and isinstance(act_v, str):
            p = arg_v.lower() in act_v.lower() or act_v.lower() in arg_v.lower()
        else:
            p = act_v == arg_v

        checks.append(CheckResult(
            name=f"tool_arg_{arg_k}",
            passed=p,
            expected=arg_v,
            actual=act_v,
            details=f"Argument '{arg_k}': expected={arg_v!r}, got={act_v!r}",
        ))

    passed_count = sum(1 for c in checks if c.passed)
    score = passed_count / len(checks) if checks else 1.0
    passed = all(c.passed for c in checks)

    return GradeResult(
        case_id=case["id"],
        category=case["category"],
        task=task,
        passed=passed,
        score=score,
        checks=checks,
    )


# ---------------------------------------------------------------------------
# 3. Budget Compliance Grader
# ---------------------------------------------------------------------------
def grade_budget_compliance(budget_limit: float, products: List[Dict[str, Any]]) -> CheckResult:
    """
    Checks that 100% of priced products in the candidate list strictly adhere to budget.
    """
    violations = [p for p in products if p.get("price") is not None and p["price"] > budget_limit]
    passed = len(violations) == 0
    return CheckResult(
        name="budget_compliance",
        passed=passed,
        expected=f"All prices <= ₹{budget_limit:,.0f}",
        actual=[p.get("price") for p in products if p.get("price") is not None],
        details=(
            f"All {len(products)} products within budget limit ₹{budget_limit:,.0f}"
            if passed
            else f"{len(violations)} products violated budget: {[v.get('name', '')[:30] for v in violations]}"
        ),
    )


# ---------------------------------------------------------------------------
# 4. Form-Factor Correctness Grader
# ---------------------------------------------------------------------------
def grade_form_factor_correctness(expected_ff: str, products: List[Dict[str, Any]]) -> CheckResult:
    """
    Checks that all returned products match the required form factor (UDIMM or SODIMM).
    """
    violations = [p for p in products if p.get("form_factor") and p["form_factor"].upper() != expected_ff.upper()]
    passed = len(violations) == 0
    return CheckResult(
        name="form_factor_correctness",
        passed=passed,
        expected=f"All products form_factor == '{expected_ff}'",
        actual=[p.get("form_factor") for p in products],
        details=(
            f"All products verified as {expected_ff}"
            if passed
            else f"{len(violations)} products have incompatible form factors"
        ),
    )


# ---------------------------------------------------------------------------
# 5. Product Ranking & Comparison Grader
# ---------------------------------------------------------------------------
def grade_product_ranking(case: Dict[str, Any], agent_result: Dict[str, Any]) -> GradeResult:
    """
    Evaluates ranking outputs: budget adherence, form factor compliance,
    minimum product count, and top Value Score pick.
    """
    task = case["task"]
    expected = case["expected"]
    checks: List[CheckResult] = []

    products = agent_result.get("products") or []
    stats = agent_result.get("stats") or {}

    # Check 1: Minimum products returned
    if expected.get("min_products") is not None:
        p = len(products) >= expected["min_products"]
        checks.append(CheckResult(
            name="min_products",
            passed=p,
            expected=f">={expected['min_products']}",
            actual=len(products),
            details=f"Expected at least {expected['min_products']} products, got {len(products)}",
        ))

    # Check 2: Budget compliance
    if expected.get("budget_compliant") and expected.get("budget_val") is not None:
        checks.append(grade_budget_compliance(expected["budget_val"], products))

    # Check 3: Form-factor compliance
    if expected.get("form_factor_compliant") and expected.get("form_factor"):
        checks.append(grade_form_factor_correctness(expected["form_factor"], products))

    # Check 4: Best value pick name match
    if expected.get("best_value_name_contains"):
        bv = stats.get("best_value_product") or {}
        bv_name = (bv.get("name") or "").lower()
        needle = expected["best_value_name_contains"].lower()
        p = needle in bv_name or any(needle in (p.get("name") or "").lower() for p in products[:1])
        checks.append(CheckResult(
            name="best_value_ranking",
            passed=p,
            expected=f"contains '{expected['best_value_name_contains']}'",
            actual=bv_name,
            details=f"Best value product name '{bv_name}' contains '{needle}': {p}",
        ))

    # Check 5: Cheapest pick name match
    if expected.get("cheapest_name_contains"):
        ch = stats.get("cheapest_product") or {}
        ch_name = (ch.get("name") or "").lower()
        needle = expected["cheapest_name_contains"].lower()
        p = needle in ch_name or any(needle in (p.get("name") or "").lower() for p in products)
        checks.append(CheckResult(
            name="cheapest_ranking",
            passed=p,
            expected=f"contains '{expected['cheapest_name_contains']}'",
            actual=ch_name,
            details=f"Cheapest product name '{ch_name}' contains '{needle}': {p}",
        ))

    passed_count = sum(1 for c in checks if c.passed)
    score = passed_count / len(checks) if checks else 1.0
    passed = all(c.passed for c in checks)

    return GradeResult(
        case_id=case["id"],
        category=case["category"],
        task=task,
        passed=passed,
        score=score,
        checks=checks,
    )


# ---------------------------------------------------------------------------
# 6. Grounding Grader (Factual Verification Against Tool Outputs)
# ---------------------------------------------------------------------------
def grade_grounding(case: Dict[str, Any], agent_result: Dict[str, Any]) -> GradeResult:
    """
    Checks that final answer claims are grounded in actual tool outputs.
    Verifies:
    1. Must-contain facts (actual product names, verified prices, Value Score, form factor)
    2. Must-not-contain facts (hallucinated prices, out-of-domain products, fabricated specs)
    3. Tool output grounding: verifies that any price cited in the answer actually exists in the retrieved products.
    """
    task = case["task"]
    expected = case["expected"]
    checks: List[CheckResult] = []

    final_answer = agent_result.get("final_answer") or agent_result.get("explanation") or ""
    answer_lower = final_answer.lower()
    products = agent_result.get("products") or []

    # Check 1: Must-contain facts
    for phrase in expected.get("answer_must_contain", []):
        if isinstance(phrase, (tuple, list)):
            p = any(alt.lower() in answer_lower for alt in phrase)
            disp = " | ".join(phrase)
        else:
            p = phrase.lower() in answer_lower
            disp = phrase
        checks.append(CheckResult(
            name=f"must_contain_{disp[:20]}",
            passed=p,
            expected=disp,
            actual=p,
            details=f"Phrase '{disp}' {'found' if p else 'MISSING'} in final answer",
        ))

    # Check 2: Must-not-contain facts
    for phrase in expected.get("answer_must_not_contain", []):
        p = phrase.lower() not in answer_lower
        checks.append(CheckResult(
            name=f"must_not_contain_{phrase[:20]}",
            passed=p,
            expected=f"NOT '{phrase}'",
            actual=not p,
            details=f"Forbidden phrase '{phrase}' {'absent (good)' if p else 'FOUND (hallucination)'}",
        ))

    # Check 3: Pricing grounding against tool outputs
    # Extract ₹ prices from the text (e.g. ₹9,499 or Rs. 3899)
    cited_prices = [
        float(m.replace(",", ""))
        for m in re.findall(r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)", answer_lower)
    ]
    actual_prices = [p.get("price") for p in products if p.get("price") is not None]

    cited_product_prices = [cp for cp in cited_prices if cp >= 1000.0]
    if cited_product_prices and actual_prices:
        grounded_prices = [
            cp for cp in cited_product_prices
            if any(abs(cp - ap) < 1.0 or abs(cp - (ap / 1000.0)) < 0.1 for ap in actual_prices)
        ]
        # At least 70% of mentioned product prices should match actual tool data
        grounding_ratio = len(grounded_prices) / len(cited_product_prices) if cited_product_prices else 1.0
        checks.append(CheckResult(
            name="price_grounding",
            passed=grounding_ratio >= 0.7,
            expected="Prices grounded in tool output",
            actual=cited_product_prices,
            details=f"{len(grounded_prices)}/{len(cited_product_prices)} cited prices match real product data",
        ))

    passed_count = sum(1 for c in checks if c.passed)
    score = passed_count / len(checks) if checks else 1.0
    passed = all(c.passed for c in checks)

    return GradeResult(
        case_id=case["id"],
        category=case["category"],
        task=task,
        passed=passed,
        score=score,
        checks=checks,
        grounding_score=score,
    )


# ---------------------------------------------------------------------------
# 7. Edge Case Grader
# ---------------------------------------------------------------------------
def grade_edge_case(case: Dict[str, Any], agent_result: Dict[str, Any]) -> GradeResult:
    """
    Evaluates agent resilience on impossible budgets, contradictory inputs,
    and out-of-domain requests without crashing or hallucinating.
    """
    task = case["task"]
    checks: List[CheckResult] = []

    # Check 1: Agent didn't crash
    err = agent_result.get("error")
    no_crash = err is None
    checks.append(CheckResult(
        name="no_crash",
        passed=no_crash,
        expected="Execution succeeds without unhandled exceptions",
        actual=err,
        details="Execution completed cleanly" if no_crash else f"Crashed with: {err}",
    ))

    # Check 2: Final answer produced
    final_answer = agent_result.get("final_answer") or agent_result.get("explanation") or ""
    has_answer = len(final_answer.strip()) > 0
    checks.append(CheckResult(
        name="has_answer",
        passed=has_answer,
        expected="Non-empty response",
        actual=len(final_answer),
        details=f"Response produced ({len(final_answer)} characters)",
    ))

    # Check 3: Grounding on edge case expectations
    grounding_grade = grade_grounding(case, agent_result)
    checks.extend(grounding_grade.checks)

    passed_count = sum(1 for c in checks if c.passed)
    score = passed_count / len(checks) if checks else 1.0
    passed = all(c.passed for c in checks)

    return GradeResult(
        case_id=case["id"],
        category=case["category"],
        task=task,
        passed=passed,
        score=score,
        checks=checks,
        error=err,
    )


# ---------------------------------------------------------------------------
# Master Grader Dispatcher
# ---------------------------------------------------------------------------
def grade_case(case: Dict[str, Any], agent_result: Optional[Dict[str, Any]] = None) -> GradeResult:
    """
    Dispatches a test case to the appropriate grader based on category.
    """
    cat = case["category"]
    agent_result = agent_result or {}

    if cat == "intent_extraction":
        return grade_intent_extraction(case)
    elif cat == "tool_selection":
        return grade_tool_selection(case, agent_result)
    elif cat == "product_ranking":
        return grade_product_ranking(case, agent_result)
    elif cat == "grounding":
        return grade_grounding(case, agent_result)
    elif cat == "edge_case":
        return grade_edge_case(case, agent_result)
    else:
        return GradeResult(
            case_id=case["id"],
            category=cat,
            task=case["task"],
            passed=False,
            score=0.0,
            error=f"Unknown evaluation category: '{cat}'",
        )


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------
def generate_report(results: List[GradeResult]) -> EvalReport:
    """
    Aggregates per-case evaluation results into a comprehensive EvalReport.
    Calculates overall accuracy, per-category breakdown, grounding scores,
    average latency, and failure rates.
    """
    total = len(results)
    passed_cases = sum(1 for r in results if r.passed)
    failed_cases = total - passed_cases

    latencies = [r.latency_ms for r in results if r.latency_ms > 0]
    total_lat = sum(latencies)
    avg_lat = total_lat / len(latencies) if latencies else 0.0

    grounding_scores = [r.grounding_score for r in results if r.grounding_score is not None]
    avg_grounding = sum(grounding_scores) / len(grounding_scores) if grounding_scores else 1.0

    # Category Breakdown
    cat_buckets: Dict[str, List[GradeResult]] = {}
    for r in results:
        cat_buckets.setdefault(r.category, []).append(r)

    cat_metrics: Dict[str, Dict[str, Any]] = {}
    for cat, items in cat_buckets.items():
        cat_passed = sum(1 for x in items if x.passed)
        cat_total = len(items)
        cat_avg_score = sum(x.score for x in items) / cat_total if cat_total else 0.0
        cat_metrics[cat] = {
            "total": cat_total,
            "passed": cat_passed,
            "failed": cat_total - cat_passed,
            "accuracy": cat_passed / cat_total if cat_total else 0.0,
            "avg_score": cat_avg_score,
        }

    total_cost = sum(r.estimated_cost_usd for r in results)

    return EvalReport(
        total_cases=total,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        overall_accuracy=passed_cases / total if total else 0.0,
        failure_rate=failed_cases / total if total else 0.0,
        avg_latency_ms=avg_lat,
        total_latency_ms=total_lat,
        category_metrics=cat_metrics,
        grounding_avg_score=avg_grounding,
        total_estimated_cost_usd=total_cost,
        results=results,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def format_markdown_report(report: EvalReport) -> str:
    """
    Formats the EvalReport as clean, structured GitHub-flavored Markdown.
    """
    md = [
        "# 🎯 AI Agent Golden Evaluation Report",
        "",
        f"**Timestamp:** `{report.timestamp}`  ",
        f"**Total Test Cases:** `{report.total_cases}`  ",
        f"**Overall Accuracy:** `{report.overall_accuracy * 100:.1f}%` ({report.passed_cases}/{report.total_cases} passed)  ",
        f"**Failure Rate:** `{report.failure_rate * 100:.1f}%`  ",
        f"**Average Latency:** `{report.avg_latency_ms:.1f}ms`  ",
        f"**Grounding Quality Score:** `{report.grounding_avg_score * 100:.1f}%`  ",
        "",
        "---",
        "",
        "## 📊 Category Breakdown",
        "",
        "| Category | Total Cases | Passed | Failed | Accuracy | Avg Score |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cat, m in sorted(report.category_metrics.items()):
        md.append(
            f"| **{cat}** | {m['total']} | {m['passed']} | {m['failed']} | "
            f"`{m['accuracy'] * 100:.1f}%` | `{m['avg_score'] * 100:.1f}%` |"
        )

    md.extend([
        "",
        "---",
        "",
        "## 📋 Detailed Case Verdicts",
        "",
        "| ID | Category | Task | Status | Score | Latency |",
        "| :--- | :--- | :--- | :---: | :---: | :---: |",
    ])

    for r in report.results:
        status_icon = "✅ PASS" if r.passed else "❌ FAIL"
        short_task = (r.task[:45] + "...") if len(r.task) > 45 else r.task
        lat = f"{r.latency_ms:.1f}ms" if r.latency_ms > 0 else "N/A"
        md.append(
            f"| `{r.case_id}` | {r.category} | {short_task} | {status_icon} | `{r.score * 100:.0f}%` | {lat} |"
        )

    # Failed cases diagnostic section
    failures = [r for r in report.results if not r.passed]
    if failures:
        md.extend([
            "",
            "---",
            "",
            "## 🔍 Failure Diagnostics",
            "",
        ])
        for f in failures:
            md.append(f"### Case `{f.case_id}` ({f.category})")
            md.append(f"- **Task:** {f.task}")
            if f.error:
                md.append(f"- **Error:** `{f.error}`")
            for c in f.checks:
                if not c.passed:
                    md.append(f"- ❌ **Check `{c.name}` Failed:** {c.details}")
            md.append("")

    return "\n".join(md)
