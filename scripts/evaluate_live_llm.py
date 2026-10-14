import os
import sys

# Force unbuffered UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

import time
import json
import asyncio
import argparse
from typing import List, Dict, Any, Optional
from unittest.mock import patch
from dotenv import load_dotenv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

load_dotenv()

from app.agent.service import AgentService
from tests.golden_eval_dataset import GOLDEN_EVAL_DATASET, CASE_MAP, CATEGORY_INDEX
from tests.eval_graders import (
    grade_case,
    grade_intent_extraction,
    generate_report,
    format_markdown_report,
    GradeResult,
    EvalReport,
)
from tests.conftest import (
    MOCK_AMAZON_SEARCH_HTML,
    MOCK_AMAZON_DETAIL_HTML,
    FIXED_PRODUCTS_DATASET,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate real LLM against Golden Dataset.")
    parser.add_argument("--category", choices=["intent_extraction", "tool_selection", "product_ranking", "grounding", "edge_case"])
    parser.add_argument("--case", help="Single case ID (e.g. tool_01)")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between tasks in seconds (default: 1.5s)")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", default=os.path.join(REPO_ROOT, "tests", "output", "live_llm_eval_report.md"))
    parser.add_argument("--json-output", default=os.path.join(REPO_ROOT, "tests", "output", "live_llm_eval_report.json"))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

def save_current_report(results: List[GradeResult], output_md: str, output_json: str):
    if not results:
        return
    report = generate_report(results)
    md_content = format_markdown_report(report)
    os.makedirs(os.path.dirname(output_md), exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": report.timestamp,
                "total_cases": report.total_cases,
                "passed_cases": report.passed_cases,
                "failed_cases": report.failed_cases,
                "overall_accuracy": report.overall_accuracy,
                "failure_rate": report.failure_rate,
                "avg_latency_ms": report.avg_latency_ms,
                "grounding_avg_score": report.grounding_avg_score,
                "category_metrics": report.category_metrics,
                "results": [
                    {
                        "case_id": r.case_id,
                        "category": r.category,
                        "task": r.task,
                        "passed": r.passed,
                        "score": r.score,
                        "latency_ms": r.latency_ms,
                        "error": r.error,
                        "checks": [
                            {"name": c.name, "passed": c.passed, "details": c.details}
                            for c in r.checks
                        ],
                    }
                    for r in report.results
                ],
            }, f, indent=2)

async def run_live_evaluation():
    args = parse_args()
    model_name = args.model or os.getenv("OLLAMA_MODEL") or os.getenv("GROQ_MODEL") or "qwen3.5:4b"
    is_remote = "/" in model_name or "gpt" in model_name.lower() or "gemini" in model_name.lower()
    if is_remote and "qwen" not in model_name.lower():
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            print("[ERROR] Remote API key (GROQ_API_KEY or GEMINI_API_KEY) is not set in .env.", flush=True)
            sys.exit(1)

    cases_to_run = list(GOLDEN_EVAL_DATASET)
    if args.category:
        cases_to_run = [c for c in cases_to_run if c["category"] == args.category]
    if args.case:
        cases_to_run = [c for c in cases_to_run if c["id"] == args.case]
    if args.max_cases:
        cases_to_run = cases_to_run[:args.max_cases]

    if not cases_to_run:
        print("[ERROR] No test cases matched the specified filters.", flush=True)
        sys.exit(1)

    print("=" * 75, flush=True)
    print("LIVE LLM GOLDEN EVALUATION (MEASUREMENT ENGINE)", flush=True)
    print(f"Model        : {model_name}", flush=True)
    print(f"Cases to run : {len(cases_to_run)} cases", flush=True)
    print(f"Pacing delay : {args.delay:.1f}s between tasks", flush=True)
    print(f"Report Path  : {args.output}", flush=True)
    print("=" * 75, flush=True)

    agent = AgentService(llm_model=args.model)
    results = []

    for idx, case in enumerate(cases_to_run, 1):
        case_id = case["id"]
        category = case["category"]
        task = case["task"]

        print(f"\n[{idx}/{len(cases_to_run)}] ({category}) Case: {case_id}", flush=True)
        print(f"  Task: \"{task}\"", flush=True)
        t0 = time.perf_counter()

        with patch("app.scrapers.amazon_scraper.fetch_html", return_value=MOCK_AMAZON_SEARCH_HTML), \
             patch("app.scrapers.amazon_scraper.get_amazon_product_details", return_value=FIXED_PRODUCTS_DATASET[0]), \
             patch("app.scrapers.base_scraper.fetch_html", return_value="<html><body>Mock Hardware Specs</body></html>"):
            if category == "intent_extraction":
                grade = grade_intent_extraction(case)
                agent_out = None
            else:
                try:
                    agent_out = await agent.run_task(task, max_steps=args.max_steps)
                    grade = grade_case(case, agent_out)
                except Exception as e:
                    grade = GradeResult(
                        case_id=case_id,
                        category=category,
                        task=task,
                        passed=False,
                        score=0.0,
                        error=str(e),
                    )
                    agent_out = None

        latency_ms = (time.perf_counter() - t0) * 1000
        grade.latency_ms = latency_ms
        results.append(grade)

        status_str = "PASS" if grade.passed else "FAIL"
        print(f"  Result : {status_str} | Score: {grade.score * 100:.0f}% | Latency: {latency_ms:.0f}ms", flush=True)

        if not grade.passed:
            for c in grade.checks:
                if not c.passed:
                    print(f"    - {c.name}: {c.details}", flush=True)
            if grade.error:
                print(f"    - Error: {grade.error}", flush=True)

        if args.verbose and agent_out:
            print(f"  Tools called: {agent_out.get('tools_called')}", flush=True)
            answer_prev = (agent_out.get('final_answer') or '')[:120].replace('\n', ' ')
            print(f"  Answer preview: {answer_prev}...", flush=True)

        # Incrementally persist report after each case
        save_current_report(results, args.output, args.json_output)

        if idx < len(cases_to_run) and args.delay > 0:
            await asyncio.sleep(args.delay)

    report = generate_report(results)
    print("\n" + "=" * 75, flush=True)
    print("EVALUATION COMPLETE", flush=True)
    print(f"Overall Accuracy : {report.overall_accuracy * 100:.1f}% ({report.passed_cases}/{report.total_cases} passed)", flush=True)
    print(f"Failure Rate     : {report.failure_rate * 100:.1f}%", flush=True)
    print(f"Average Latency  : {report.avg_latency_ms:.0f}ms", flush=True)
    print(f"Report Saved To  : {args.output}", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    asyncio.run(run_live_evaluation())