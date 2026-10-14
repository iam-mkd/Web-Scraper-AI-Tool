# 🎯 AI Agent Golden Evaluation Report

**Timestamp:** `2026-10-14 15:44:40`  
**Total Test Cases:** `10`  
**Overall Accuracy:** `80.0%` (8/10 passed)  
**Failure Rate:** `20.0%`  
**Average Latency:** `3352.0ms`  
**Grounding Quality Score:** `100.0%`  

---

## 📊 Category Breakdown

| Category | Total Cases | Passed | Failed | Accuracy | Avg Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **intent_extraction** | 8 | 8 | 0 | `100.0%` | `100.0%` |
| **tool_selection** | 2 | 0 | 2 | `0.0%` | `0.0%` |

---

## 📋 Detailed Case Verdicts

| ID | Category | Task | Status | Score | Latency |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `intent_01` | intent_extraction | Find me the best DDR5 RAM deal on Amazon unde... | ✅ PASS | `100%` | 1.9ms |
| `intent_02` | intent_extraction | I need 32GB laptop DDR5 memory | ✅ PASS | `100%` | 0.9ms |
| `intent_03` | intent_extraction | What's the cheapest DDR5 RAM available for my... | ✅ PASS | `100%` | 1.5ms |
| `intent_04` | intent_extraction | I don't care about the price, give me the fas... | ✅ PASS | `100%` | 0.8ms |
| `intent_05` | intent_extraction | Top rated 16GB desktop RAM below 5000 | ✅ PASS | `100%` | 0.9ms |
| `intent_06` | intent_extraction | Find SODIMM DDR5 for my notebook below 8000 | ✅ PASS | `100%` | 0.7ms |
| `intent_07` | intent_extraction | Recommend the most reliable DDR5 RAM kit for ... | ✅ PASS | `100%` | 0.9ms |
| `intent_08` | intent_extraction | Find fastest DDR5 RAM under 50k for my gaming... | ✅ PASS | `100%` | 0.8ms |
| `tool_01` | tool_selection | Search Amazon India for DDR5 6000MHz memory k... | ❌ FAIL | `0%` | 27196.6ms |
| `tool_02` | tool_selection | Inspect the operating voltage and timings of ... | ❌ FAIL | `0%` | 6315.0ms |

---

## 🔍 Failure Diagnostics

### Case `tool_01` (tool_selection)
- **Task:** Search Amazon India for DDR5 6000MHz memory kits under 12000
- **Error:** `Error code: 429 - {'error': {'message': "Request too large for model `qwen/qwen3.8-27b` in organization `org_01m418d0b2eyzveq061td61ngm` service tier `on_demand` on output tokens per minute (OTPM): Limit 1000, Requested 1080. The request's expected output tokens exceed the enforced limit; reduce max_tokens (or the request's expected output) and try again. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing", 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`

### Case `tool_02` (tool_selection)
- **Task:** Inspect the operating voltage and timings of Corsair Vengeance DDR5 from Amazon: https://www.amazon.in/CORSAIR-Vengeance-5200MHz-CL40-40-40-77-Computer/dp/B0DM2PGP4N/ref=sr_1_3?sr=8-3
- **Error:** `Error code: 429 - {'error': {'message': "Request too large for model `qwen/qwen3.8-27b` in organization `org_01m418d0b2eyzveq061td61ngm` service tier `on_demand` on output tokens per minute (OTPM): Limit 1000, Requested 1089. The request's expected output tokens exceed the enforced limit; reduce max_tokens (or the request's expected output) and try again. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing", 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
