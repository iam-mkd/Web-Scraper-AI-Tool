# 🎯 AI Agent Golden Evaluation Report

**Timestamp:** `2026-10-14 18:38:46`  
**Total Test Cases:** `15`  
**Overall Accuracy:** `80.0%` (12/15 passed)  
**Failure Rate:** `20.0%`  
**Average Latency:** `14917.9ms`  
**Grounding Quality Score:** `100.0%`  

---

## 📊 Category Breakdown

| Category | Total Cases | Passed | Failed | Accuracy | Avg Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **intent_extraction** | 8 | 8 | 0 | `100.0%` | `100.0%` |
| **product_ranking** | 1 | 0 | 1 | `0.0%` | `0.0%` |
| **tool_selection** | 6 | 4 | 2 | `66.7%` | `77.8%` |

---

## 📋 Detailed Case Verdicts

| ID | Category | Task | Status | Score | Latency |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `intent_01` | intent_extraction | Find me the best DDR5 RAM deal on Amazon unde... | ✅ PASS | `100%` | 1.3ms |
| `intent_02` | intent_extraction | I need 32GB laptop DDR5 memory | ✅ PASS | `100%` | 0.8ms |
| `intent_03` | intent_extraction | What's the cheapest DDR5 RAM available for my... | ✅ PASS | `100%` | 1.6ms |
| `intent_04` | intent_extraction | I don't care about the price, give me the fas... | ✅ PASS | `100%` | 0.7ms |
| `intent_05` | intent_extraction | Top rated 16GB desktop RAM below 5000 | ✅ PASS | `100%` | 1.0ms |
| `intent_06` | intent_extraction | Find SODIMM DDR5 for my notebook below 8000 | ✅ PASS | `100%` | 0.9ms |
| `intent_07` | intent_extraction | Recommend the most reliable DDR5 RAM kit for ... | ✅ PASS | `100%` | 1.0ms |
| `intent_08` | intent_extraction | Find fastest DDR5 RAM under 50k for my gaming... | ✅ PASS | `100%` | 0.8ms |
| `tool_01` | tool_selection | Search Amazon India for DDR5 6000MHz memory k... | ✅ PASS | `100%` | 4492.0ms |
| `tool_02` | tool_selection | Inspect the operating voltage and timings of ... | ✅ PASS | `100%` | 36499.7ms |
| `tool_03` | tool_selection | Compare Corsair Vengeance and G.Skill Trident... | ❌ FAIL | `0%` | 58524.7ms |
| `tool_04` | tool_selection | Scrape the hardware specification article at ... | ✅ PASS | `100%` | 29406.8ms |
| `tool_05` | tool_selection | Find 64GB DDR5 memory modules for server virt... | ❌ FAIL | `67%` | 37546.6ms |
| `tool_06` | tool_selection | Search for SODIMM laptop RAM kits under 9000 | ✅ PASS | `100%` | 36742.3ms |
| `rank_01` | product_ranking | Find DDR5 RAM under 10000 and confirm all res... | ❌ FAIL | `0%` | 20547.9ms |

---

## 🔍 Failure Diagnostics

### Case `tool_03` (tool_selection)
- **Task:** Compare Corsair Vengeance and G.Skill Trident Z5 side by side
- ❌ **Check `permitted_tools` Failed:** Tools called ['search_amazon', 'search_amazon', 'scrape_url'] contain unpermitted tool
- ❌ **Check `forbidden_tool_scrape_url` Failed:** Forbidden tool 'scrape_url' unexpectedly called
- ❌ **Check `tool_arg_product_urls` Failed:** Argument 'product_urls': expected=['https://www.amazon.in/CORSAIR-Vengeance-5200MHz-CL40-40-40-77-Computer/dp/B0DM2PGP4N/ref=sr_1_3?sr=8-3', 'https://www.amazon.in/G-Skill-Trident-CL36-36-36-96-Desktop-Memory/dp/B09TPY273P/ref=sr_1_1?sr=8-1'], got=None

### Case `tool_05` (tool_selection)
- **Task:** Find 64GB DDR5 memory modules for server virtualization
- ❌ **Check `tool_arg_query` Failed:** Argument 'query': expected='64GB DDR5 RAM', got='DDR5 64GB UDIMM ECC'

### Case `rank_01` (product_ranking)
- **Task:** Find DDR5 RAM under 10000 and confirm all results respect the budget
- **Error:** `Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-20b` in organization `org_01m418d0b2eyzveq061td61ngm` service tier `on_demand` on tokens per day (TPD): Limit 200000, Used 199567, Requested 1924. Please try again in 10m44.112s. Need more tokens? Upgrade to Dev Tier today at https://console.groq.com/settings/billing', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}`
