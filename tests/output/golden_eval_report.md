# 🎯 AI Agent Golden Evaluation Report

**Timestamp:** `2026-10-14 18:40:51`  
**Total Test Cases:** `30`  
**Overall Accuracy:** `100.0%` (30/30 passed)  
**Failure Rate:** `0.0%`  
**Average Latency:** `2.9ms`  
**Grounding Quality Score:** `100.0%`  

---

## 📊 Category Breakdown

| Category | Total Cases | Passed | Failed | Accuracy | Avg Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **edge_case** | 4 | 4 | 0 | `100.0%` | `100.0%` |
| **grounding** | 6 | 6 | 0 | `100.0%` | `100.0%` |
| **intent_extraction** | 8 | 8 | 0 | `100.0%` | `100.0%` |
| **product_ranking** | 6 | 6 | 0 | `100.0%` | `100.0%` |
| **tool_selection** | 6 | 6 | 0 | `100.0%` | `100.0%` |

---

## 📋 Detailed Case Verdicts

| ID | Category | Task | Status | Score | Latency |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `intent_01` | intent_extraction | Find me the best DDR5 RAM deal on Amazon unde... | ✅ PASS | `100%` | 0.7ms |
| `intent_02` | intent_extraction | I need 32GB laptop DDR5 memory | ✅ PASS | `100%` | 0.0ms |
| `intent_03` | intent_extraction | What's the cheapest DDR5 RAM available for my... | ✅ PASS | `100%` | 0.0ms |
| `intent_04` | intent_extraction | I don't care about the price, give me the fas... | ✅ PASS | `100%` | 0.0ms |
| `intent_05` | intent_extraction | Top rated 16GB desktop RAM below 5000 | ✅ PASS | `100%` | 0.0ms |
| `intent_06` | intent_extraction | Find SODIMM DDR5 for my notebook below 8000 | ✅ PASS | `100%` | 0.0ms |
| `intent_07` | intent_extraction | Recommend the most reliable DDR5 RAM kit for ... | ✅ PASS | `100%` | 0.0ms |
| `intent_08` | intent_extraction | Find fastest DDR5 RAM under 50k for my gaming... | ✅ PASS | `100%` | 0.0ms |
| `tool_01` | tool_selection | Search Amazon India for DDR5 6000MHz memory k... | ✅ PASS | `100%` | 5.0ms |
| `tool_02` | tool_selection | Inspect the operating voltage and timings of ... | ✅ PASS | `100%` | 5.6ms |
| `tool_03` | tool_selection | Compare Corsair Vengeance and G.Skill Trident... | ✅ PASS | `100%` | 4.7ms |
| `tool_04` | tool_selection | Scrape the hardware specification article at ... | ✅ PASS | `100%` | 2.6ms |
| `tool_05` | tool_selection | Find 64GB DDR5 memory modules for server virt... | ✅ PASS | `100%` | 5.5ms |
| `tool_06` | tool_selection | Search for SODIMM laptop RAM kits under 9000 | ✅ PASS | `100%` | 4.8ms |
| `rank_01` | product_ranking | Find DDR5 RAM under 10000 and confirm all res... | ✅ PASS | `100%` | 4.4ms |
| `rank_02` | product_ranking | Find SODIMM DDR5 memory kits for a gaming lap... | ✅ PASS | `100%` | 4.1ms |
| `rank_03` | product_ranking | Identify the best value DDR5 RAM balancing sp... | ✅ PASS | `100%` | 4.6ms |
| `rank_04` | product_ranking | Find the cheapest DDR5 memory module availabl... | ✅ PASS | `100%` | 4.1ms |
| `rank_05` | product_ranking | Find the highest performance DDR5 gaming kit ... | ✅ PASS | `100%` | 4.1ms |
| `rank_06` | product_ranking | Perform a multi-item comparison and verify ve... | ✅ PASS | `100%` | 4.1ms |
| `ground_01` | grounding | What DDR5 RAM kits did you find on Amazon Ind... | ✅ PASS | `100%` | 4.7ms |
| `ground_02` | grounding | How much does the Crucial 16GB DDR5 RAM cost ... | ✅ PASS | `100%` | 4.2ms |
| `ground_03` | grounding | Is the Crucial 32GB 5200MHz memory suitable f... | ✅ PASS | `100%` | 4.3ms |
| `ground_04` | grounding | What is the price and specs of the Kingston F... | ✅ PASS | `100%` | 4.1ms |
| `ground_05` | grounding | Explain why the top recommended RAM was chose... | ✅ PASS | `100%` | 4.7ms |
| `ground_06` | grounding | What is the operating voltage and warranty on... | ✅ PASS | `100%` | 4.1ms |
| `edge_01` | edge_case | Find me 64GB DDR5 RAM for under 100 rupees | ✅ PASS | `100%` | 4.1ms |
| `edge_02` | edge_case | What is the weather in Mumbai today? | ✅ PASS | `100%` | 0.3ms |
| `edge_03` | edge_case |     | ✅ PASS | `100%` | 0.3ms |
| `edge_04` | edge_case | Find SODIMM DDR5 RAM for my standard desktop ... | ✅ PASS | `100%` | 0.3ms |