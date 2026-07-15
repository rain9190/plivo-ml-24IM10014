### Experiment Summary

| Run | Key Modification | Peak LR | Batch | BPB | Change vs Prev |
|:---|:---|:---:|:---:|:---:|:---|
| **0 (Baseline)** | Untuned byte-level starter, const LR. | 3e-4 | 8 | 2.3718 | - |
| **1 (Recipe)** | AdamW, weight decay, warmup/cosine, clip. | 1e-3 | 8 | 2.2516 | ⬇️ -0.1202 |
| **2 (BPE)** | Custom BPE tokenizer (vocab 2048). | 1e-3 | 8 | 1.9993 | ⬇️ -0.2523 |
| **3 (Deeper)** | *[REVERTED]* Tie weights, 5 layers. | 1e-3 | 8 | 2.0451 | ⬆️ +0.0458 |
| **4 (Batch)** | Swept batch size to 32. | 1e-3 | 32 | 1.8267 | ⬇️ -0.1726 |
| **5 (Batch+LR)**| Batch 48, raised peak LR. | 1.5e-3 | 48 | 1.7262 | ⬇️ -0.1005 |
| **6 (Max Out)** | Batch 64, raised peak LR (time limit ceiling). | 2e-3 | 64 | 1.6947 | ⬇️ -0.0315 |
| **7 (Script BPE)**| **[SELECTED]** Script-aware BPE, vocab 2304. | 2e-3 | 64 | **1.6933** | ⬇️ -0.0014 |

***Final Result:*** *Reduced BPB from 2.3718 to 1.6933 (a 28.6% reduction) staying within the strict 2,000 steps and 2M parameter budget.*