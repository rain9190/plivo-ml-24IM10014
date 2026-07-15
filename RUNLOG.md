# RUNLOG

One entry per training run: hypothesis → what changed → dev bpb before/after → conclusion.
Score command: `python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt`

## Corpus facts (measured, drive every decision below)
- Train corpus: 7,318,592 bytes / 5,703,936 chars. Mixed English + Hindi (Wikipedia-style).
- **Byte budget split: ASCII 66.9%, Devanagari 32.9%, other 0.3%.**
- Byte-level tokenizer pays **3.00 bytes/char on Devanagari** → ~1/3 of all compute is spent
  predicting Hindi one byte at a time. This is the single biggest inefficiency.
- Measured BPE compression (trained on corpus): vocab 1024 → ~2.69 bytes/token.
  → a BPE tokenizer lets the fixed 2000-step budget see ~2.7× more text, and makes
    block_size=128 span ~344 bytes of real context instead of 128.

---

## Run 0 — Baseline (reference)
- **Hypothesis:** starter is "mediocre on purpose"; establish the number to beat.
- **Config:** byte-level tokenizer (vocab 256), block 128, 4 layer / 4 head / n_embd 160,
  Adam constant LR 3e-4, no warmup/schedule/weight-decay/clipping, tie_weights=False,
  init N(0, 0.05), 2000 steps, batch 8.
- **dev bpb: 2.3718** | n_params 1,339,840 | steps 2000 | ~31s train.
- **Observation:** train loss still falling at step 2000 (1.73, not plateaued)
  → model is **under-optimizing**, not overfitting. Headroom is in the training recipe
  (LR too timid, no schedule) AND in the tokenizer (byte-level wastes the Hindi third).
- **Conclusion:** two independent levers to attack — recipe (cheap, safe) then tokenizer (big).

---

<!-- New entries appended below as experiments run. -->
