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

## Run 0 - Baseline (reference)
- **Hypothesis:** starter is "mediocre on purpose"; establish the number to beat.
- **Config:** byte-level tokenizer (vocab 256), block 128, 4 layer / 4 head / n_embd 160,
  Adam constant LR 3e-4, no warmup/schedule/weight-decay/clipping, tie_weights=False,
  init N(0, 0.05), 2000 steps, batch 8.
- **dev bpb: 2.3718** | n_params 1,339,840 | steps 2000 | ~31s train.
- **Observation:** train loss still falling at step 2000 (1.73, not plateaued)
  → model is **under-optimizing**, not overfitting. Headroom is in the training recipe
  (LR too timid, no schedule) AND in the tokenizer (byte-level wastes the Hindi third).
- **Conclusion:** two independent levers to attack - recipe (cheap, safe) then tokenizer (big).

---

<!-- New entries appended below as experiments run. -->

## Run 1 - Modern training recipe
- Hypothesis: baseline under-optimized (loss still falling at step 2000). A proper
  LR schedule plus higher peak LR should use the fixed 2000 steps far better.
- Changed (vs baseline): plain Adam to AdamW with weight decay 0.1, constant LR 3e-4
  to peak 1e-3 with 100-step linear warmup and cosine decay to 10 percent, added
  gradient clipping at norm 1.0, betas (0.9, 0.95). Nothing else touched.
- dev bpb: 2.3718 -> 2.2516 (down 0.1202). Train loss 1.73 -> 1.61.
- Observation: loss is now flat around 1.60 by step 2000 (converged this run), where
  baseline was cut off mid-descent. LR annealed from 1e-3 to 1e-4 as planned.
- Conclusion: recipe was a real but bounded win. The model now fits the byte-level
  data about as well as it can in 2000 steps. Remaining gains must come from the
  tokenizer (the Hindi third is still 3 bytes per char) and from param reallocation.

## Run 2 - BPE tokenizer (vocab 2048)
- Hypothesis: byte-level spends ~1/3 of compute predicting Hindi 3 bytes at a time.
  A BPE trained on the corpus compresses bytes, so the fixed 2000 steps cover more
  text and block_size spans more real context. bpb should drop even though per-token
  loss rises, because bpb divides bits by bytes-per-token.
- Changed (vs Run 1): replaced byte tokenizer with corpus-trained BPE, vocab 2048,
  stdlib-only trainer (indexed incremental merges), lossless with byte fallback.
  Merges saved to bpe_merges.json, loaded relative to tokenizer file. Model config
  unchanged (untied, n_embd 160) to isolate the tokenizer effect.
- Compression: corpus 7,318,592 bytes -> 2,149,341 tokens (3.405 bytes/token).
  dev_eval 3.434 bytes/token. Roundtrip verified lossless on train, dev, and
  arbitrary utf-8 (emoji, CJK, control bytes).
- dev bpb: 2.2516 -> 1.9993 (down 0.2523). Broke below 2.0.
- n_params 1,913,280 (under 2M cap). Per-token loss now ~4.48 (was 1.60) because
  each token carries more information and vocab is larger, but bytes-per-token more
  than compensates.
- Observation: loss still declining at step 2000 (fewer tokens per step means less
  total corpus seen), so the model is mildly under-trained again.
- Conclusion: tokenizer is the dominant lever (biggest single drop so far). Next:
  tie weights to free ~328K params for capacity, and consider larger batch to see
  more corpus per step.

## Run 3 - Tie weights + add depth (5 layers) [REVERTED]
- Hypothesis: Run 2 left the model mildly under-trained and we had param headroom.
  Tie input/output embeddings to free ~328K params, spend them on a 5th layer for
  more capacity. Expected bpb to drop.
- Changed (vs Run 2): tie_weights True, n_layer 4 -> 5. Params 1,913,280 -> 1,894,880.
  Tokenizer and recipe unchanged.
- dev bpb: 1.9993 -> 2.0451 (UP 0.0458). Train loss ended higher: 4.65 vs 4.48, and
  the curve was still dropping steeply at step 2000 (vs nearly flat for 4-layer).
- Diagnosis: this FAILED because the binding constraint at 2000 CPU steps is
  optimization, not capacity. A deeper model has more to learn per step and did not
  converge inside the step budget, so it underfit worse. Tying removed some output
  flexibility on top of that.
- Conclusion: reverted both changes. Key takeaway that reshapes strategy: we are
  STEP-LIMITED, not PARAM-LIMITED. Adding capacity hurts. Remaining gains should come
  from faster convergence within 2000 steps (larger effective batch, better init,
  LR tuning), not from a bigger model.
## Run 4 - Batch size sweep (8 -> 24 -> 32)
- Hypothesis: Run 3 proved we are step-limited, not param-limited. Larger batches give
  each of the fixed 2000 steps a cleaner, lower-variance gradient, so the model
  converges better inside the step budget without adding any parameters.
- Changed (vs Run 2 best): batch 8 -> 24 -> 32. Everything else identical (BPE 2048,
  4 layer untied, cosine recipe).
- dev bpb: batch 8 = 1.9993 (ref) -> batch 24 = 1.8634 -> batch 32 = 1.8267.
  Train loss fell 4.48 -> 4.01 -> 3.83 across the sweep.
- Observation: monotonic improvement, no plateau yet. Loss still dropping at step 2000
  even at batch 32, so convergence is still not saturated. Per-step time rose
  20 -> 59 -> 80 ms; batch 32 run took ~3 min (still within time budget).
- Conclusion: batch size is the second major lever after the tokenizer, and it
  confirms the step-limited diagnosis. Best so far batch 32 = 1.8267. Next: push batch
  further, paired with a higher peak LR to keep converging inside 2000 steps.

## Run 5 - Batch 48 + higher peak LR (1.5e-3)
- Hypothesis: at batch 32 the loss was still falling at step 2000, so we had not
  saturated convergence. Push batch to 48 for even cleaner gradients, and raise peak
  LR 1e-3 -> 1.5e-3 so the model travels further within the fixed 2000 steps. Pairing
  batch and LR is the standard way to keep a large-batch run converging.
- Changed (vs Run 4 best): batch 32 -> 48, peak LR 1e-3 -> 1.5e-3. Tokenizer, model,
  schedule shape all unchanged.
- dev bpb: 1.8267 -> 1.7262 (down 0.1005). Train loss 3.83 -> 3.40.
- Observation: early loss curve stayed smooth (no spikes at steps 100-300), so 1.5e-3
  was stable at this batch. Loss still declining slightly at step 2000, so not fully
  saturated. Run took ~4.5 min (per-step ~120 ms).
- Conclusion: the batch-plus-LR axis is the dominant remaining lever now that the
  tokenizer is fixed. Best so far 1.7262. This confirms twice over that the model is
  optimization-limited: everything that helps it converge more inside 2000 steps wins.

## Run 6 - Batch 64 + peak LR 2e-3
- Hypothesis: batch 48 still had loss falling at step 2000; push batch to 64 and LR to
  2e-3 to extract the remaining convergence. Expect a smaller but real drop.
- Changed (vs Run 5): batch 48 -> 64, peak LR 1.5e-3 -> 2e-3.
- dev bpb: 1.7262 -> 1.6947 (down 0.0315). Train loss 3.40 -> 3.16, curve stable.
- Observation: still slightly descending at step 2000, so convergence is not fully
  saturated, but run time hit 406s (~6.8 min, 200 ms/step). This is the practical
  ceiling on batch under the time budget; further batch increases are not worth the
  wall-clock cost even though bpb would likely keep inching down.
- Conclusion: locked batch 64 + lr 2e-3 as the convergence config. Diminishing returns
  on this axis. Best 1.6947. Next lever is the tokenizer allocation, not more compute.

## Run 7 - Script-aware BPE with protected Hindi budget (best so far)
- Motivation (measured): on dev, plain BPE compressed English to ~3.16 chars/tok but
  Hindi to only ~1.58 chars/tok. BPE greedily takes the most frequent byte-pairs, and
  since English is 67 percent of the corpus, English pairs win almost every merge and
  Devanagari is starved, even though Hindi is 33 percent of the byte budget and costs
  3 bytes per char (where compression pays most).
- First attempt (naive): reallocate merges toward Hindi at vocab 2048, hindi_frac 0.45.
  Hindi improved to ~1.75 chars/tok but English got worse, total tokens 46365 -> 46442.
  Zero-sum, no bpb gain, not shipped.
- Fix: do not steal English merges. Expand budget to vocab 2304 (untied, 1,995,200
  params, under 2M cap) and give Devanagari a protected pass (hindi_frac 0.42). English
  merges preserved, Hindi added on top.
- Compression (dev): total tokens 46365 -> 45073 (down 2.8 percent). English held,
  Hindi 1.58 -> 1.74 chars/tok. Lossless on train, dev, arbitrary utf-8.
- dev bpb: 1.6947 -> 1.6933 (new best). Overall vs baseline: 2.3718 -> 1.6933, 28.6
  percent reduction, within 2000 steps and under 2M params.
