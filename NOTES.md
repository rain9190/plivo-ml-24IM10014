# NOTES - best configuration

The final model reaches dev bpb 1.6933 (down from the 2.3718 byte-level baseline, a
28.6 percent reduction) within the 2000-step and 2M-parameter caps on CPU.

Configuration: a script-aware BPE tokenizer (vocab 2304) trained on the corpus, feeding
a 4-layer, 4-head, 160-dim GPT (untied, 1,995,200 params), trained for 2000 steps with
AdamW (weight decay 0.1), a cosine learning-rate schedule with 100-step warmup and peak
LR 2e-3, gradient clipping at norm 1.0, and batch size 64.

Why it works: two levers dominate under these caps. First, the tokenizer — byte-level
spends three tokens per Devanagari character and Hindi is a third of the byte budget, so
a corpus-trained BPE that compresses ~3.4 bytes per token lets the fixed 2000 steps see
far more text; making it script-aware (a protected Hindi merge budget) recovers
compression on the part plain BPE starves, since English frequency otherwise wins every
merge. Second, the run is optimization-limited, not capacity-limited (proven when adding
a 5th layer made bpb worse): larger batches give each of the fixed steps a cleaner
gradient, so raising batch 8 to 64 with a paired higher peak LR converges much better in
the same budget. Everything that helped attacked either bytes-per-token or
convergence-per-step; adding parameters did not.
