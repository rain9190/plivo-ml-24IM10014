"""Script-aware BPE. Trains merges with a PROTECTED budget for Devanagari, because
measured per-script compression showed Hindi at ~1.58 chars/token vs English ~3.16:
plain BPE lets English (67% of bytes) dominate the merge pool and starves Hindi,
which is exactly where bytes are most expensive (3 bytes/char). We reserve a share
of merges for Devanagari-containing chunks so Hindi gets compression proportional to
its DIFFICULTY, not just its frequency.

stdlib only. Output vocab is a single flat merge list -> fully compatible with the
existing lossless tokenizer.py (byte fallback preserved).

    python train_bpe_sa.py --data ../data/train_corpus.txt --vocab 2048 --hindi_frac 0.45
"""
import argparse, collections, json, os, re, time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPLIT = re.compile(r"\s*\S+|\s+")


def is_deva_chunk(s):
    return any('\u0900' <= ch <= '\u097f' for ch in s)


def _train_on(chunks_freq, n_merges, start_id, log_tag=""):
    """Run indexed incremental BPE on a {word_tuple: freq} set. Returns merge list."""
    wlist = [list(w) for w in chunks_freq]
    wfreq = [chunks_freq[w] for w in chunks_freq]
    pair_count = collections.Counter()
    pair_where = collections.defaultdict(set)
    for wi, word in enumerate(wlist):
        c = wfreq[wi]
        for a, b in zip(word, word[1:]):
            pair_count[(a, b)] += c
            pair_where[(a, b)].add(wi)
    merges = []
    for m in range(n_merges):
        if not pair_count:
            break
        pair = max(pair_count, key=pair_count.get)
        if pair_count[pair] <= 0:
            break
        new_id = start_id + m
        merges.append([pair[0], pair[1], new_id])
        for wi in list(pair_where[pair]):
            word = wlist[wi]; c = wfreq[wi]
            for a, b in zip(word, word[1:]):
                pair_count[(a, b)] -= c
            nw = []; i = 0
            while i < len(word):
                if i < len(word)-1 and word[i]==pair[0] and word[i+1]==pair[1]:
                    nw.append(new_id); i += 2
                else:
                    nw.append(word[i]); i += 1
            wlist[wi] = nw
            for a, b in zip(nw, nw[1:]):
                pair_count[(a, b)] += c
                pair_where[(a, b)].add(wi)
        del pair_count[pair]; pair_where[pair] = set()
    return merges


def train(text, vocab_size, hindi_frac):
    freq = collections.Counter(_SPLIT.findall(text))
    deva = collections.Counter()
    genl = collections.Counter()
    for w, c in freq.items():
        wb = tuple(w.encode("utf-8"))
        if is_deva_chunk(w):
            deva[wb] += c
        else:
            genl[wb] += c
    total_merges = vocab_size - 256
    n_hi = int(total_merges * hindi_frac)
    n_gen = total_merges - n_hi

    t0 = time.time()
    # Pass 1: protected Devanagari merges first (ids 256..256+n_hi-1)
    hi_merges = _train_on(deva, n_hi, 256)
    used = len(hi_merges)
    # Pass 2: general merges continue the id space. Apply learned Hindi merges to the
    # general set first is unnecessary (disjoint byte ranges), so train general fresh.
    gen_merges = _train_on(genl, n_gen, 256 + used)
    merges = hi_merges + gen_merges
    print(f"  hindi merges: {len(hi_merges)}  general merges: {len(gen_merges)}  "
          f"({time.time()-t0:.0f}s)")
    return merges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--hindi_frac", type=float, default=0.45)
    ap.add_argument("--out", default=os.path.join(_HERE, "bpe_merges.json"))
    args = ap.parse_args()
    text = open(args.data, encoding="utf-8").read()
    merges = train(text, args.vocab, args.hindi_frac)
    with open(args.out, "w") as f:
        json.dump({"type": "bpe", "merges": merges}, f)
    print(f"trained vocab={256+len(merges)} ({len(merges)} merges, "
          f"hindi_frac={args.hindi_frac}) -> {args.out}")


if __name__ == "__main__":
    main()
