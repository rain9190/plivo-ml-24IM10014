"""Train the BPE tokenizer on the corpus and save bpe_merges.json next to
tokenizer.py. stdlib only. Fast: indexed incremental merges over unique words.

    python train_bpe.py --data ../data/train_corpus.txt --vocab 2048
"""
import argparse
import collections
import json
import os
import re
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPLIT = re.compile(r"\s*\S+|\s+")


def train(text, vocab_size, log=True):
    freq = collections.Counter(_SPLIT.findall(text))
    wlist = [list(w.encode("utf-8")) for w in freq]
    wfreq = [freq[w] for w in freq]
    pair_count = collections.Counter()
    pair_where = collections.defaultdict(set)
    for wi, word in enumerate(wlist):
        c = wfreq[wi]
        for a, b in zip(word, word[1:]):
            pair_count[(a, b)] += c
            pair_where[(a, b)].add(wi)
    merges = []
    t0 = time.time()
    for m in range(vocab_size - 256):
        if not pair_count:
            break
        pair = max(pair_count, key=pair_count.get)
        if pair_count[pair] <= 0:
            break
        new_id = 256 + m
        merges.append([pair[0], pair[1], new_id])
        for wi in list(pair_where[pair]):
            word = wlist[wi]
            c = wfreq[wi]
            for a, b in zip(word, word[1:]):
                pair_count[(a, b)] -= c
            nw = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    nw.append(new_id); i += 2
                else:
                    nw.append(word[i]); i += 1
            wlist[wi] = nw
            for a, b in zip(nw, nw[1:]):
                pair_count[(a, b)] += c
                pair_where[(a, b)].add(wi)
        del pair_count[pair]
        pair_where[pair] = set()
        if log and m % 256 == 0:
            print(f"  merge {m}/{vocab_size-256}  {time.time()-t0:.0f}s", flush=True)
    return merges, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--out", default=os.path.join(_HERE, "bpe_merges.json"))
    args = ap.parse_args()
    text = open(args.data, encoding="utf-8").read()
    merges, dt = train(text, args.vocab)
    with open(args.out, "w") as f:
        json.dump({"type": "bpe", "merges": merges}, f)
    print(f"trained vocab={256+len(merges)} ({len(merges)} merges) in {dt:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
