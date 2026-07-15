"""BPE tokenizer trained on the provided corpus. stdlib only (no external deps).

Interface required by train.py / evaluate.py:
  load() -> object with .encode(str)->list[int], .decode(list[int])->str, .vocab_size
Guarantees:
  * lossless: decode(encode(text)) == text exactly (graders verify this round-trip)
  * encodes ARBITRARY utf-8 via byte-level base (every byte 0..255 is a token id),
    so unseen characters always fall back to raw bytes and never fail.
  * merges are saved next to this file (bpe_merges.json) and loaded with no internet;
    paths resolve relative to __file__ so grading (cwd = submission folder) works.
"""
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_MERGES_PATH = os.path.join(_HERE, "bpe_merges.json")
_SPLIT = re.compile(r"\s*\S+|\s+")


class BPETokenizer:
    def __init__(self, merges):
        self.merges = {(a, b): nid for a, b, nid in merges}
        self.vocab_size = 256 + len(merges)
        self._id2bytes = {i: bytes([i]) for i in range(256)}
        for a, b, nid in merges:
            self._id2bytes[nid] = self._id2bytes[a] + self._id2bytes[b]
        self._cache = {}

    def _encode_chunk(self, wb):
        cached = self._cache.get(wb)
        if cached is not None:
            return cached
        ids = list(wb)
        merges = self.merges
        while len(ids) >= 2:
            best = None
            for j in range(len(ids) - 1):
                p = (ids[j], ids[j + 1])
                r = merges.get(p)
                if r is not None and (best is None or r < best[0]):
                    best = (r, j, p)
            if best is None:
                break
            _, j, p = best
            ids = ids[:j] + [merges[p]] + ids[j + 2:]
        self._cache[wb] = ids
        return ids

    def encode(self, text):
        out = []
        for chunk in _SPLIT.findall(text):
            out.extend(self._encode_chunk(chunk.encode("utf-8")))
        return out

    def decode(self, ids):
        return b"".join(self._id2bytes[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path=_MERGES_PATH):
        merges = [[a, b, nid] for (a, b), nid in
                  sorted(self.merges.items(), key=lambda kv: kv[1])]
        with open(path, "w") as f:
            json.dump({"type": "bpe", "merges": merges}, f)


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


def load(path=None):
    path = path or _MERGES_PATH
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if data.get("type") == "bpe":
            return BPETokenizer(data["merges"])
    return ByteTokenizer()
