#!/usr/bin/env python3
# GN-TOK — растущий токенизатор: символы добавляются по мере встречи, BPE-слияния учатся из опыта,
# модальные коды (картинки/звук) регистрируются как атомарные токены. Всё в одном словаре с текстом.
import json, os, re
from collections import Counter

_WORD = re.compile(r"\s?\w+|\s?[^\w\s]|\s")          # пословный режим: слово вместе с пробелом перед ним, знак, пробел
SPECIAL = ["<pad>", "<bos>", "<eos>", "<img>", "</img>", "<aud>", "</aud>", "<vid>", "</vid>", "<row>", "<think>", "</think>"]

class Tok:
    def __init__(self, vcap=8192):
        self.vcap = vcap
        self.toks, self.idx, self.atomic = [], {}, set()
        self.merges = {}                  # (a,b) -> (rank, new_id)
        self.wm = False; self._wc = {}    # wm: слияния только внутри слов, кодирование через кэш слов (быстро при тысячах слияний)
        for s in SPECIAL: self._add(s, atomic=True)

    def _add(self, s, atomic=False):
        if s in self.idx: return self.idx[s]
        if len(self.toks) >= self.vcap: return None
        i = len(self.toks); self.toks.append(s); self.idx[s] = i
        if atomic: self.atomic.add(i)
        return i

    def __len__(self): return len(self.toks)
    def id(self, s): return self.idx[s]
    PAD, BOS, EOS = 0, 1, 2

    def code(self, prefix, n):
        """Атомарный токен модального кода: ⟦i17⟧ (картинка), ⟦a5⟧ (звук). Возвращает id (или None, если нет места)."""
        return self._add("⟦%s%d⟧" % (prefix, n), atomic=True)

    def base(self, text, grow=True):
        out = []
        for ch in text:
            i = self.idx.get(ch)
            if i is None and grow: i = self._add(ch)
            if i is not None: out.append(i)
        return out

    def encode(self, text, grow=True):
        if not self.wm: return self.apply_merges(self.base(text, grow))
        out = []; c = self._wc
        for w in _WORD.findall(text):
            ids = c.get(w)
            if ids is None:
                b = self.base(w, grow); ids = self.apply_merges(b)
                if len(b) == len(w) and len(c) < 300000: c[w] = ids
            out += ids
        return out

    def apply_merges(self, ids):
        if not self.merges or len(ids) < 2: return ids
        ids = list(ids)
        while True:
            best, bi = None, -1
            for i in range(len(ids) - 1):
                m = self.merges.get((ids[i], ids[i + 1]))
                if m and (best is None or m[0] < best[0]): best, bi = m, i
            if best is None: return ids
            a, b = ids[bi], ids[bi + 1]; out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b: out.append(best[1]); i += 2
                else: out.append(ids[i]); i += 1
            ids = out

    def decode(self, ids):
        return "".join(self.toks[i] if i < len(self.toks) else "" for i in ids if i > 2)

    def learn_merges(self, seqs, n=16, min_count=4):
        """Учит до n новых слияний на текущем корпусе. Возвращает [(new_id,[a,b])] для инициализации эмбеддингов."""
        made = []
        for _ in range(n):
            c = Counter()
            for s in seqs:
                for a, b in zip(s, s[1:]):
                    if a not in self.atomic and b not in self.atomic: c[(a, b)] += 1
            if not c: break
            (a, b), k = c.most_common(1)[0]
            if k < min_count: break
            nid = self._add(self.toks[a] + self.toks[b])
            if nid is None: break
            self.merges[(a, b)] = (len(self.merges), nid); made.append((nid, [a, b]))
            for s in seqs:
                i, o = 0, []
                while i < len(s):
                    if i < len(s) - 1 and s[i] == a and s[i + 1] == b: o.append(nid); i += 2
                    else: o.append(s[i]); i += 1
                s[:] = o
        return made

    def save(self, path):
        d = {"toks": self.toks, "atomic": sorted(self.atomic), "merges": [[a, b, r, n] for (a, b), (r, n) in self.merges.items()], "vcap": self.vcap, "wm": self.wm}
        tmp = path + ".tmp"; json.dump(d, open(tmp, "w", encoding="utf-8"), ensure_ascii=False); os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        d = json.load(open(path, encoding="utf-8")); t = cls(d["vcap"]); t.toks = d["toks"]
        t.idx = {s: i for i, s in enumerate(t.toks)}; t.atomic = set(d["atomic"])
        t.merges = {(a, b): (r, n) for a, b, r, n in d["merges"]}; t.wm = bool(d.get("wm", False)); return t
