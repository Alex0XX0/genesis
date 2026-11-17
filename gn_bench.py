#!/usr/bin/env python3
# GN-BENCH: стенд для сравнения архитектур. Судья — bpb (бит/байт) на отложенном корпусе по минутам CPU и по байтам.
# python3 gn_bench.py base|muon [секунд] — результат в bench/<вариант>.jsonl
import os, sys, json, time, math, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gn_net as N

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "bench"); os.makedirs(OUT, exist_ok=True)
F = np.float32

def corpus(n_bytes=1_500_000):
    cp = os.path.join(OUT, "corpus.bin")
    if os.path.exists(cp): return np.frombuffer(open(cp, "rb").read(), dtype=np.uint8)
    parts = [e["text"] for e in json.load(open(os.path.join(HERE, "mind.json")))["ep"]]
    ru = "\n".join(parts).encode("utf-8")
    code = b"".join(open(f, "rb").read() for f in sorted(glob.glob("/usr/lib/python3.12/*.py"))[:60])
    ru, code = ru[: n_bytes * 2 // 3], code[: n_bytes // 3]
    b = (ru + b"\n" + code); open(cp, "wb").write(b)
    return np.frombuffer(b, dtype=np.uint8)

def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (np.linalg.norm(G) + 1e-7); tr = X.shape[0] > X.shape[1]
    if tr: X = X.T
    for _ in range(steps):
        A = N.mm(X, X, tb=True); Bm = b * A + c * N.mm(A, A); X = a * X + N.mm(Bm, X)
    return (X.T if tr else X).astype(F)

def step(net, ids, tgt, w, lr, opt, t_state):
    for p in net.P.values(): p.g = None
    l, nll = net.loss(ids, tgt, w); N.backward(l)
    gs = {n: p.g for n, p in net.P.items() if p.g is not None}
    gn = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in gs.values()))
    sc = min(1.0, 1.0 / (gn + 1e-9)); net.t += 1; t = net.t; b1, b2 = 0.9, 0.95
    for n, g in gs.items():
        p = net.P[n]; g = g * sc
        if n not in net.m or net.m[n].shape != p.d.shape: net.m[n] = np.zeros_like(p.d); net.v[n] = np.zeros_like(p.d)
        m, v = net.m[n], net.v[n]
        if opt == "muon" and p.d.ndim == 2 and n not in ("E", "pos"):
            m *= 0.95; m += g; u = newton_schulz(g + 0.95 * m)       # нестеровский импульс
            u *= 0.2 * math.sqrt(max(p.d.shape))                      # согласование RMS с Adam
            p.d -= lr * 0.01 * p.d; p.d -= lr * 3.0 * u               # Muon допускает больший lr
        else:
            m *= b1; m += (1 - b1) * g; v *= b2; v += (1 - b2) * g * g
            upd = (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8)
            if p.d.ndim >= 2 and n not in ("E", "pos"): p.d -= lr * 0.01 * p.d
            p.d -= lr * upd
    return float(l.d)

def evaluate(net, val, ctx, n=24):
    tot = cnt = 0.0; rng = np.random.default_rng(1)
    with N.no_grad():
        for _ in range(n // 8):
            ix = rng.integers(0, len(val) - ctx - 1, 8)
            x = np.stack([val[i:i + ctx] for i in ix]).astype(np.int64); y = np.stack([val[i + 1:i + ctx + 1] for i in ix]).astype(np.int64)
            _, nll = net.loss(x, y, np.ones_like(x, dtype=F)); tot += float(nll.sum()); cnt += nll.size
    return tot / cnt / math.log(2)

def run(variant, secs, d=128, L=3, ctx=128, B=16, lr=1e-3, seed=0):
    data = corpus(); k = int(len(data) * 0.95); tr, val = data[:k], data[k:]
    net = N.Net(d=d, L=L, H=4, ctx=ctx, vcap=256, seed=seed); net.set_vocab(256)
    opt = "muon" if variant.startswith("muon") else "adam"
    rng = np.random.default_rng(seed); t0 = time.time(); seen = 0; nxt = 0; log = open(os.path.join(OUT, variant + ".jsonl"), "w")
    print(variant, "параметров %.2fM, корпус %d байт" % (net.n_params() / 1e6, len(data)), flush=True)
    while True:
        el = time.time() - t0
        if el >= nxt or el >= secs:
            bpb = evaluate(net, val, ctx); r = {"t": round(el, 1), "step": net.t, "bytes": seen, "bpb": round(bpb, 4)}
            log.write(json.dumps(r) + "\n"); log.flush(); print(r, flush=True); nxt += 20
            if el >= secs: break
        ix = rng.integers(0, len(tr) - ctx - 1, B)
        x = np.stack([tr[i:i + ctx] for i in ix]).astype(np.int64); y = np.stack([tr[i + 1:i + ctx + 1] for i in ix]).astype(np.int64)
        warm = min(1.0, (net.t + 1) / 30); step(net, x, y, np.ones_like(x, dtype=F), lr * warm, opt, None); seen += x.size

if __name__ == "__main__":
    os.environ.setdefault("GEN_THREADS", "4")
    run(sys.argv[1] if len(sys.argv) > 1 else "base", float(sys.argv[2]) if len(sys.argv) > 2 else 240)
