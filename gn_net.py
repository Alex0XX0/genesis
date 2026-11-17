#!/usr/bin/env python3
# GN-NET — собственная нейросеть Genesis, с нуля, только numpy.
# Автоград + causal-трансформер + Adam. ДИНАМИКА: рост без потери функции (новые слои / нейроны),
# прореживание слабых нейронов, расширение словаря (ёмкость заложена, активная часть растёт).
import json, math, os, time
import numpy as np

F = np.float32
_GRAD = [True]
try:                                   # системный BLAS здесь референсный (6 ГФЛОПС) — берём быстрый gemm из OpenCV
    import cv2
    cv2.setNumThreads(int(os.environ.get("GEN_THREADS", "4")))
except Exception: cv2 = None

def mm(a, b, ta=False, tb=False):
    """2D-умножение (с транспонированием): OpenCV gemm для float32, иначе numpy."""
    if cv2 is not None and a.dtype == np.float32 and b.dtype == np.float32:
        return cv2.gemm(np.ascontiguousarray(a), np.ascontiguousarray(b), 1.0, None, 0, flags=(1 if ta else 0) | (2 if tb else 0))
    return (a.T if ta else a) @ (b.T if tb else b)

def bmm(a, b, ta=False, tb=False):
    """Батч малых умножений (внимание): цикл по (B,H) через быстрый gemm вместо референсного np.matmul. Те же произведения float32."""
    if cv2 is None or a.dtype != np.float32 or b.dtype != np.float32 or a.ndim < 3:
        return np.matmul(np.swapaxes(a, -1, -2) if ta else a, np.swapaxes(b, -1, -2) if tb else b)
    lead = a.shape[:-2]; A = a.reshape((-1,) + a.shape[-2:]); Bm = b.reshape((-1,) + b.shape[-2:]); fl = (1 if ta else 0) | (2 if tb else 0)
    m = a.shape[-1] if ta else a.shape[-2]; n = b.shape[-2] if tb else b.shape[-1]; out = np.empty((A.shape[0], m, n), np.float32)
    for i in range(A.shape[0]): out[i] = cv2.gemm(A[i], Bm[i], 1.0, None, 0, flags=fl)
    return out.reshape(lead + (m, n))

class no_grad:
    def __enter__(self): self.p = _GRAD[0]; _GRAD[0] = False
    def __exit__(self, *a): _GRAD[0] = self.p

class T:
    __slots__ = ("d", "g", "_p", "_b", "req")
    def __init__(s, d, p=(), b=None, req=False): s.d, s.g, s._p, s._b, s.req = d, None, p, b, req

def _mk(d, p, b):
    if _GRAD[0] and any(x.req or x._b is not None for x in p): return T(d, p, b)
    return T(d)

def _acc(t, g):
    if t.req or t._b is not None: t.g = g if t.g is None else t.g + g

def _unb(g, shape):
    while g.ndim > len(shape): g = g.sum(0)
    for i, n in enumerate(shape):
        if n == 1 and g.shape[i] != 1: g = g.sum(i, keepdims=True)
    return g

def backward(loss):
    order, seen, st = [], set(), [(loss, False)]
    while st:
        n, done = st.pop()
        if done: order.append(n); continue
        if id(n) in seen: continue
        seen.add(id(n)); st.append((n, True))
        for p in n._p:
            if id(p) not in seen: st.append((p, False))
    loss.g = np.ones_like(loss.d)
    for n in reversed(order):
        if n._b is not None and n.g is not None: n._b(n.g)

# ---------- операции ----------
def add(a, b):
    o = _mk(a.d + b.d, (a, b), None)
    if o._p: o._b = lambda g: (_acc(a, _unb(g, a.d.shape)), _acc(b, _unb(g, b.d.shape)))
    return o

def matmul(a, b):
    fast = b.d.ndim == 2 and a.d.ndim > 2          # (B,T,k) @ (k,n): один большой BLAS-вызов вместо батча малых
    if fast: k = a.d.shape[-1]; n = b.d.shape[-1]; d = mm(a.d.reshape(-1, k), b.d).reshape(a.d.shape[:-1] + (n,))
    else: d = bmm(a.d, b.d)
    o = _mk(d, (a, b), None)
    if o._p:
        if fast:
            def bk(g):
                g2 = g.reshape(-1, n)
                _acc(a, mm(g2, b.d, tb=True).reshape(a.d.shape)); _acc(b, mm(a.d.reshape(-1, k), g2, ta=True))
        else:
            def bk(g):
                _acc(a, _unb(bmm(g, b.d, tb=True), a.d.shape))
                _acc(b, _unb(bmm(a.d, g, ta=True), b.d.shape))
        o._b = bk
    return o

def scale(a, c):
    o = _mk(a.d * F(c), (a,), None)
    if o._p: o._b = lambda g: _acc(a, g * F(c))
    return o

def relu2(x):
    xd = x.d; r = np.maximum(xd, F(0)); o = _mk(r * r, (x,), None)
    if o._p:
        def bk(g): _acc(x, g * (F(2) * r))
        o._b = bk
    return o

def gelu(x):
    c = F(0.7978845608); c3 = F(0.7978845608 * 0.044715); xd = x.d; x2 = xd * xd
    u = x2 * c3; u += c; u *= xd; t = np.tanh(u, out=u)                  # та же формула GELU (tanh), операции на месте: меньше временных массивов
    y = t + F(1); y *= xd; y *= F(0.5); o = _mk(y, (x,), None)
    if o._p:
        def bk(g):
            k = x2 * (F(3) * c3); k += c; k *= xd; w = t * t; np.subtract(F(1), w, out=w); k *= w; k += t; k += F(1); k *= F(0.5); k *= g
            _acc(x, k)
        o._b = bk
    return o

def layernorm(x, gm, bt, eps=1e-5):
    mu = x.d.mean(-1, keepdims=True); xc = x.d - mu
    inv = 1.0 / np.sqrt((xc * xc).mean(-1, keepdims=True) + eps); xh = xc * inv
    o = _mk(xh * gm.d + bt.d, (x, gm, bt), None)
    if o._p:
        def bk(g):
            _acc(gm, _unb(g * xh, gm.d.shape)); _acc(bt, _unb(g, bt.d.shape))
            dxh = g * gm.d; n = x.d.shape[-1]
            _acc(x, inv * (dxh - dxh.mean(-1, keepdims=True) - xh * (dxh * xh).mean(-1, keepdims=True)))
        o._b = bk
    return o

def softmax(x, mask=None):
    z = x.d if mask is None else x.d + mask
    z = z - z.max(-1, keepdims=True); e = np.exp(z); y = e / e.sum(-1, keepdims=True)
    o = _mk(y, (x,), None)
    if o._p: o._b = lambda g: _acc(x, y * (g - (g * y).sum(-1, keepdims=True)))
    return o

def reshape(x, shape):
    o = _mk(x.d.reshape(shape), (x,), None)
    if o._p: o._b = lambda g: _acc(x, g.reshape(x.d.shape))
    return o

def transpose(x, axes):
    inv = np.argsort(axes)
    o = _mk(np.transpose(x.d, axes), (x,), None)
    if o._p: o._b = lambda g: _acc(x, np.transpose(g, inv))
    return o

def embed(E, ids):
    o = _mk(E.d[ids], (E,), None)
    if o._p:
        def bk(g):
            gE = np.zeros_like(E.d); np.add.at(gE, ids, g); _acc(E, gE)
        o._b = bk
    return o

def tied_logits(x, E, V):
    """logits = x @ E[:V].T — выходные веса общие со входными эмбеддингами."""
    d = x.d.shape[-1]; x2 = x.d.reshape(-1, d)
    o = _mk(mm(x2, E.d[:V], tb=True).reshape(x.d.shape[:-1] + (V,)), (x, E), None)
    if o._p:
        def bk(g):
            g2 = g.reshape(-1, V)
            _acc(x, mm(g2, E.d[:V]).reshape(x.d.shape))
            gE = np.zeros_like(E.d); gE[:V] = mm(g2, x2, ta=True); _acc(E, gE)
        o._b = bk
    return o

def cross_entropy(logits, tgt, w):
    """logits (N,V); tgt (N,); w (N,) веса токенов. Возвращает (T-скаляр, вектор nll по токенам)."""
    z = logits.d - logits.d.max(-1, keepdims=True); lse = np.log(np.exp(z).sum(-1)); n = len(tgt)
    nll = lse - z[np.arange(n), tgt]; ws = max(float(w.sum()), 1e-6)
    o = _mk(np.array((nll * w).sum() / ws, dtype=F), (logits,), None)
    if o._p:
        def bk(g):
            p = np.exp(z) / np.exp(z).sum(-1, keepdims=True); p[np.arange(n), tgt] -= 1
            _acc(logits, p * (w / ws)[:, None] * g)
        o._b = bk
    return o, nll

_PE = [(8.28721201814563, -23.595886519098837, 17.300387312530933), (4.107059111542203, -2.9478499167379106, 0.5448431082926601), (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
       (3.3184196573706015, -2.488488024314874, 0.51004894012372), (2.300652019954817, -1.6689039845747493, 0.4188073119525673)]   # Polar Express (arXiv 2505.16932)
NS_MODE = os.environ.get("GEN_NS", "cur")

def newton_schulz(G, steps=5):
    """Ортогонализация обновления (Muon). pe: коэффициенты Polar Express — на тех же 5 итерациях точнее, чем классические (cur)."""
    pe = NS_MODE == "pe"; coefs = _PE[:steps] if pe else [(3.4445, -4.7750, 2.0315)] * steps
    X = G / (np.linalg.norm(G) * (1.01 if pe else 1.0) + 1e-7); tr = X.shape[0] > X.shape[1]
    if tr: X = X.T
    for a, b, c in coefs:
        A = mm(X, X, tb=True); Bm = b * A + c * mm(A, A); X = a * X + mm(Bm, X)
    return (X.T if tr else X).astype(F)

OPT = os.environ.get("GEN_OPT", "muon")           # muon на стенде выиграл у Adam: ошибка 2.96 против 3.21 за 120 с

# ---------- сеть ----------
_ACT = {"gelu": gelu, "relu2": relu2}
NS = 64          # размер вектора состояния (эмоции, тело, внимание, последнее дело): подаётся в сеть вместе с текстом

class Net:
    def __init__(self, d=128, L=2, H=4, ctx=128, vcap=8192, seed=0):
        self.cfg = {"d": d, "H": H, "ctx": ctx, "vcap": vcap}
        self.rng = np.random.default_rng(seed)
        self.P, self.layers, self.uid, self.V = {}, [], 0, 0
        self.m, self.v, self.t = {}, {}, 0
        self.P["E"] = self._p(self.rng.normal(0, 0.02, (vcap, d)))
        self.P["pos"] = self._p(self.rng.normal(0, 0.02, (ctx, d)))
        self.P["Ws"] = self._p(np.zeros((NS, d)))          # проектор состояния в «настроение» токена; нулевой при рождении: функция сети не меняется
        self.P["lnf.g"] = self._p(np.ones(d)); self.P["lnf.b"] = self._p(np.zeros(d))
        for _ in range(L): self._new_layer(zero_out=False)
        self._mask = {}

    @staticmethod
    def _p(a): return T(np.asarray(a, dtype=F), req=True)

    def _new_layer(self, zero_out, hidden=None):
        d = self.cfg["d"]; h = hidden or 4 * d; k = "l%d" % self.uid; self.uid += 1; r = self.rng
        s = 0.02; so = 0.0 if zero_out else 0.02 / math.sqrt(2 * max(1, len(self.layers) + 1))
        for nm in ("q", "k", "v"): self.P["%s.W%s" % (k, nm)] = self._p(r.normal(0, s, (d, d)))
        self.P[k + ".Wo"] = self._p(r.normal(0, 1, (d, d)) * so if so else np.zeros((d, d)))
        self.P[k + ".ln1.g"] = self._p(np.ones(d)); self.P[k + ".ln1.b"] = self._p(np.zeros(d))
        self.P[k + ".ln2.g"] = self._p(np.ones(d)); self.P[k + ".ln2.b"] = self._p(np.zeros(d))
        self.P[k + ".W1"] = self._p(r.normal(0, s, (d, h))); self.P[k + ".b1"] = self._p(np.zeros(h))
        self.P[k + ".W2"] = self._p(r.normal(0, 1, (h, d)) * so if so else np.zeros((h, d))); self.P[k + ".b2"] = self._p(np.zeros(d))
        self.layers.append(k)

    # --- динамика архитектуры ---
    def grow_layer(self):
        """Новый слой с нулевым выходом: функция сети НЕ меняется, ёмкость растёт."""
        self._new_layer(zero_out=True, hidden=self.P[self.layers[-1] + ".W1"].d.shape[1])

    def clone(self):
        """Независимая копия сети (веса, моменты оптимизатора, множители) для эволюционных испытаний."""
        net = Net.__new__(Net); net.cfg = dict(self.cfg); net.layers = list(self.layers); net.uid, net.V, net.t = self.uid, self.V, self.t
        net.rng = np.random.default_rng(); net._mask = {}; net.P = {k: T(p.d.copy(), req=True) for k, p in self.P.items()}
        net.m = {k: v.copy() for k, v in self.m.items()}; net.v = {k: v.copy() for k, v in self.v.items()}; net.lrm = dict(getattr(self, "lrm", {}))
        return net

    def drop_layer(self, k):
        """Убирает слой целиком (сжатие). Функция меняется — после этого сети нужно дообучиться."""
        for n in [n for n in self.P if n.startswith(k + ".")]: del self.P[n]; self.m.pop(n, None); self.v.pop(n, None)
        self.layers.remove(k); getattr(self, "lrm", {}).pop(k, None)

    def widen_mlp(self, k, extra):
        r = self.rng; d = self.cfg["d"]; W1, b1, W2 = (self.P[k + s] for s in (".W1", ".b1", ".W2"))
        W1.d = np.concatenate([W1.d, r.normal(0, 0.02, (d, extra)).astype(F)], 1)
        b1.d = np.concatenate([b1.d, np.zeros(extra, F)]); W2.d = np.concatenate([W2.d, np.zeros((extra, d), F)], 0)

    def prune_mlp(self, k, thresh=0.05):
        """Удаляет «мёртвые» нейроны (малый вклад). Возвращает сколько удалено."""
        W1, b1, W2 = (self.P[k + s] for s in (".W1", ".b1", ".W2"))
        imp = np.linalg.norm(W1.d, axis=0) * np.linalg.norm(W2.d, axis=1)
        keep = imp > thresh * (np.median(imp) + 1e-9)
        if keep.sum() < 32 or keep.all(): return 0
        W1.d, b1.d, W2.d = W1.d[:, keep].copy(), b1.d[keep].copy(), W2.d[keep].copy()
        return int((~keep).sum())

    def set_vocab(self, V):
        self.V = V

    def init_rows(self, ids_parts):
        """Новые токены (слияния BPE и т.п.) получают эмбеддинг как среднее составляющих."""
        E = self.P["E"].d
        for nid, parts in ids_parts:
            E[nid] = E[parts].mean(0) if parts else self.rng.normal(0, 0.02, E.shape[1])

    def n_params(self):
        return int(sum(p.d.size for n, p in self.P.items() if n != "E") + self.V * self.cfg["d"])

    # --- прямой проход ---
    def _causal(self, T_):
        if T_ not in self._mask: self._mask[T_] = np.triu(np.full((T_, T_), -1e9, F), 1)
        return self._mask[T_]

    def forward(self, ids, S=None):
        """S (B,T,NS) — состояние на каждый токен (у разных кусков опыта своё): добавляется к эмбеддингу, дальше его читает внимание всех слоёв."""
        B, T_ = ids.shape; d, H = self.cfg["d"], self.cfg["H"]; dh = d // H; P = self.P
        x = add(embed(P["E"], ids), _slice_pos(P["pos"], T_))
        if S is not None and "Ws" in P: x = add(x, matmul(T(np.asarray(S, dtype=F)), P["Ws"]))
        for k in self.layers:
            h = layernorm(x, P[k + ".ln1.g"], P[k + ".ln1.b"])
            q = transpose(reshape(matmul(h, P[k + ".Wq"]), (B, T_, H, dh)), (0, 2, 1, 3))
            kk = transpose(reshape(matmul(h, P[k + ".Wk"]), (B, T_, H, dh)), (0, 2, 3, 1))
            v = transpose(reshape(matmul(h, P[k + ".Wv"]), (B, T_, H, dh)), (0, 2, 1, 3))
            att = softmax(scale(matmul(q, kk), 1.0 / math.sqrt(dh)), self._causal(T_))
            o = reshape(transpose(matmul(att, v), (0, 2, 1, 3)), (B, T_, d))
            x = add(x, matmul(o, P[k + ".Wo"]))
            h = layernorm(x, P[k + ".ln2.g"], P[k + ".ln2.b"])
            m = matmul(_ACT[self.cfg.get("act", "gelu")](add(matmul(h, P[k + ".W1"]), P[k + ".b1"])), P[k + ".W2"])
            x = add(x, add(m, P[k + ".b2"]))
        return layernorm(x, P["lnf.g"], P["lnf.b"])

    def loss(self, ids, tgt, w, S=None):
        """ids/tgt (B,T) int, w (B,T) веса. Возвращает (loss T, nll (B,T))."""
        self.V = max(self.V, int(ids.max()) + 1, int(tgt.max()) + 1)       # словарь мог вырасти до записи V
        V = self.V                                                          # один и тот же V на весь проход: другой поток может поднять self.V (новые коды картинок) посреди расчёта
        x = self.forward(ids, S); B, T_ = ids.shape
        lg = tied_logits(x, self.P["E"], V)
        l, nll = cross_entropy(reshape(lg, (B * T_, V)), tgt.reshape(-1), w.reshape(-1).astype(F))
        return l, nll.reshape(B, T_)

    # --- обучение ---
    def step(self, ids, tgt, w, lr=3e-4, wd=0.01, clip=1.0, b1=0.9, b2=0.95, S=None):
        for p in self.P.values(): p.g = None
        l, nll = self.loss(ids, tgt, w, S); backward(l)
        gs = {n: p.g for n, p in self.P.items() if p.g is not None}
        gn = math.sqrt(sum(float(np.dot(g.ravel(), g.ravel())) for g in gs.values()))          # float32-скалярное произведение вместо копий в float64
        sc = min(1.0, clip / (gn + 1e-9)); self.t += 1; t = self.t
        lr0 = lr
        for n, g in gs.items():
            p = self.P[n]; g = g * sc if sc < 1.0 else g; lr = lr0 * getattr(self, "lrm", {}).get(n.split(".")[0], 1.0)       # у слоя может быть свой множитель скорости (эволюция)
            if n not in self.m or self.m[n].shape != p.d.shape: self.m[n] = np.zeros_like(p.d); self.v[n] = np.zeros_like(p.d)
            m, v = self.m[n], self.v[n]
            if OPT == "muon" and p.d.ndim == 2 and n not in ("E", "pos", "Ws"):
                m *= 0.95; m += g; u = newton_schulz(g + 0.95 * m); u *= 0.2 * math.sqrt(max(p.d.shape))
                p.d -= lr * wd * p.d; p.d -= lr * 0.5 * 3.0 * u            # lr службы 2e-3 -> эффективный как на стенде (1e-3 * 3)
                continue
            m *= b1; m += (1 - b1) * g; v *= b2; v += (1 - b2) * g * g
            upd = (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8)
            if p.d.ndim >= 2 and n not in ("E", "pos", "Ws"): p.d -= lr * wd * p.d
            p.d -= lr * upd
        return float(l.d), gn, nll


    def probe(self, ids, topk=8, sv=None):
        """Прогон одной последовательности без обучения с записью активности: что «горит» в каждом слое. Для окна мозга и журнала."""
        P = self.P; d, H = self.cfg["d"], self.cfg["H"]; dh = d // H; ids = np.asarray(ids, dtype=np.int64)[-self.cfg["ctx"]:]; T_ = len(ids)
        def ln(x, g, b):
            xc = x - x.mean(-1, keepdims=True); return xc / np.sqrt((xc * xc).mean(-1, keepdims=True) + 1e-5) * g + b
        x = P["E"].d[ids] + P["pos"].d[:T_]; mask = np.triu(np.full((T_, T_), -1e9, F), 1); out = {"layers": [], "T": int(T_)}
        if sv is not None and "Ws" in P: x = x + (np.asarray(sv, F) @ P["Ws"].d)
        for k in self.layers:
            h = ln(x, P[k + ".ln1.g"].d, P[k + ".ln1.b"].d)
            q = (h @ P[k + ".Wq"].d).reshape(T_, H, dh).transpose(1, 0, 2); kk = (h @ P[k + ".Wk"].d).reshape(T_, H, dh).transpose(1, 0, 2); v = (h @ P[k + ".Wv"].d).reshape(T_, H, dh).transpose(1, 0, 2)
            a = q @ kk.transpose(0, 2, 1) / math.sqrt(dh) + mask; a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
            o = (a @ v).transpose(1, 0, 2).reshape(T_, d) @ P[k + ".Wo"].d; x = x + o
            h2 = ln(x, P[k + ".ln2.g"].d, P[k + ".ln2.b"].d); u = h2 @ P[k + ".W1"].d + P[k + ".b1"].d
            act = (np.maximum(u, 0) ** 2) if self.cfg.get("act") == "relu2" else 0.5 * u * (1 + np.tanh(0.7978845608 * (u + 0.044715 * u ** 3))); m = act @ P[k + ".W2"].d + P[k + ".b2"].d; x = x + m
            last = act[-1]; ent = -(a[:, -1] * np.log(a[:, -1] + 1e-9)).sum(-1)
            out["layers"].append({"name": k, "act": last.tolist(), "att": a[:, -1].tolist(), "att_ent": ent.tolist(),
                                  "attn_out": float(np.linalg.norm(o[-1])), "mlp_out": float(np.linalg.norm(m[-1])), "resid": float(np.linalg.norm(x[-1])),
                                  "active": float((np.abs(act) > 0.1).mean())})
        xf = ln(x, P["lnf.g"].d, P["lnf.b"].d); lg = (xf[-1] @ P["E"].d[:self.V].T).astype(np.float64); p = np.exp(lg - lg.max()); p /= p.sum()
        top = np.argsort(-p)[:topk]; out["top"] = [[int(i), float(p[i])] for i in top]; out["entropy"] = float(-(p * np.log(p + 1e-12)).sum())
        return out

    # --- генерация ---
    def _gen_cached(self, prompt, max_new, temp, top_k, top_p, rep, stop, rng, cb, lock, ban, sv=None):
        """Быстрая генерация с кэшем K/V: каждый токен считается один раз, а не весь контекст заново (в десятки раз быстрее)."""
        P = self.P; d, H = self.cfg["d"], self.cfg["H"]; dh = d // H; ctx = self.cfg["ctx"]; ids = list(prompt)[-(ctx - 1):]; out = []
        def ln(x, g, b):
            xc = x - x.mean(-1, keepdims=True); return xc / np.sqrt((xc * xc).mean(-1, keepdims=True) + 1e-5) * g + b
        gl = (lambda u: np.maximum(u, 0) ** 2) if self.cfg.get("act") == "relu2" else (lambda u: 0.5 * u * (1 + np.tanh(0.7978845608 * (u + 0.044715 * u ** 3))))
        svb = (np.asarray(sv, F) @ P["Ws"].d) if (sv is not None and "Ws" in P) else None
        Kc = [np.zeros((H, ctx, dh), F) for _ in self.layers]; Vc = [np.zeros((H, ctx, dh), F) for _ in self.layers]
        def run(tokens, pos0):
            T_ = len(tokens); x = P["E"].d[tokens] + P["pos"].d[pos0:pos0 + T_]
            if svb is not None: x = x + svb
            mask = np.triu(np.full((T_, pos0 + T_), -1e9, F), 1 + pos0)
            for li, k in enumerate(self.layers):
                h = ln(x, P[k + ".ln1.g"].d, P[k + ".ln1.b"].d)
                q = (h @ P[k + ".Wq"].d).reshape(T_, H, dh).transpose(1, 0, 2)
                Kc[li][:, pos0:pos0 + T_] = (h @ P[k + ".Wk"].d).reshape(T_, H, dh).transpose(1, 0, 2); Vc[li][:, pos0:pos0 + T_] = (h @ P[k + ".Wv"].d).reshape(T_, H, dh).transpose(1, 0, 2)
                a = q @ Kc[li][:, :pos0 + T_].transpose(0, 2, 1) / math.sqrt(dh) + mask; a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
                x = x + (a @ Vc[li][:, :pos0 + T_]).transpose(1, 0, 2).reshape(T_, d) @ P[k + ".Wo"].d
                h2 = ln(x, P[k + ".ln2.g"].d, P[k + ".ln2.b"].d); x = x + gl(h2 @ P[k + ".W1"].d + P[k + ".b1"].d) @ P[k + ".W2"].d + P[k + ".b2"].d
            xf = ln(x[-1], P["lnf.g"].d, P["lnf.b"].d); return (xf @ P["E"].d[:self.V].T).astype(np.float64)
        pos = 0
        for _ in range(max_new):
            if len(ids) + len(out) >= ctx: break
            if lock: lock.acquire()
            try: lg = run(ids if pos == 0 else [out[-1]], pos); pos = pos + (len(ids) if pos == 0 else 1)
            finally:
                if lock: lock.release()
            for t in set((ids + out)[-48:]): lg[t] = lg[t] / rep if lg[t] > 0 else lg[t] * rep
            if ban is not None: lg[ban[ban < len(lg)]] = -1e9
            lg /= max(temp, 1e-3)
            if top_k and top_k < len(lg): kth = np.partition(lg, -top_k)[-top_k]; lg[lg < kth] = -1e9
            p = np.exp(lg - lg.max()); p /= p.sum()
            if top_p < 1:
                o = np.argsort(-p); c = np.cumsum(p[o]); cut = o[c > top_p]
                if len(cut) > 1: p[cut[1:]] = 0; p /= p.sum()
            nid = int(rng.choice(len(p), p=p)); out.append(nid)
            if cb and cb(nid) is False: break
            if nid in stop: break
        return out

    def generate(self, prompt, max_new=120, temp=0.8, top_k=40, top_p=0.95, rep=1.15, stop=(), rng=None, cb=None, lock=None, ban=None, sv=None):
        return self._gen_cached(prompt, max_new, temp, top_k, top_p, rep, stop, rng or np.random.default_rng(), cb, lock, ban, sv)

    def generate_slow(self, prompt, max_new=120, temp=0.8, top_k=40, top_p=0.95, rep=1.15, stop=(), rng=None, cb=None, lock=None, ban=None):
        rng = rng or np.random.default_rng(); ids = list(prompt)[-self.cfg["ctx"] + 1:]; out = []
        with no_grad():
            for _ in range(max_new):
                if lock: lock.acquire()                      # замок только на один проход, чтобы обучение не простаивало
                try:
                    x = self.forward(np.array([ids[-self.cfg["ctx"]:]], dtype=np.int64))
                    lg = (x.d[0, -1] @ self.P["E"].d[:self.V].T).astype(np.float64)
                finally:
                    if lock: lock.release()
                for t in set(ids[-48:]): lg[t] = lg[t] / rep if lg[t] > 0 else lg[t] * rep
                if ban is not None: lg[ban[ban < len(lg)]] = -1e9
                lg /= max(temp, 1e-3)
                if top_k and top_k < len(lg): kth = np.partition(lg, -top_k)[-top_k]; lg[lg < kth] = -1e9
                p = np.exp(lg - lg.max()); p /= p.sum()
                if top_p < 1:
                    o = np.argsort(-p); c = np.cumsum(p[o]); cut = o[c > top_p]
                    if len(cut) > 1: p[cut[1:]] = 0; p /= p.sum()
                nid = int(rng.choice(len(p), p=p)); ids.append(nid); out.append(nid)
                if cb and cb(nid) is False: break
                if nid in stop: break
        return out

    def nll_of(self, ids_list, maxlen=None, sv=None):
        """Средний nll на токен для списка последовательностей (без обучения)."""
        tot = n = 0
        with no_grad():
            for ids in ids_list:
                ids = ids[: (maxlen or self.cfg["ctx"])]
                if len(ids) < 3: continue
                a = np.array([ids[:-1]], dtype=np.int64); b = np.array([ids[1:]], dtype=np.int64)
                S = None if sv is None else np.broadcast_to(np.asarray(sv, F), (1, a.shape[1], len(sv)))
                _, nll = self.loss(a, b, np.ones_like(b, dtype=F), S); tot += float(nll.sum()); n += nll.size
        return tot / max(n, 1)

    # --- сохранение ---
    def save(self, path):
        arrs = {"P_" + n: p.d for n, p in self.P.items()}
        arrs["meta"] = np.frombuffer(json.dumps({"cfg": self.cfg, "layers": self.layers, "uid": self.uid, "V": self.V, "t": self.t}).encode(), dtype=np.uint8)
        for n in self.m: arrs["M_" + n] = self.m[n]; arrs["S_" + n] = self.v[n]
        tmp = path + ".tmp.npz"; np.savez(tmp, **arrs); os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        z = np.load(path); meta = json.loads(bytes(z["meta"]).decode())
        net = cls.__new__(cls); net.cfg, net.layers, net.uid, net.V, net.t = meta["cfg"], meta["layers"], meta["uid"], meta["V"], meta["t"]
        net.rng = np.random.default_rng(); net._mask = {}
        net.P = {k[2:]: T(z[k].astype(F), req=True) for k in z.files if k.startswith("P_")}
        net.m = {k[2:]: z[k] for k in z.files if k.startswith("M_")}; net.v = {k[2:]: z[k] for k in z.files if k.startswith("S_")}
        if "Ws" not in net.P: net.P["Ws"] = T(np.zeros((NS, net.cfg["d"]), F), req=True)       # сеть, рождённая до «состояния в мозге»
        return net

def _slice_pos(pos, T_):
    o = _mk(pos.d[:T_][None], (pos,), None)
    if o._p:
        def bk(g): gp = np.zeros_like(pos.d); gp[:T_] = g.sum(0); _acc(pos, gp)
        o._b = bk
    return o
