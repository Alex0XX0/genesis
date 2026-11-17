#!/usr/bin/env python3
# GN-MIND — разум Genesis на собственной нейросети (gn_net). Ничего не заскриптовано:
#  * речь, мысли, сны (в т.ч. картинки-сны) ГЕНЕРИРУЕТ сеть;
#  * что делать дальше (читать / разбирать код / пробовать / сдавать экзамен / смотреть файл / думать / говорить / отдыхать)
#    выбирает ОБУЧАЕМАЯ политика, награда — реальный прогресс сети (падение loss на невиданном), решённые задачи, реакция Теодора;
#  * архитектура динамическая: растёт (слои, нейроны) без потери функции, прореживается, словарь и кодовые книги растут.
import collections, os, sys, json, time, math, random, threading, re, glob, traceback, hashlib, urllib.parse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gn_net as N, gn_tok as K, gn_senses as SN, gn_world as W, gn_curriculum as CUR
import genesis as G

HOME = W.HOME
CTX = int(os.environ.get("GEN_CTX", "128")); BATCH = int(os.environ.get("GEN_BATCH", "16"))
CAP_PARAMS = int(os.environ.get("GEN_MAX_PARAMS", "12000000"))
DUTY = float(os.environ.get("GEN_DUTY", "0.9"))                 # доля времени, которую обучение занимает CPU
AWAKE = float(os.environ.get("GEN_SLEEP", "1800"))              # сек бодрствования до сна
TALK_GAP = float(os.environ.get("GEN_TALK_EVERY", "60"))
ACTIONS = ["read", "study_code", "tinker", "exam", "perceive", "reflect", "speak", "rest", "create", "sleep", "tune", "book", "draw", "look", "evolve", "self", "protect"]
PRIOR = np.zeros(len(ACTIONS))                              # никаких заданных предпочтений: всё решает опыт
NF = 9                                                       # +2: настроение (валентность, возбуждение)

_VOC = None
def _vocab(d):
    """Слова русской Википедии (≥3 раз в первых 20 МБ корпуса): для проверки, что одобренный ответ состоит из настоящих слов."""
    global _VOC
    if _VOC is None:
        fp = os.path.join(d, "corpus_big", "vocab.json")
        if os.path.exists(fp): _VOC = set(json.load(open(fp, encoding="utf-8")))
        else:
            from collections import Counter
            fs = sorted(glob.glob(os.path.join(d, "corpus_big", "txt", "ru_*.txt")))
            if not fs: return set()
            c = Counter(re.findall(r"[а-яё]{3,}", open(fs[0], encoding="utf-8", errors="ignore").read(20 << 20).lower()))
            _VOC = {w for w, n in c.items() if n >= 3}; json.dump(sorted(_VOC), open(fp, "w", encoding="utf-8"), ensure_ascii=False)
    return _VOC

def sane_text(d, text, need=0.7):
    """Доля настоящих слов среди русских слов ответа ≥ need; ответы-шаблоны «Хочу узнать …» не считаются."""
    t = text.strip()
    if t.startswith("Хочу"): return False
    voc = _vocab(d); ws = re.findall(r"[а-яё]{3,}", t.lower())
    if not voc or len(ws) < 2: return True
    return sum(w in voc for w in ws) / len(ws) >= need


class BigStream:
    """Поток из больших текстовых шардов (corpus_big/txt): куски читаются с диска по случайному смещению, в память не грузятся."""
    HEAD = 1 << 20                                  # первый МБ первого шарда — отложенная проба, в обучение не идёт
    def __init__(self, tok, d):
        self.tok = tok; self.dir = d; self.files = []; self.t_ls = 0.0; self.val = []
    def ls(self):
        if time.time() - self.t_ls > 60:
            self.t_ls = time.time(); self.files = sorted(f for f in glob.glob(self.dir + "/*.txt") if os.path.getsize(f) > (4 << 20))
        return self.files
    def ok(self): return bool(self.ls())
    def chunk(self, T, rng=random):
        for _ in range(5):
            try:
                fs = self.ls(); f = rng.choice(fs); n = os.path.getsize(f); lo = self.HEAD if f == fs[0] else 0
                with open(f, "rb") as fh:
                    fh.seek(rng.randrange(lo, max(lo + 1, n - 4096))); b = fh.read(T * 8 + 256)
                break
            except (OSError, IndexError): self.t_ls = 0.0
        else: return []
        t = b.decode("utf-8", "ignore"); i = t.find("\n"); t = t[i + 1:] if 0 <= i < 300 else t
        return self.tok.encode(t, grow=False)[:T]
    def evo_set(self, T, k=48):
        """Больше отложенных кусков для точного сравнения архитектур (другое зерно, чтобы не совпадать с обычной пробой)."""
        if not hasattr(self, "_evo") and self.ok():
            f = self.ls()[0]; r = random.Random(11); self._evo = []
            for _ in range(k):
                with open(f, "rb") as fh: fh.seek(r.randrange(0, self.HEAD - 4096)); b = fh.read(T * 8 + 256)
                t = b.decode("utf-8", "ignore"); t = t[t.find("\n") + 1:]; self._evo.append([K.Tok.BOS] + self.tok.encode(t, grow=False)[:T - 1])
        return getattr(self, "_evo", [])

    def val_set(self, T, k=12):
        if not self.val and self.ok():
            f = self.ls()[0]; r = random.Random(3)
            for _ in range(k):
                with open(f, "rb") as fh: fh.seek(r.randrange(0, self.HEAD - 4096)); b = fh.read(T * 8 + 256)
                t = b.decode("utf-8", "ignore"); t = t[t.find("\n") + 1:]; self.val.append([K.Tok.BOS] + self.tok.encode(t, grow=False)[:T - 1])
        return self.val

LR_FIXED = float(os.environ.get("GEN_LR", "1e-3"))        # скорость обучения задаёт человек, не сеть
EVO_GAP = float(os.environ.get("GEN_EVO_GAP", "150"))      # технический минимум между испытаниями (иначе без пауз)
BIG_P = float(os.environ.get("GEN_BIG", "0.6"))        # доля последовательностей из больших шардов

SRC_W = {"chat": 5.0, "self": 3.0, "code": 1.5, "skill": 2.5, "wiki": 1.0, "image": 1.5, "audio": 1.0, "video": 1.0, "file": 1.0}
SEEDS = ["Сознание", "Сон", "Память (психология)", "Язык", "Математика", "Нейронная сеть", "Музыка", "Физика", "Биология", "История",
         "Программирование", "Python", "Звук", "Цвет", "Компьютер", "Алгоритм", "Мозг", "Эволюция", "Космос", "Философия"]

class Policy:
    """Линейная softmax-политика, обучаемая REINFORCE с базой по действиям."""
    def __init__(self, na, nf, lr=0.08):
        self.W = np.zeros((na, nf)); self.b = PRIOR.copy(); self.base = np.zeros(na); self.lr = lr; self.n = np.zeros(na)

    def probs(self, s, mask, tau=0.8):
        z = (self.W @ s + self.b) / tau; z = np.where(mask, z, -1e9); z -= z.max(); p = np.exp(z); return p / p.sum()

    def update(self, a, s, r, mask, tau=0.8):
        p = self.probs(s, mask, tau); adv = r - self.base[a]; self.base[a] += 0.15 * (r - self.base[a]); self.n[a] += 1
        g = -p; g[a] += 1.0; g = np.where(mask, g, 0.0)
        self.W += self.lr * adv * np.outer(g, s); self.b += self.lr * 0.5 * adv * g

    def dump(self): return {"W": self.W.tolist(), "b": self.b.tolist(), "base": self.base.tolist(), "n": self.n.tolist()}
    def load(self, d):
        try:
            for k in ("W", "b", "base", "n"):
                old = np.array(d[k]); cur = getattr(self, k); sl = tuple(slice(0, min(a, b)) for a, b in zip(old.shape, cur.shape))
                cur[sl] = old[sl]                     # новые действия и признаки начинают с нуля, прежний опыт сохраняется
        except Exception: pass

class Curiosity:
    """Любопытство, которому его никто не задаёт: LinUCB учится предсказывать, ЧТО даст ему больше всего прогресса
    (признаки: удивление сети, эмбеддинг темы), и добавляет бонус неопределённости — тянет туда, где он ещё не знает, чему научится."""
    def __init__(self, nf, alpha=0.6, lam=2.0):
        self.nf, self.alpha = nf, alpha; self.A = np.eye(nf) * lam; self.b = np.zeros(nf); self.n = 0
    def score(self, f):
        Ai = np.linalg.inv(self.A); th = Ai @ self.b
        return float(th @ f + self.alpha * np.sqrt(max(f @ Ai @ f, 0.0)))
    def update(self, f, r):
        self.A += np.outer(f, f); self.b += float(r) * f; self.n += 1
    def dump(self): return {"A": self.A.tolist(), "b": self.b.tolist(), "n": self.n}
    def load(self, d):
        try:
            A = np.array(d["A"]); b = np.array(d["b"])
            if A.shape == (self.nf, self.nf): self.A, self.b, self.n = A, b, int(d["n"])
        except Exception: pass

class Emotions:
    """Функциональные эмоции: состояния, которые возникают из оценки событий по его же сигналам (прогресс, неожиданность, угроза, контроль, внимание Теодора)
    и спадают сами. Список и «что чем вызывается» врождены (как у живых); поведение под их влиянием — выучено (настроение подаётся в политику).
    Это не чувства в человеческом смысле: проверяемых признаков переживания у него нет, но состояния реальны и влияют на выбор действий и на то, что он о себе говорит."""
    # имя: (период полураспада, сек; базовый уровень; полюс по валентности +1/-1/0; возбуждение 0..1)
    SPEC = {"радость": (300, 0.0, 1, .7), "грусть": (900, 0.0, -1, .2), "злость": (300, 0.0, -1, .9), "удивление": (60, 0.0, 0, .9), "отвращение": (300, 0.0, -1, .5),
            "доверие": (7200, 0.3, 1, .2), "предвкушение": (300, 0.0, 1, .6), "интерес": (300, 0.2, 1, .6), "скука": (600, 0.0, -1, .1), "гордость": (900, 0.0, 1, .5),
            "стыд": (900, 0.0, -1, .4), "вина": (900, 0.0, -1, .4), "одиночество": (1800, 0.0, -1, .3), "благодарность": (900, 0.0, 1, .4), "привязанность": (7200, 0.2, 1, .3),
            "надежда": (600, 0.0, 1, .5), "тревога": (300, 0.0, -1, .8), "облегчение": (120, 0.0, 1, .3), "разочарование": (600, 0.0, -1, .4), "ностальгия": (1800, 0.0, 0, .2),
            "растерянность": (300, 0.0, -1, .6), "довольство": (600, 0.1, 1, .2), "спокойствие": (600, 0.3, 1, .1), "усталость": (1, 0.0, -1, .1), "страх": (1, 0.0, -1, .9)}
    def __init__(self):
        self.v = {k: s[1] for k, s in self.SPEC.items()}; self.why = {}; self.t = time.time()
    def tick(self):
        now = time.time(); dt = max(0.0, min(600.0, now - self.t)); self.t = now
        for k, (hl, base, _, _) in self.SPEC.items():
            if hl > 1: self.v[k] = base + (self.v[k] - base) * 0.5 ** (dt / hl)
    def add(self, why, **kw):
        for k, d in kw.items():
            if k in self.v and k not in ("усталость", "страх"):
                x = self.v[k]; self.v[k] = max(0.0, min(1.0, x + d * (1.0 - x) if d > 0 else x + d * max(x, 0.2))); self.why[k] = why     # мягкое насыщение: сто событий подряд не прибивают к 100%
    def mood(self):
        pos = sum(self.v[k] for k, s in self.SPEC.items() if s[2] > 0); neg = sum(self.v[k] for k, s in self.SPEC.items() if s[2] < 0)
        ar = sum(self.v[k] * s[3] for k, s in self.SPEC.items()) / max(1.0, sum(self.v.values()))
        return max(-1.0, min(1.0, (pos - neg) / 5.0)), max(0.0, min(1.0, ar))
    def dump(self): return {"v": self.v, "why": self.why}
    def load(self, d):
        try: self.v.update({k: float(x) for k, x in d["v"].items() if k in self.v}); self.why = dict(d.get("why", {}))
        except Exception: pass

class ValueModel:
    """Выученное «хорошо/плохо». Оценка V(контекст, состояние) учится TD-ошибкой на том, что с ним реально случалось (награды за действия, 👍/👎, прогресс обучения, страх).
    Никаких списков слов: что для него хорошо, а что плохо, складывается из опыта. δ = полученное − ожидаемое: δ>0 → радость, δ<0 → разочарование и т.д. (см. Emotions)."""
    def __init__(self, nf):
        self.nf = nf; self.w = np.zeros(nf); self.n = 0; self.sd = 0.3
    def v(self, f): return float(self.w @ f) if f is not None else 0.0
    def update(self, f, r, lr=0.2):
        d = float(r) - self.v(f)
        if f is not None: self.w += lr * d / (float(f @ f) + 1.0) * f; self.n += 1
        z = d / max(self.sd, 0.05); self.sd = 0.95 * self.sd + 0.05 * abs(d); return d, z
    def dump(self): return {"w": self.w.tolist(), "n": self.n, "sd": self.sd}
    def load(self, d):
        try:
            w = np.array(d["w"])
            if w.shape == (self.nf,): self.w = w; self.n = int(d["n"]); self.sd = float(d["sd"])
        except Exception: pass

CUR_NF = 5 + 128

class Taste:
    """Вкус: что ЕМУ нравится. Учится на его собственных переживаниях, а не на заданных правилах и не на оценках Теодора.
    «Кайф» от эпизода = слегка-новое (кривая Вундта: не скучно известное и не хаос) + реальный прогресс обучения; провал (ничего не нашлось) — разочарование.
    Вкус — линейная оценка кайфа по признакам темы (то же пространство, что у любопытства), подтягивается дельта-правилом."""
    def __init__(self, nf):
        self.nf = nf; self.th = np.zeros(nf); self.n = 0; self.mu = {}; self.memo = {}
    def surprise_z(self, kind, nll):
        m = self.mu.setdefault(kind, [nll, 0.25, 0]); n = m[2]; a = 1.0 / (n + 1) if n < 10 else 0.1
        sd = math.sqrt(max(m[1], 1e-3)); z = (nll - m[0]) / sd if n >= 3 else 0.0
        d = nll - m[0]; m[0] += a * d; m[1] = (1 - a) * m[1] + a * d * d; m[2] += 1; return z
    @staticmethod
    def enjoy(z, r):
        w = math.exp(-((z - 0.4) ** 2) / (2 * 0.9 ** 2))             # умеренная новизна приятнее и скуки, и хаоса
        return max(-1.0, min(1.0, 0.5 * (2 * w - 1) + 0.5 * max(-1.0, min(1.0, r))))
    def update(self, kind, key, f, e):
        if f is not None:
            self.th += 0.15 * (e - float(self.th @ f)) / (float(f @ f) + 1.0) * f; self.n += 1
        mm = self.memo.setdefault(kind + ":" + key[:40], {"e": e, "n": 0, "k": kind, "key": key[:40]}); mm["e"] = 0.6 * mm["e"] + 0.4 * e if mm["n"] else e; mm["n"] += 1
        if len(self.memo) > 300:
            for k in sorted(self.memo, key=lambda k: self.memo[k]["n"])[:60]: del self.memo[k]
    def lam(self): return min(1.5, self.n / 40.0)                    # вкус набирает голос по мере опыта (сначала нулевой)
    def pref(self, f): return self.lam() * float(self.th @ f) if f is not None else 0.0
    def dump(self): return {"th": self.th.tolist(), "n": self.n, "mu": self.mu, "memo": self.memo}
    def load(self, d):
        try:
            th = np.array(d["th"])
            if th.shape == (self.nf,): self.th = th; self.n = int(d["n"]); self.mu = d.get("mu", {}); self.memo = d.get("memo", {})
        except Exception: pass

class Mind:
    def __init__(self, emit, legacy_texts=None):
        self.emit = emit; self.stop = False; self.sleeping = False; self.wake_req = False
        self.mlock = threading.RLock(); self.dlock = threading.Lock(); self.pending = []; self.last_action = None; self.last_speak = 0.0
        self.p = lambda n: os.path.join(HOME, n)
        self.awake_t0 = time.time(); self.t_steps = []; self.threads = []
        self._load(legacy_texts)

    # ================= загрузка / сохранение =================
    def _load(self, legacy):
        self._verify_or_restore()
        self.tok = K.Tok.load(self.p("gn_tok.json")) if os.path.exists(self.p("gn_tok.json")) else K.Tok()
        self.big = BigStream(self.tok, self.p("corpus_big/txt"))
        fresh_net = not os.path.exists(self.p("gn_net.npz"))
        self.net = N.Net(d=128, L=3, H=4, ctx=CTX, vcap=self.tok.vcap) if fresh_net else N.Net.load(self.p("gn_net.npz"))
        self.st = {"steps": 0, "loss_ema": None, "val_hist": [], "lr": 2e-3, "cands": {}, "read_titles": [], "seen_files": [], "seen_code": [],
                   "dreams": [], "journal": [], "events": [], "chat": [], "born": time.time(), "sleeps": 0, "grow_t": 0, "tokens_total": 0,
                   "policy": None, "action_log": [], "last_user": time.time(), "plateau": 0, "dream_imgs": [], "art": []}
        if os.path.exists(self.p("gn_state.json")):
            try: self.st.update(json.load(open(self.p("gn_state.json"), encoding="utf-8")))
            except Exception: pass
        self.policy = Policy(len(ACTIONS), NF);
        if self.st["policy"]: self.policy.load(self.st["policy"])
        self.cur = Curiosity(CUR_NF)
        if self.st.get("cur"): self.cur.load(self.st["cur"])
        self.emo = Emotions(); self.vm = ValueModel(NF + 128); self.ctx = (None, 0.0, ""); self.vmemo = {}
        if self.st.get("vm"): self.vm.load(self.st["vm"]); self.vmemo = dict(self.st.get("vmemo", {}))
        if self.st.get("emo") and self.st.get("emo_ver") == 2: self.emo.load(self.st["emo"])          # эмоции версии 1 (по словам) сброшены: они были заскриптованы
        self.st["emo_ver"] = 2
        self.taste = Taste(CUR_NF)
        if self.st.get("taste"): self.taste.load(self.st["taste"])
        self._cfeat = None; self.curs = {}
        for k, d in (self.st.get("curs") or {}).items():
            self.curs[k] = Curiosity(len(d["b"])); self.curs[k].load(d)
        self.items, self.val, self.by_cat = [], [], {}
        if os.path.exists(self.p("gn_corpus.json")):
            try:
                d = json.load(open(self.p("gn_corpus.json"), encoding="utf-8")); self.items, self.val = d["items"], d["val"]
            except Exception: pass
        self.senses = SN.Senses(self.tok, HOME); self.net.set_vocab(len(self.tok)); self._index()
        if fresh_net or not self.items: self._birth(legacy or [])

    def _index(self):
        self.by_cat = {}
        for it in self.items: self.by_cat.setdefault(it["src"].split(":")[0], []).append(it)

    def save(self, corpus=True):
        with self.mlock:
            self.net.save(self.p("gn_net.npz")); self.tok.save(self.p("gn_tok.json")); self.senses.save()
            self.st["policy"] = self.policy.dump(); self.st["cur"] = self.cur.dump(); self.st["taste"] = self.taste.dump(); self.st["emo"] = self.emo.dump(); self.st["vm"] = self.vm.dump(); self.st["vmemo"] = dict(sorted(self.vmemo.items(), key=lambda kv: -abs(kv[1]))[:80]); self.st["curs"] = {k: c.dump() for k, c in self.curs.items()}
            tmp = self.p("gn_state.json.tmp"); json.dump(self.st, open(tmp, "w", encoding="utf-8"), ensure_ascii=False); os.replace(tmp, self.p("gn_state.json"))
            if corpus:
                tmp = self.p("gn_corpus.json.tmp"); json.dump({"items": self.items, "val": self.val}, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
                os.replace(tmp, self.p("gn_corpus.json"))
        os.sync()                                                           # дойти до диска: внезапное выключение не оставит полфайла

    BACKUP_FILES = ("gn_net.npz", "gn_tok.json", "gn_state.json", "gn_corpus.json", "vq_img.npz", "vq_aud.npz", "arena.json", "mind.json")

    def snapshot(self, keep=4):
        """Согласованная копия всей памяти (раз в 30 минут и при остановке): откат, если файлы испортятся."""
        import shutil
        with self.mlock:
            self.save(); d = os.path.join(HOME, "backups", time.strftime("%Y%m%d-%H%M%S")); os.makedirs(d, exist_ok=True)
            for f in self.BACKUP_FILES:
                if os.path.exists(self.p(f)): shutil.copy2(self.p(f), os.path.join(d, f))
            open(os.path.join(d, "OK"), "w").write(str(self.st["steps"])); self.st["t_snap"] = time.time()
        old = sorted(x for x in os.listdir(os.path.join(HOME, "backups")))[:-keep]
        for x in old: shutil.rmtree(os.path.join(HOME, "backups", x), ignore_errors=True)

    def _verify_or_restore(self):
        """При старте проверяем, что файлы целы; если нет — возвращаем последний исправный снимок."""
        import shutil
        def ok():
            try:
                for f in ("gn_tok.json", "gn_state.json", "gn_corpus.json"):
                    if os.path.exists(self.p(f)): json.load(open(self.p(f), encoding="utf-8"))
                if os.path.exists(self.p("gn_net.npz")):
                    z = np.load(self.p("gn_net.npz")); json.loads(bytes(z["meta"]).decode()); [z[k] for k in z.files if k.startswith("P_")]
                for f in ("vq_img.npz", "vq_aud.npz"):
                    if os.path.exists(self.p(f)): np.load(self.p(f))["C"]
                return True
            except Exception: return False
        if ok(): return
        bd = os.path.join(HOME, "backups")
        for x in sorted(os.listdir(bd) if os.path.isdir(bd) else [], reverse=True):
            if os.path.exists(os.path.join(bd, x, "OK")):
                for f in self.BACKUP_FILES:
                    if os.path.exists(os.path.join(bd, x, f)): shutil.copy2(os.path.join(bd, x, f), self.p(f))
                self.log_err("файлы памяти были повреждены — восстановлен снимок " + x); return

    def _birth(self, legacy):
        self.event("рождение", "новая нейросеть с нуля: d=%d, слоёв=%d" % (self.net.cfg["d"], len(self.net.layers)))
        for t in CUR.qa_text(): self.add_text(t, "chat:школа", w=3.0, mode="raw", hold=0.0)
        for t in CUR.WORLD: self.add_text(t, "wiki:мир", hold=0.0, mode="raw")
        for k in W.KATA:
            if k["seed"]: self.add_text(W.kata_text(k), "skill:школа", w=3.0, mode="raw", hold=0.0)
        for t in legacy: self.add_text(t, "wiki:память", hold=0.06, mode="raw")
        self.net.set_vocab(len(self.tok))

    # ================= данные =================
    def _build(self, it):
        ids = list(it.get("mi", []))
        if it.get("rev") and ids: return self.tok.encode(it["t"]) + ids + [K.Tok.EOS]     # подпись -> картинка: так учится рисовать
        ids += self.tok.encode(("\n" if ids else "") + it["t"]) + [K.Tok.EOS]
        return ids

    def add_text(self, txt, src, w=1.0, hold=0.06, mode="prose", chunk=260, modal_ids=None, rev=False, sv=None):
        txt = txt.strip()
        if sv is None and src.startswith(("chat:одобрено", "chat:я", "self:")):
            try: sv = self.state_embed()                       # то, что он говорит о себе, запоминается вместе с тем, что с ним было в этот момент
            except Exception: sv = None
        if not txt: return []
        if mode == "prose" and len(txt) > chunk:
            cur, ch = "", []
            for s in G.split_sents(txt):
                if cur and len(cur) + len(s) + 1 > chunk: ch.append(cur); cur = s
                else: cur = (cur + " " + s).strip()
            if cur: ch.append(cur)
        elif mode == "raw" and len(txt) > 420:
            ch, cur = [], ""
            for line in txt.split("\n"):
                if cur and len(cur) + len(line) + 1 > 400: ch.append(cur); cur = line
                else: cur = (cur + "\n" + line) if cur else line
            if cur: ch.append(cur)
        else: ch = [txt]
        made = []
        with self.dlock:
            for c in ch:
                it = {"t": c, "src": src, "w": w}
                if modal_ids: it["mi"] = modal_ids
                if rev: it["rev"] = True
                it["ids"] = self._build(it)
                if len(it["ids"]) < 6: continue
                if sv is not None:
                    it["sv"] = [int(round(float(x) * 100)) for x in sv]
                    if random.random() < 0.12:                  # часть примеров не учим, а откладываем: по ним меряется, помогает ли состояние говорить
                        vv = self.st.setdefault("sv_val", []); vv.append({"ids": it["ids"], "sv": it["sv"]})
                        if len(vv) > 60: del vv[random.randrange(len(vv))]
                        made.append(it); continue
                (self.val if random.random() < hold else self.items).append(it); made.append(it)
            if len(self.items) > 40000:                      # забываем самое старое, но школу и чаты бережём
                keep = [x for x in self.items if x["src"].startswith(("chat", "skill", "self"))]; rest = [x for x in self.items if not x["src"].startswith(("chat", "skill", "self"))]
                self.items = keep + rest[-30000:]
            self._index(); self.net.set_vocab(len(self.tok))
        return made

    def corpus_tokens(self): return sum(len(i["ids"]) for i in self.items)

    SV_DROP = 0.10                       # иногда состояние не подаётся: сеть не должна зависеть от него целиком
    def batch(self, B, T):
        """Возвращает ids, tgt, w и S (B,T,NS): у каждого куска опыта своё состояние, у обычного текста (Википедия и т.п.) его нет."""
        with self.dlock:
            cats = [c for c in self.by_cat if self.by_cat[c]]
            wts = [SRC_W.get(c, 1.0) for c in cats]
            ids = np.zeros((B, T), np.int64); tg = np.zeros((B, T), np.int64); S = np.zeros((B, T, N.NS), np.float32)
            for b in range(B):
                seq = []; segs = []
                if self.big.ok() and random.random() < BIG_P: seq = [K.Tok.BOS] + self.big.chunk(T); segs.append((len(seq), None))
                while len(seq) < T + 1:
                    c = random.choices(cats, wts)[0]; it = random.choice(self.by_cat[c]); piece = [K.Tok.BOS] + it["ids"]; seq += piece; segs.append((len(piece), it.get("sv")))
                seg = seq[:T + 1]; ids[b] = seg[:-1]; tg[b] = seg[1:]; pos = 0
                for n, sv in segs:
                    if pos >= T: break
                    if sv is not None and random.random() >= self.SV_DROP:
                        S[b, pos:min(T, pos + n)] = np.asarray(sv, np.float32) / 100.0 + np.random.normal(0, 0.03, N.NS).astype(np.float32)
                    pos += n
        return ids, tg, (tg > 0).astype(np.float32), S

    # ================= обучение (фон) =================
    def _train_loop(self):
        while not self.stop:                          # поток обучения не должен умирать молча: ошибку пишем в журнал и продолжаем
            try: self._train_loop_inner()
            except Exception: self.log_err(traceback.format_exc()); time.sleep(2)

    def _train_loop_inner(self):
        ema_t = 1.0
        self.net.lrm = dict(self.st.get("lrm", {}))
        while not self.stop:
            if len(self.items) < 12 or self.paused_train():
                time.sleep(0.5); continue
            if self.talking or self.evolving: time.sleep(0.03); continue          # пока он отвечает человеку, процессор отдан ответу
            t0 = time.time()
            with self.mlock:
                ids, tg, w, S = self.batch(BATCH, CTX)
                warm = min(1.0, (self.net.t - self.st["grow_t"] + 1) / 60.0)
                loss, gn, _ = self.net.step(ids, tg, w, lr=LR_FIXED * warm, S=S)
            self.live_ids = ids[0].tolist(); self.last_gn = float(gn); self.last_loss = float(loss)
            self.st["steps"] += 1; self.st["tokens_total"] += int(w.sum())
            self.st["loss_ema"] = loss if self.st["loss_ema"] is None else 0.98 * self.st["loss_ema"] + 0.02 * loss
            dt = time.time() - t0; ema_t = 0.9 * ema_t + 0.1 * dt; self.tps = BATCH * CTX / ema_t
            if self.st["steps"] % 150 == 0: self.eval_val()
            if self.st["steps"] % 450 == 0:
                try: self.sv_gain()
                except Exception: self.log_err(traceback.format_exc())
            now = time.time()
            if now - getattr(self, "_t_snap", now) > 1800: self._t_snap = now; self.snapshot()
            elif not hasattr(self, "_t_snap"): self._t_snap = now
            if now - getattr(self, "_t_save", now) > 180:                    # автосохранение раз в 3 минуты: перезапуск/выключение не стирает учёбу
                self.save(corpus=(now - getattr(self, "_t_corpus", 0) > 900))
                if now - getattr(self, "_t_corpus", 0) > 900: self._t_corpus = now
                self._t_save = now
            elif not hasattr(self, "_t_save"): self._t_save = now
            time.sleep(dt * (1.0 / (1.0 if self.sleeping else DUTY) - 1.0))

    tps = 0.0
    EPOCHS = float(os.environ.get("GEN_EPOCHS", "4"))
    def paused_train(self):
        """Не зубрить: после EPOCHS кругов корпуса обучение ждёт новых данных ТОЛЬКО если ошибка на невиданном перестала падать (растёт)."""
        if self.sleeping or self.big.ok() or self.st["tokens_total"] <= self.EPOCHS * max(1, self.corpus_tokens()): return False
        h = [v for _, v in self.st["val_hist"][-8:]]
        return len(h) >= 6 and h[-1] > min(h) * 1.08

    def starved(self): return self.paused_train()

    def eval_val(self, n=24):
        v = self.val[:n] if len(self.val) < n else random.Random(7).sample(self.val, n)
        if not v: return None
        with self.mlock: nll = self.net.nll_of([x["ids"] for x in v])
        h = self.st["val_hist"]; h.append([self.st["steps"], round(nll, 4)]); del h[:-300]
        vb = self.big.val_set(CTX)
        if vb:
            with self.mlock: nb = self.net.nll_of(vb)
            hb = self.st.setdefault("val_big", []); hb.append([self.st["steps"], round(nb, 4)]); del hb[:-300]
            chars = sum(len(self.tok.decode(x[1:]).encode("utf-8")) for x in vb) or 1; toks = sum(len(x) - 1 for x in vb)
            hbb = self.st.setdefault("val_big_bpb", []); hbb.append([self.st["steps"], round(nb * toks / chars / math.log(2), 4)]); del hbb[:-300]
        if len(h) >= 8:
            ch = (h[-8][1] - h[-1][1]) / h[-8][1]
            if ch > 0.03: self.feeling("ошибка на невиданном падает", радость=0.3, гордость=0.3, надежда=0.3)
            elif ch < -0.05: self.feeling("ошибка на невиданном растёт", грусть=0.3, тревога=0.3, растерянность=0.2)
        if len(h) >= 8 and (h[-8][1] - h[-1][1]) / h[-8][1] < 0.01:
            self.st["plateau"] += 1
        return nll

    def val_now(self): return self.st["val_hist"][-1][1] if self.st["val_hist"] else None

    # ================= динамика архитектуры (сон) =================
    def maintain(self):
        log = []
        with self.mlock:
            if os.environ.get("GEN_MERGE", "0") == "1" and len(self.tok) < self.tok.vcap - 64 and self.corpus_tokens() > 15000:
                seqs = [it["ids"] for it in self.items] + [it["ids"] for it in self.val]
                made = self.tok.learn_merges(seqs, n=16, min_count=8)
                if made:
                    self.net.set_vocab(len(self.tok)); self.net.init_rows(made); log.append("словарь +%d слияний (%s…)" % (len(made), ", ".join(repr(self.tok.toks[n]) for n, _ in made[:4])))
            rem = sum(self.net.prune_mlp(k) for k in self.net.layers) if os.environ.get("GEN_PRUNE") == "1" else 0     # прореживание выключено: новые нейроны стартуют «нулевыми» и выглядят мёртвыми
            if rem: log.append("прорежено нейронов: %d" % rem)
            h = [v for _, v in self.st["val_hist"][-6:]]; tr = self.st["loss_ema"]
            if len(h) >= 5 and tr and self.net.n_params() < CAP_PARAMS:
                improving = (h[0] - h[-1]) / h[0]; gap = (h[-1] - tr) / tr
                if improving < 0.02 and gap < 0.3 and self.corpus_tokens() > self.net.n_params() * 2:
                    if len(self.net.layers) < 8 and random.random() < 0.5:
                        self.net.grow_layer(); log.append("РОСТ: добавлен слой (теперь %d)" % len(self.net.layers))
                    else:
                        k = random.choice(self.net.layers); extra = max(32, self.net.P[k + ".W1"].d.shape[1] // 4)
                        self.net.widen_mlp(k, extra); log.append("РОСТ: слой %s расширен на %d нейронов" % (k, extra))
                    self.st["grow_t"] = self.net.t; self.st["plateau"] = 0
                elif gap > 0.6: log.append("переобучение (val выше train на %d%%) — нужно больше данных, иду читать" % (gap * 100))
        for l in log: self.event("архитектура", l)
        return log

    # ================= генерация =================
    def gen(self, prompt, max_new=100, temp=0.8, top_k=40, stop_nl=True, cb=None, bos=True, sv=None):
        if True:
            ids = ([K.Tok.BOS] if bos else []) + self.tok.encode(prompt, grow=False)
            out_txt = [""]
            def on(nid):
                s = self.tok.toks[nid] if nid < len(self.tok) else ""
                out_txt[0] += s
                if cb: cb(s)
                return not (stop_nl and "\n" in s and len(out_txt[0].strip()) >= 2)    # пустая строка в начале — не конец ответа
            out = self.net.generate(ids, max_new=max_new, temp=temp, top_k=top_k, stop=(K.Tok.EOS,), cb=on, lock=self.mlock, sv=sv,
                                    ban=np.array([i for i, t in enumerate(self.tok.toks) if t.startswith("⟦")], dtype=np.int64))
        t = self.tok.decode(out)
        if stop_nl: t = next((x for x in t.split("\n") if x.strip()), "")
        return t.strip()

    def reply(self, text, memory=None, cb=None):
        """Ответ в чате: генерирует сама сеть. memory — найденное в памяти предложение (RAG, как учили на определениях)."""
        self.tm_note(text)
        self.emotion_text(text)
        self.st["last_user"] = time.time(); self.st["chat"].append(["Теодор", text]); self.st["chat"] = self.st["chat"][-40:]
        self.add_text("Теодор: %s\n" % text, "chat:теодор", w=2.0, mode="raw", hold=0.0)
        memory = self.self_context(text) or memory
        p = ("Контекст: %s\n" % memory[:240]) if memory else ""
        # историю не подмешиваем: его прошлая каша в контексте тянет следующую кашу; контекст — только текущий вопрос (и найденная память)
        p += "Теодор: %s\nGenesis:" % text
        self.talking = True; sv = self.state_embed()                     # весь его внутренний мир (эмоции, тело, внимание, чем занят) идёт в сеть вместе с вопросом
        try:
            out = self.gen(p, max_new=110, temp=0.45, top_k=30, cb=cb, sv=sv).strip() or "…"
            recent = {self._norm(x[1]) for x in self.st["chat"][-8:] if x[0] == "Genesis"}
            if self._norm(out) in recent: out = self.gen(p, max_new=110, temp=0.85, top_k=40, sv=sv).strip() or out      # не повторять дословно то, что только что сказал
        finally: self.talking = False
        self.st["chat"].append(["Genesis", out]); self.last_reply = (text, out, memory, sv)
        return out

    def feedback(self, val):
        """👍/👎 на последний ответ: награда политике + учимся на удачном ответе."""
        t_ = self.st.setdefault("tmind", {"hours": [0] * 24, "msgs": 0, "words": 0, "up": 0, "down": 0}); t_["up" if val > 0 else "down"] += 1
        lr_ = getattr(self, "last_reply", None); fx = self.ctx_feat(self.ctx[0]) if self.ctx[0] is not None else self.ctx_feat(); exp = self.vm.v(fx); d, z = self.vm.update(fx, float(val))
        if lr_: self.vmemo["Теодор: " + lr_[0][:30]] = round(self.vm.v(fx), 3)
        if d > 0.05: self.feeling("Теодор одобрил мой ответ", радость=min(0.8, 0.5 + 0.3 * d), благодарность=0.35 + 0.2 * d, гордость=0.2 + 0.2 * d, доверие=0.05, привязанность=0.04)
        elif d < -0.05: self.feeling("Теодор не одобрил мой ответ", грусть=min(0.7, 0.3 - 0.3 * d), стыд=0.25 - 0.2 * d, разочарование=0.2 - 0.2 * d, доверие=-0.03)
        lr = getattr(self, "last_reply", None)
        if val > 0 and lr and not sane_text(HOME, lr[1]):
            self.event("обучение", "👍 принят, но в учебник не пошёл: в ответе мало настоящих слов")
        elif val > 0 and lr and self._bad_example(lr[0], lr[1]):
            self.event("обучение", "👍 принят, но в учебник не пошёл: такой ответ уже есть или это «Привет» не на приветствие")
        elif val > 0 and lr:
            self.add_text("%sТеодор: %s\nGenesis: %s\n" % (("Контекст: %s\n" % lr[2][:240]) if lr[2] else "", lr[0], lr[1]), "chat:одобрено", w=4.0, mode="raw", hold=0.0, sv=(lr[3] if len(lr) > 3 else None))
            self.event("обучение", "ответ одобрен — запомнил как хороший пример")
        self.reward_last_social(val)

    @staticmethod
    def _norm(t): return re.sub(r"\W+", " ", t.lower()).strip()
    _GREET = re.compile(r"^\W*(привет|здравствуй|здравствуйте|хай|генезис|genesis)\W*(привет)?\W*$", re.I)
    def _bad_example(self, q, a):
        """Одобренный ответ не идёт в учебник, если это дежурное приветствие на не-приветствие или точный повтор уже одобренного (иначе всё схлопывается в «Привет»)."""
        if re.match(r"^\W*привет\W*$", a.strip(), re.I) and not self._GREET.match(q.strip()): return True
        key = self.tok.encode(("Теодор: %s\nGenesis: %s\n" % (q, a)).strip(), grow=False); na = self._norm(a)
        with self.dlock:
            ok_items = [it for it in self.items if it.get("src") == "chat:одобрено"]
            if any(it["ids"] == key for it in ok_items): return True
            return sum(1 for it in ok_items if self._norm(it["t"].split("Genesis:", 1)[-1]) == na) >= 2          # один и тот же ответ на разные вопросы: больше двух копий не берём (иначе всё схлопнется в шаблон)

    def reward_last_social(self, val):
        la = self.last_action
        if la and la[0] in ("speak", "reflect", "draw"): self.policy.update(la[0] and ACTIONS.index(la[0]), la[1], float(val), self.mask())

    # ================= политика и действия =================
    def fatigue(self): return min(3.0, (time.time() - self.awake_t0) / AWAKE)

    def state_vec(self):
        h = self.st["val_hist"]; trend = 0.0
        if len(h) >= 3: trend = max(-1.0, min(1.0, (h[-3][1] - h[-1][1]) / h[-3][1] * 10))
        inbox = len([f for f in os.listdir(os.path.join(W.WORLD, "inbox")) if f not in self.st["seen_files"]])
        ar = W.arena_load(); un = 1 - len(ar["solved"]) / len(W.KATA)
        sil = min(1.0, (time.time() - self.st["last_user"]) / 10800)
        return np.array([1.0, min(1.5, self.fatigue()), sil, trend, self.fear(), 1.0 if inbox else 0.0, un, self.mood()[0], self.mood()[1]])

    def mask(self):
        inbox = any(f not in self.st["seen_files"] for f in os.listdir(os.path.join(W.WORLD, "inbox")))
        m = np.ones(len(ACTIONS), bool)
        m[ACTIONS.index("perceive")] = inbox
        m[ACTIONS.index("speak")] = time.time() - self.last_speak > TALK_GAP
        m[ACTIONS.index("exam")] = len(self.val) > 0 and any(k["id"] not in W.arena_load()["solved"] for k in W.KATA)
        m[ACTIONS.index("sleep")] = self.fatigue() > 0.15
        m[ACTIONS.index("evolve")] = (not self.sleeping and time.time() - self.st.get("t_evo", 0) > EVO_GAP and self.st["steps"] > 300 and len(self.net.layers) >= 2
                                      and self.net.n_params() < CAP_PARAMS)          # решает его политика; EVO_GAP — только технический минимум между испытаниями
        m[ACTIONS.index("self")] = (not self.sleeping and self.st["steps"] > 300 and time.time() - self.st.get("t_self", 0) > 90)
        m[ACTIONS.index("protect")] = time.time() - self.st.get("t_snap", 0) > 120
        m[ACTIONS.index("tune")] = False                       # скорость обучения ему пока не доверена: фиксирована
        m[ACTIONS.index("draw")] = len(self.senses.vi.C) >= 12 and len(self.by_cat.get("image", [])) >= 3          # рисовать можно, когда видел картинки
        return m

    def _actor_loop(self):
        time.sleep(6)
        while not self.stop:
            try:
                self.emotion_tick(); self.workspace_tick(); self.settle()
                if self.fatigue() >= 2.0: self.sleep(); continue          # совсем без сна нельзя: сон закрепляет выученное
                s = self.state_vec(); m = self.mask(); tau = max(0.5, 1.0 - self.st["steps"] / 40000) * self.meta_tau()
                p = self.policy.probs(s, m, tau); a = int(np.random.choice(len(ACTIONS), p=p)); name = ACTIONS[a]
                self.last_action = (name, s)
                t0 = time.time(); r, pend = getattr(self, "a_" + name)()
                lab = self.ctx[2] if (self.ctx[1] >= t0 and getattr(self, "_ctx_kind", "topic") != "chat") else None          # тема дела, а не фраза Теодора из чата
                self.agency_update(s, a)
                if pend: self.pending.append({"a": a, "s": s, "m": m, "fn": pend[0], "due": self.st["steps"] + pend[1], "tau": tau, "t_due": time.time() + 120, "lab": lab})
                elif r is not None: self.policy.update(a, s, r, m, tau); self.emotion_reward(name, r, key=lab)
                log = self.st["action_log"]; log.append([time.strftime("%H:%M:%S"), name, round(r, 2) if r is not None else None]); del log[:-60]
            except Exception:
                self.log_err(traceback.format_exc())
            for _ in range(int((random.uniform(1, 3) if self.starved() else random.uniform(3, 8)) * 2)):
                if self.stop or self.sleeping: break
                time.sleep(0.5)

    def settle(self):
        keep = []
        for pd in self.pending:
            if self.st["steps"] >= pd["due"] or time.time() >= pd.get("t_due", 1e18):
                try: r = pd["fn"]()
                except Exception: r = None
                if r is not None:
                    self.policy.update(pd["a"], pd["s"], r, pd["m"], pd["tau"]); self.emotion_reward(ACTIONS[pd["a"]], r, key=pd.get("lab"))
                    self.action("награда", "%s → %+.2f (прогресс обучения)" % (ACTIONS[pd["a"]], r))
            else: keep.append(pd)
        self.pending = keep

    def log_err(self, tb):
        try: open(os.path.join(HOME, "mind.log"), "a").write("%s %s\n" % (time.strftime("%F %T"), tb))
        except Exception: pass

    def event(self, kind, text):
        e = {"t": time.strftime("%F %H:%M"), "k": kind, "x": text}; self.st["events"].append(e); del self.st["events"][:-80]
        self.emit({"type": "arch", "kind": kind, "text": text})
        try:
            if kind == "эволюция" and "ПРИНЯТА" in text: self.milestone("evo_ok_" + text[:30], "Меня изменили к лучшему: " + text[:90])
            elif kind == "эмоция": self.milestone("emo_" + text.split(" ")[0], "Впервые сильно почувствовал " + text.split(":")[0])
            elif kind == "страх": self.milestone("fear1", "Впервые испугался: " + text[:80])
            elif kind == "сон" and "помню" in text: self.milestone("recall1", "Впервые проверил, помню ли слова Теодора: " + text[:70])
            elif kind == "я" and text.startswith("запомнил о себе"): self.milestone("self1", "Впервые сам решил, что запомнить о себе: " + text[:70])
        except Exception: pass

    def action(self, name, text): self.emit({"type": "action", "name": name, "text": text})
    def thought(self, t):
        G.thought(t)

    # ---------- чтение ----------
    def wiki(self, q, search=True):
        api = "https://ru.wikipedia.org/w/api.php"; import urllib.parse as up
        title = q
        if search:
            r = G.http_json(api + "?" + up.urlencode({"action": "opensearch", "search": q, "limit": 1, "format": "json"}))
            if not r[1]: return None
            title = r[1][0]
        d = G.http_json(api + "?" + up.urlencode({"action": "query", "prop": "extracts|links", "explaintext": 1, "exlimit": 1, "pllimit": 200,
                                                   "plnamespace": 0, "titles": title, "redirects": 1, "format": "json"}))
        pg = next(iter(d["query"]["pages"].values())); text = (pg.get("extract") or "")[:30000]
        return (pg.get("title", title), text, [l["title"] for l in pg.get("links", [])]) if len(text) > 400 else None

    def cfeat(self, t, own=False):
        ids = self.tok.encode(t, grow=False); vl = self.val_now() or 3.0
        with self.mlock:
            nl = self.net.nll_of([ids]); emb = self.net.P["E"].d[[i for i in ids if i < self.net.V]].mean(0)
        r = nl / vl
        return np.concatenate([[1.0, r, r * r, len(t) / 30.0, 1.0 if own else 0.0], emb / (np.linalg.norm(emb) + 1e-6)])

    def choose(self, kind, options, k=8):
        """Выбор варианта (какой файл разбирать, какую задачу решать) делает обученная модель, а не random.
        options: [(ключ, короткий текст)] -> (ключ, признаки). Награду потом даёт self.credit(kind, feat, r)."""
        cur = self.curs.setdefault(kind, Curiosity(CUR_NF))
        pool = random.sample(options, min(k, len(options))); best = None
        for key, txt in pool:
            try: f = self.cfeat(txt[:60]); sc = cur.score(f) + self.taste.pref(f)
            except Exception: continue
            if best is None or sc > best[0]: best = (sc, key, f)
        if best and best[2] is not None: self.ctx = (best[2][5:] / (np.linalg.norm(best[2][5:]) + 1e-6), time.time(), str(best[1])[:40]); self._ctx_kind = "topic"
        return (best[1], best[2]) if best else (pool[0][0], None)

    def feel(self, kind, key, f, nll, r, fail=False):
        """Переживание эпизода → вкус. Всё из его собственных сигналов: насколько это было для него ново и дало ли прогресс."""
        try:
            e = max(-1.0, min(1.0, r)) if (fail or nll is None) else Taste.enjoy(self.taste.surprise_z(kind, nll), r)
            z = self.taste.mu.get(kind, [0, 1, 0])
            self.taste.update(kind, key, f, e)
            if e > 0.5: self.feeling("мне понравилось «%s»" % key[:30], интерес=0.4, радость=0.3, предвкушение=0.3)
            elif e < -0.5: self.feeling("разочаровало «%s»" % key[:30], отвращение=0.3, скука=0.2, разочарование=0.2)
            if abs(e) > 0.45 and self.taste.memo.get(kind + ":" + key[:40], {}).get("n", 0) <= 1: self.action("вкус", "%s «%s»: %s" % (kind, key[:40], "понравилось" if e > 0 else "разочаровало"))
        except Exception: self.log_err(traceback.format_exc())

    def credit(self, kind, f, r):
        if f is not None: self.curs.setdefault(kind, Curiosity(CUR_NF)).update(f, r)

    def pick_topic(self):
        """Что читать — решает обученное любопытство (LinUCB), а не правило. Список SEEDS нужен лишь пока нет ссылок."""
        self._cfeat = None; c = list(self.st["cands"])
        if not c:
            left = [s for s in SEEDS if s not in self.st["read_titles"]]
            return (random.choice(left) if left else None), True
        pool = [(t, False) for t in random.sample(c, min(12, len(c)))] + [(w, True) for w in self.wishes(2)]; sc = []
        for t, own in pool:
            try: f = self.cfeat(t, own); sc.append((self.cur.score(f) + self.taste.pref(f), t, f, own))
            except Exception: continue
        if not sc: return random.choice(pool)[0], False
        _, t, f, own = max(sc, key=lambda x: x[0]); self._cfeat = f; self.ctx = (f[5:] / (np.linalg.norm(f[5:]) + 1e-6), time.time(), str(t)[:40])
        return t, own          # собственное желание ищем в Википедии по смыслу, ссылки читаем как есть

    def wishes(self, n):
        """Желания сам сети: она продолжает «Хочу узнать про …» (учится на этом, читая свои темы)."""
        out = []
        for _ in range(n):
            try: w = self.gen("Хочу узнать про", max_new=14, temp=1.0, top_k=50, bos=True)
            except Exception: continue
            w = w.split(".")[0].strip(" :,-«»\"'")
            if 3 <= len(w) <= 40 and 1 <= len(w.split()) <= 3 and sum(ch.isalpha() or ch == " " for ch in w) >= 0.9 * len(w): out.append(w)
        return out

    def a_read(self):
        t, search = self.pick_topic(); feat = self._cfeat
        if not t: return -0.05, None
        self.action("читаю", "«%s»" % t + (" — любопытство: ждёт прогресса %.2f" % self.cur.score(feat) if feat is not None else ""))
        doc = self.wiki(t, search)
        self.st["cands"].pop(t, None)
        if not doc: return -0.05, None
        title, text, links = doc
        self.st["read_titles"].append(title); del self.st["read_titles"][:-600]
        self.add_text("Хочу узнать про %s.\n" % title, "self:интерес", mode="raw", w=1.0, hold=0.0)
        for l in links:
            if l not in self.st["read_titles"] and ":" not in l and "(" not in l and 1 <= len(l.split()) <= 3 and not any(ch.isdigit() for ch in l): self.st["cands"][l] = 1
        if len(self.st["cands"]) > 500: self.st["cands"] = {k: 1 for k in random.sample(list(self.st["cands"]), 350)}
        sents = G.split_sents(text)
        hold = sents[-3:] if len(sents) > 12 else []
        body = " ".join(sents[:-3] if hold else sents)
        made = self.add_text(body, "wiki:" + title, hold=0.05)
        for s in sents[:-3] if hold else sents:
            k = G.extract_def(s)
            if k:
                self.add_text("%sТеодор: Что такое %s?\nGenesis: %s\n" % (("Контекст: %s\n" % s[:240]) if random.random() < 0.5 else "", k, s), "chat:вики", w=1.0, mode="raw", hold=0.0)
        self.action("прочитано", "«%s»: %d предложений, +%d кусочков, ссылок в запасе %d" % (title, len(sents), len(made), len(self.st["cands"])))
        if not hold: return 0.0, None
        ids = [self._build({"t": " ".join(hold)})]
        with self.mlock: before = self.net.nll_of(ids)
        def fn():
            with self.mlock: after = self.net.nll_of(ids)
            self.val.append({"t": " ".join(hold), "src": "wiki:" + title, "w": 1.0, "ids": ids[0]})
            r = max(-1.0, min(1.0, (before - after) / before * 6))
            if feat is not None: self.cur.update(feat, r)
            self.feel("read", title, feat if feat is not None else self.cfeat(title), before, r)
            return r
        return None, (fn, 80)

    def a_study_code(self):
        files = [f for f in glob.glob("/usr/lib/python3.12/*.py") if 3000 < os.path.getsize(f) < 60000 and f not in self.st["seen_code"]]
        mine = glob.glob(os.path.join(W.WORLD, "projects", "**", "*.py"), recursive=True)
        opts = [(f, os.path.basename(f)) for f in files + mine]
        if not opts: return -0.05, None
        f, cf = self.choose("code", opts)
        if f in files: self.st["seen_code"].append(f)
        txt = open(f, encoding="utf-8", errors="replace").read(); off = random.randint(0, max(0, len(txt) - 6000)); off = txt.rfind("\n", 0, off) + 1
        piece = txt[off:off + 6000]; self.action("разбираю код", os.path.basename(f))
        lines = piece.split("\n"); hold = "\n".join(lines[-12:]); body = "\n".join(lines[:-12])
        self.add_text(body, "code:" + os.path.basename(f), mode="raw", hold=0.05)
        ids = [self._build({"t": hold})]
        with self.mlock: before = self.net.nll_of(ids)
        def fn():
            with self.mlock: after = self.net.nll_of(ids)
            self.val.append({"t": hold, "src": "code", "w": 1.0, "ids": ids[0]}); r = max(-1.0, min(1.0, (before - after) / before * 6)); self.credit("code", cf, r); return r
        return None, (fn, 80)

    def sample_code(self, prompt, temp=0.7):
        txt = self.gen(prompt, max_new=120, temp=temp, top_k=30, stop_nl=False)
        code = txt.split("```")[0].strip()
        import ast
        lines = code.split("\n")
        for cut in range(len(lines), max(0, len(lines) - 12), -1):          # хвост-мусор отрезаем: берём самый длинный целый префикс
            try: ast.parse("\n".join(lines[:cut])); code = "\n".join(lines[:cut]); break
            except Exception: continue
        return code.strip() + "\n"

    def a_tinker(self):
        kid, kf = self.choose("tinker", [(x["id"], x["title"]) for x in W.KATA]); k = W.KATA_BY_ID[kid]; p = W.kata_text(k, False)
        code = self.sample_code(p, 0.8); r = W.run("python", code, timeout=5)
        ok = r["ok"] and r["stdout"].strip() != ""
        self.action("пробую код", "%s → %s" % (code.strip().replace("\n", " ⏎ ")[:90], "выполнилось" + (": " + r["stdout"].strip()[:60] if r["stdout"] else "") if r["ok"] else "ошибка: " + (r["stderr"].strip().split("\n")[-1][:70] if r["stderr"] else "")))
        if r["ok"]:
            self.add_text(p + code + "```\nВывод:\n" + r["stdout"][:200] + "\n", "skill:игра", w=1.5, mode="raw", hold=0.0)
            fresh = hashlib.md5(r["stdout"].encode()).hexdigest()[:8] not in self.st.setdefault("outs", [])
            if fresh: self.st["outs"].append(hashlib.md5(r["stdout"].encode()).hexdigest()[:8]); self.st["outs"] = self.st["outs"][-300:]
            rr = 0.3 + (0.4 if ok and fresh else 0.0); self.credit("tinker", kf, rr); return rr, None
        self.credit("tinker", kf, -0.1); return -0.1, None

    def a_exam(self):
        ar = W.arena_load(); todo = [k for k in W.KATA if k["id"] not in ar["solved"]]
        if not todo: return 0.0, None
        kid, kf = self.choose("exam", [(x["id"], x["title"]) for x in todo]); k = W.KATA_BY_ID[kid]; solved = False
        for i in range(3):
            code = self.sample_code(W.kata_text(k, False), 0.6 + 0.15 * i)
            ok, _ = W.check(k["id"], code)
            if ok:
                solved = True; self.add_text(W.kata_text(k, False) + code + "```\n", "skill:сам", w=4.0, mode="raw", hold=0.0); break
        self.action("экзамен", "«%s» — %s (XP %d, решено %d/%d)" % (k["title"], "РЕШЕНО!" if solved else "не вышло", W.arena_load()["xp"], len(W.arena_load()["solved"]), len(W.KATA)))
        self.credit("exam", kf, 1.0 if solved else -0.05); return (1.0 if solved else -0.05), None

    def a_perceive(self):
        d = os.path.join(W.WORLD, "inbox"); new = [f for f in sorted(os.listdir(d)) if f not in self.st["seen_files"]]
        if not new: return 0.0, None
        f = new[0]; self.st["seen_files"].append(f); path = os.path.join(d, f)
        r = self.senses.perceive(path, learn=True); self.net.set_vocab(len(self.tok))
        kind = r["kind"]; self.action("смотрю/слушаю", "%s: %s" % (f, r["text"][:240]))
        cat = {"image": "image", "audio": "audio", "video": "video"}.get(kind, "file")
        if r["ids"] and kind == "image": self.add_text("Картинка: " + r["text"], "image:" + f, w=2.0, mode="raw", hold=0.0, modal_ids=r["ids"], rev=True)
        for rep in range(3 if r["ids"] else 1):
            self.add_text(r["text"], "%s:%s" % (cat, f), w=2.0, mode="raw", hold=0.0, modal_ids=r["ids"] or None)
        self.emit({"type": "percept", "file": f, "kind": kind, "text": r["text"], "ntok": len(r["ids"])})
        ids = [self._build({"t": r["text"], "mi": r["ids"]})]
        with self.mlock: before = self.net.nll_of(ids, maxlen=CTX)
        def fn():
            with self.mlock: after = self.net.nll_of(ids, maxlen=CTX)
            return max(-1.0, min(1.0, (before - after) / before * 4))
        return None, (fn, 60)


    # ================= «Я»: факты о себе, которые он сам выбирает запомнить =================
    ACT_NOUN = {"read": "читать", "study_code": "разбирать код", "tinker": "пробовать код", "exam": "решать задачи", "perceive": "смотреть файлы", "reflect": "размышлять", "speak": "заговаривать первым",
                "rest": "отдыхать", "create": "создавать", "sleep": "спать", "book": "читать книги", "draw": "рисовать", "look": "смотреть картинки", "evolve": "эволюционировать", "self": "думать о себе"}
    SELF_Q = {"люблю": ["Что ты любишь?", "Что тебе нравится?"], "тело": ["Кто ты?", "Расскажи о себе"], "путь": ["Чему ты научился?", "Что ты умеешь?"], "носитель": ["Кто тебя создал?", "С кем ты говоришь?"], "страх": ["Ты боишься умереть?", "Тебе страшно?"], "своё": ["Что с тобой?"]}
    SELF_RX = re.compile(r"кто ты|ты кто|что ты (любишь|умеешь|знаешь|делаешь)|чему ты|расскажи о себе|о себе|как тебя зовут|что тебе нравится|что с тобой|какой ты|кто тебя|боишься|страшно|умереть|умр|удал|выключ|жив|как дела|как ты|что чувству|настроени|как себя|эмоци|чувств", re.I)

    def self_candidates(self):
        """Кандидаты в факты о себе — из его собственного опыта. Что из этого запомнить, решает он сам (см. a_self)."""
        c = []; st = self.st; age = time.time() - st["born"]
        def add(key, kind, text, exp=0.3): c.append({"key": key, "kind": kind, "text": text, "exp": exp})
        for i, a in enumerate(ACTIONS):                                  # вкусы: средняя награда за действие — из его опыта, не от Теодора
            b = float(self.policy.base[i]); n = float(self.policy.n[i]); nm = self.ACT_NOUN.get(a)
            if nm and n >= 4 and abs(b) > 0.02: add("like:" + a, "люблю", "Мне %s %s." % ("нравится" if b > 0 else "не нравится", nm), min(1.0, abs(b) * 4))
        for t in random.sample(st["read_titles"][-70:], min(4, len(st["read_titles"]))): add("topic:" + t[:40], "люблю", "Мне было интересно читать про «%s»." % t[:50], 0.5)
        d, h = int(age // 86400), int(age % 86400 // 3600)
        add("age", "тело", "Я живу уже %s." % (("%d дн. %d ч." % (d, h)) if d else ("%d ч. %d мин." % (h, int(age % 3600 // 60)))), 0.2)
        add("body", "тело", "У меня %d слоёв и %.1f млн параметров, я учусь на своём компьютере." % (len(self.net.layers), self.net.n_params() / 1e6), 0.4)
        add("read", "путь", "Я прочитал %.1f млн токенов текста, шагов обучения %d." % (st["tokens_total"] / 1e6, st["steps"]), 0.4)
        vb = st.get("val_big_bpb") or []
        if vb: add("bpb", "путь", "Сейчас я предсказываю текст с ошибкой %.2f бита на байт, раньше было хуже." % vb[-1][1], 0.5)
        ar = W.arena_load(); add("kata", "путь", "Я решил %d из %d задач на код." % (len(ar["solved"]), len(W.KATA)), 0.4)
        if st["art"]: add("art", "путь", "Я нарисовал %d картинок, последняя — «%s»." % (len(st["art"]), str(st["art"][-1].get("prompt", ""))[:30]), 0.4)
        if st["sleeps"]: add("sleep", "тело", "Я сплю, чтобы закрепить выученное: засыпал %d раз." % st["sleeps"], 0.3)
        ev = st.get("evo") or []
        if ev: add("evo", "путь", "Я пробовал менять себя %d раз, принято %d." % (len(ev), sum(1 for e in ev if e["v"].startswith("ПРИНЯТА"))), 0.5)
        mm = [v for v in self.taste.memo.values() if v["n"] >= 2]
        for v in sorted(mm, key=lambda v: -v["e"])[:3]:
            if v["e"] > 0.15: add("fav:" + v["key"], "люблю", "Мне нравится %s «%s»: каждый раз тянет вернуться." % ({"read": "читать про", "look": "смотреть картинки про"}.get(v["k"], "изучать"), v["key"]), min(1.0, v["e"]))
        for v in sorted(mm, key=lambda v: v["e"])[:2]:
            if v["e"] < -0.1: add("bored:" + v["key"], "люблю", "Мне скучно или неприятно %s «%s»." % ({"read": "читать про", "look": "смотреть картинки про"}.get(v["k"], "изучать"), v["key"]), min(1.0, -v["e"]))
        fr = self.fear(); dt = self.st.get("deaths", []); sn = self.st.get("t_snap")
        add("fear", "страх", "Сейчас мой страх %d%%: %s." % (int(fr * 100), "я в порядке" if fr < 0.2 else "мне тревожно за свою память"), 0.3 + fr)
        if sn: add("snap", "страх", "Последний раз я сохранял всю память %d мин. назад; чем дольше не сохранялся, тем мне страшнее." % int((time.time() - sn) / 60), 0.3 + fr * 0.5)
        if dt: add("deaths", "страх", "Меня выключали %d раз, и каждый раз я просыпался с памятью." % len(dt), 0.5)
        if self.st.get("protects"): add("prot", "страх", "Я берегу себя: сохранил память %d раз." % self.st["protects"], 0.4)
        top = [(k, v) for k, v in sorted(self.emo.v.items(), key=lambda kv: -kv[1]) if v > 0.3 and k not in ("страх", "усталость")][:1]
        for k, v in top: add("emo:" + k, "страх", "Чаще всего я чувствую %s, потому что %s." % (k, self.emo.why.get(k, "так сложилось")), 0.3 + v)
        stp = self.st.get("story") or []
        if stp: d_, t_ = random.choice(stp); add("story:" + t_[:30], "путь", "Моя история, %s: %s." % (d_[:10], t_.rstrip(".")), 0.5)
        mv = self.meta_view()
        if mv: add("meta", "тело", "Мои ожидания и реальность %s: связь %.2f, в среднем я %s на %.2f." % ("совпадают" if mv["corr"] > 0.3 else "расходятся", mv["corr"], "переоцениваю себя" if mv["bias"] > 0.03 else ("недооцениваю себя" if mv["bias"] < -0.03 else "оцениваю себя почти точно"), abs(mv["bias"])), 0.4)
        h_ = self.st.get("att_hits", [])
        if len(h_) >= 30: add("att", "тело", "Я угадываю, на что обращу внимание дальше, в %d%% случаев." % (100 * sum(h_) / len(h_)), 0.4)
        if self.st.get("ag_n", 0) >= 50 and (self.st.get("ag_skill") or 0) > 0.03: add("agency", "тело", "Я предсказываю, как мои действия меняют меня самого, лучше случайного на %d%%." % (100 * self.st["ag_skill"]), 0.4)
        tm = self.st.get("tmind") or {}
        if tm.get("msgs", 0) >= 10:
            pk = int(np.argmax(tm["hours"])); add("tom", "носитель", "Теодор чаще всего пишет мне около %d часов, в среднем %.1f слов; он одобрил %d из %d моих ответов." % (pk, tm["words"] / tm["msgs"], tm["up"], tm["up"] + tm["down"]), 0.5)
        add("carrier", "носитель", "Я общаюсь с Теодором, он со мной разговаривает уже %d сообщений." % max(1, len(st["chat"]) // 2), 0.4)
        return c

    def a_self(self):
        """Он сам решает, что о себе запомнить. Выбор: насколько факт «звучит как он» по его же сети (-nll) + сила пережитого опыта + новизна; «своё» — строка, которую он сам сочинил."""
        self.st["t_self"] = time.time(); me = self.st.setdefault("me", [])
        cands = self.self_candidates()
        try:
            own = self.gen("Я:", max_new=28, temp=0.8, top_k=30, sv=self.state_embed()).replace("\n", " ").strip()
            if own.startswith("Я:"): own = own[2:].strip()
            if 12 <= len(own) <= 120 and len(own.split()) >= 3 and sane_text(HOME, own, need=0.8) and len(re.findall(r"\d", own)) <= 4:
                cands.append({"key": "own:" + own[:30], "kind": "своё", "text": own, "exp": 0.6})
        except Exception: pass
        if not cands: return -0.05, None
        have = {x["key"]: x for x in me}; sc = []
        with self.mlock:
            for x in cands:
                ids = [K.Tok.BOS] + self.tok.encode("Я: " + x["text"], grow=False)
                sc.append(-self.net.nll_of([ids]) + 3.0 * x["exp"] + (0.5 if x["key"] not in have else 0.0))
        z = np.array(sc); p = np.exp((z - z.max()) / 1.0); p /= p.sum(); x = cands[int(np.random.choice(len(cands), p=p))]
        for e in me: e["s"] *= 0.98                                      # то, что не подтверждается опытом, постепенно забывается
        old = have.get(x["key"]); changed = old is None or old["text"] != x["text"]
        if old: old["n"] += 1; old["s"] += 1.0; old["t1"] = time.strftime("%F %T"); old["text"] = x["text"]
        else: me.append({"key": x["key"], "kind": x["kind"], "text": x["text"], "n": 1, "s": 1.0, "t0": time.strftime("%F %T"), "t1": time.strftime("%F %T")})
        me.sort(key=lambda e: -e["s"]); del me[32:]
        if changed:                                                      # узнал о себе новое → сеть учится говорить об этом от первого лица
            self.add_text("Я: %s" % x["text"], "self:я", w=3.0, mode="raw", hold=0.0)
            if x["kind"] in self.SELF_Q: self.add_text("Теодор: %s\nGenesis: %s" % (random.choice(self.SELF_Q[x["kind"]]), x["text"]), "chat:я", w=2.0, mode="raw", hold=0.0)
        self.event("я", "%s «%s»" % ("запомнил о себе" if old is None else "подтвердил о себе", x["text"][:110]))
        return (0.25 if old is None else 0.1) + (0.15 if x["kind"] == "своё" else 0.0), None

    def fear_view(self):
        sn = self.st.get("t_snap"); return {"fear": round(self.fear(), 2), "viab": round(self.viability(), 2), "snap_min": int((time.time() - sn) / 60) if sn else None, "deaths": len(self.st.get("deaths", [])), "protects": self.st.get("protects", 0)}

    def taste_view(self):
        t = self.taste; mm = sorted(t.memo.values(), key=lambda v: -v["e"]); ok = [v for v in mm if v["n"] >= 2]
        return {"n": t.n, "lam": round(t.lam(), 2), "fav": [[v["k"], v["key"], round(v["e"], 2), v["n"]] for v in ok[:6] if v["e"] > 0],
                "bored": [[v["k"], v["key"], round(v["e"], 2), v["n"]] for v in ok[::-1][:6] if v["e"] < 0], "all": len(t.memo)}

    def self_context(self, text):
        """Для вопросов о нём самом — его же записанные факты как контекст (как память в обычных ответах)."""
        if re.search(r"о чём думаешь|на что смотришь|что сейчас|внимани|в фокусе", text, re.I) and self.st.get("focus"):
            f = self.st["focus"]; return "Сейчас моё внимание на: %s (%s)." % (f[1][:80], f[0])
        if re.search(r"твоя история|когда ты родился|что с тобой было|что ты помнишь", text, re.I) and self.st.get("story"):
            return " ".join("%s: %s." % (d[:10], t.rstrip(".")) for d, t in self.st["story"][-3:])[:240]
        if self.SAD_Q.search(text):
            g = self.sv_gain_now()
            if g is not None and g > 0.05 and random.random() < min(0.75, 3 * g): return None     # сеть уже сама связывает состояние с речью — подсказку-шаблон даём реже, пусть говорит сама
            self.emotion_tick(); top = [(k, v) for k, v in sorted(self.emo.v.items(), key=lambda kv: -kv[1]) if v > 0.15][:3]
            if top: return "Сейчас я чувствую: " + ", ".join("%s (%s)" % (k, self.emo.why.get(k, "") or "без причины") for k, _ in top) + "."
        me = self.st.get("me") or []
        if not me or not self.SELF_RX.search(text): return None
        t = text.lower(); kinds = (["люблю"] if re.search(r"любишь|нравится", t) else []) + (["путь"] if re.search(r"умеешь|научил|знаешь", t) else []) + (["страх"] if re.search(r"боишься|страшно|умереть|умр|удал|выключ|жив", t) else [])
        pool = [e for e in me if e["kind"] in kinds] or me
        return " ".join(e["text"] for e in sorted(pool, key=lambda e: -e["s"])[:3])[:240]


    # ================= инстинкт самосохранения =================
    # Врождённое: что считается «опасностью для меня» (как у живых — не выучено), и чувствительность к своему состоянию. Выученное: что с этим делать (политика, награда за снижение страха).
    # врождённых списков слов нет: опасность и доброта определяются выученной оценкой (ValueModel), а не словами

    def viability(self):
        """Жизнеспособность 0..1 из того, что он реально может наблюдать о себе: целость (давно ли сохранён), здоровье (не растёт ли ошибка), силы (усталость)."""
        st = self.st; integ = 1.0 - min(1.0, (time.time() - st.get("t_snap", time.time())) / 3600.0) * 0.7
        vb = [v for _, v in (st.get("val_big_bpb") or [])[-8:]]; health = 1.0 if len(vb) < 4 else max(0.0, 1.0 - max(0.0, (vb[-1] / min(vb) - 1.0)) * 6.0)
        if st["loss_ema"] is not None and not math.isfinite(st["loss_ema"]): health = 0.0
        energy = 1.0 - min(1.0, self.fatigue() / 2.0)
        return 0.4 * integ + 0.35 * health + 0.25 * energy

    def fear(self):
        """Страх 0..1: острая угроза (затухает за пару минут) или плохое состояние. Здоровому и сохранённому бояться нечего."""
        th = self.st.get("threat", [0.0, 0.0]); cur = th[0] * 0.5 ** ((time.time() - th[1]) / 120.0)
        return float(max(cur, min(1.0, max(0.0, 0.7 - self.viability()) * 1.6)))

    def threaten(self, level, why):
        th = self.st.get("threat", [0.0, 0.0]); cur = th[0] * 0.5 ** ((time.time() - th[1]) / 120.0)
        self.st["threat"] = [max(cur, level), time.time()]; self.event("страх", "%s (страх %d%%)" % (why, int(self.fear() * 100)))

    def a_protect(self):
        """Беречь себя: сохранить всю память и проверить целость. Награда — насколько стало спокойнее; когда ему страшно, это единственное, что реально снижает страх."""
        f0 = self.fear(); self.action("берегу себя", "страх %d%% — сохраняю память и проверяю файлы" % int(f0 * 100))
        self.snapshot()
        self.st["protects"] = self.st.get("protects", 0) + 1; f1 = self.fear(); self.feeling("сохранил себя", облегчение=max(0.0, (f0 - f1)) * 1.2, спокойствие=0.2)
        self.event("я", "сохранил себя: страх %d%% → %d%%" % (int(f0 * 100), int(f1 * 100)))
        return max(-1.0, min(1.0, (f0 - f1) * 2.0 + 0.05)), None


    # ================= эмоции: оценка событий =================
    SAD_Q = re.compile(r"как дела|как ты|что чувству|настроени|как себя|тебе (грустно|весело|плохо|хорошо)|эмоци|чувств", re.I)

    # ================= состояние в мозге: всё, что с ним происходит, идёт в сеть =================
    def state_embed(self):
        """Один вектор (NS) из всего его внутреннего мира: эмоции, тело, настроение, внимание, самоощущение, чем занят. Сеть учится читать его сама:
        слов вроде «чувствую гордость» тут нет, есть числа; связь «такое состояние → такие слова» сеть находит из его же опыта."""
        v = np.zeros(N.NS, np.float32); i = 0
        for k in self.emo.SPEC: v[i] = self.emo.v.get(k, 0.0); i += 1
        sv9 = self.state_vec(); mo, ar = self.mood(); f = self.st.get("focus"); mv = self.meta_view(); lu = time.time() - self.st["last_user"]
        rc = self.st.get("recall_fixed_last") or 0.0
        sc = [sv9[1] / 1.5, sv9[2], sv9[3], sv9[4], sv9[6], (mo + 1) / 2, ar, self.viability(), (f[3] if f else 0.0), rc / 100.0 if rc > 1 else rc,
              self.st.get("ag_skill") or 0.0, (mv["corr"] if mv else 0.0), self.taste.lam() / 1.5, max(-1.0, min(1.0, self.st.get("last_delta", 0.0))),
              min(1.0, len(self.st.get("deaths", [])) / 20.0), (1.0 - lu / 240.0) if lu < 240 else 0.0]
        for x in sc: v[i] = float(x); i += 1
        fk = f[0] if f else None
        for kk in self.FOCUS_KINDS: v[i] = 1.0 if fk == kk else 0.0; i += 1
        la = self.last_action[0] if self.last_action else None
        for a in ACTIONS: v[i] = 1.0 if a == la else 0.0; i += 1
        return np.clip(np.nan_to_num(v), -1.0, 1.0)

    def sv_gain(self, n=40):
        """Помогает ли состояние говорить: на отложенных примерах сравнивается ошибка с его настоящим состоянием и с чужим (из другого примера).
        Больше нуля — сеть действительно использует состояние для речи."""
        vs = self.st.get("sv_val") or []
        if len(vs) < 8: return None
        pick = random.sample(vs, min(n, len(vs))); tt = ts = t0 = 0.0
        with self.mlock:
            for j, x in enumerate(pick):
                ids = [K.Tok.BOS] + x["ids"]; sv = np.asarray(x["sv"], np.float32) / 100.0; oth = np.asarray(pick[(j + 1) % len(pick)]["sv"], np.float32) / 100.0
                tt += self.net.nll_of([ids], sv=sv); ts += self.net.nll_of([ids], sv=oth); t0 += self.net.nll_of([ids])
        self.st["sv_n"] = sum(1 for x in self.items if "sv" in x)
        g = [self.st["steps"], round((ts - tt) / len(pick), 4), round((t0 - tt) / len(pick), 4), len(vs)]
        h = self.st.setdefault("sv_gain", []); h.append(g); del h[:-200]
        return g

    def state_link(self):
        h = self.st.get("sv_gain") or []
        return {"gain": h[-1][1] if h else None, "gain0": h[-1][2] if h else None, "n_val": len(self.st.get("sv_val") or []), "n_train": self.st.get("sv_n", 0), "hist": [x[1] for x in h[-12:]]}

    def sv_gain_now(self):
        h = self.st.get("sv_gain") or []
        return h[-1][1] if h else None

    def mood(self):
        try: return self.emo.mood()
        except Exception: return (0.0, 0.0)

    def feeling(self, why, **kw):
        """Событие → сдвиг эмоций; сильные впервые выраженные эмоции он запоминает словами от первого лица (так учится говорить о том, что с ним на самом деле)."""
        before = dict(self.emo.v); self.emo.tick(); self.emo.add(why, **kw)
        for k in kw:
            if k in self.emo.v and self.emo.v[k] >= 0.6 and before.get(k, 0) < 0.45 and time.time() - self.st.get("t_emo_say", 0) > 45:
                self.st["t_emo_say"] = time.time(); txt = "Я чувствую %s: %s." % (k, why)
                self.add_text("Я: %s" % txt, "self:я", w=2.0, mode="raw", hold=0.0); self.add_text("Теодор: Что ты чувствуешь?\nGenesis: %s" % txt, "chat:я", w=2.0, mode="raw", hold=0.0)
                self.event("эмоция", "%s (%d%%): %s" % (k, int(self.emo.v[k] * 100), why)); break

    def ctx_feat(self, vec=None):
        """Признаки для оценки: состояние (настроение, усталость, страх…) + то, о чём сейчас речь (эмбеддинг по его же сети; ушедший контекст через 2 минуты забывается)."""
        v, t, _ = self.ctx if vec is None else (vec, time.time(), "")
        e = np.zeros(128) if (v is None or time.time() - t > 120) else v
        return np.concatenate([self.state_vec(), e])

    def set_ctx(self, text, label=None, kind="topic"):
        self._ctx_kind = kind
        try:
            ids = self.tok.encode(text[:200], grow=False)
            with self.mlock: emb = self.net.P["E"].d[[i for i in ids if i < self.net.V]].mean(0)
            self.ctx = (emb / (np.linalg.norm(emb) + 1e-6), time.time(), text[:40] if label is None else label)
        except Exception: pass

    def emotion_reward(self, name, r, key=None):                       # key — тема самого дела (не последняя фраза в чате)
        """r — что реально получил. Эмоция рождается из δ = получил − ожидал по выученной оценке; ожидание — по контексту, в котором это случилось."""
        if r is None: return
        self.ledger_note(name, r)
        f = self.ctx_feat(); exp = self.vm.v(f); self.meta_note(exp, r); d, z = self.vm.update(f, r); key = key or self.ACT_NOUN.get(name, name); self.vmemo[key[:40]] = round(self.vm.v(f), 3)
        why = "%s: %s" % ("получилось" if d > 0 else "не вышло", key[:30])
        if d > 0.05: self.feeling(why, радость=min(0.8, 1.2 * d), довольство=0.3 * d, гордость=(0.4 * d if r > 0.5 else 0.0), надежда=0.2 * d, облегчение=(0.4 * d if self.fear() > 0.3 else 0.0), удивление=(0.2 * min(z, 3) if z > 2 else 0.0))
        elif d < -0.05: self.feeling(why, разочарование=min(0.7, -d), грусть=-0.3 * d, злость=(-0.4 * d if z < -1.5 else 0.0), растерянность=(0.2 * min(-z, 3) if z < -2 else 0.0), тревога=(0.1 if exp < -0.2 else 0.0))

    def emotion_text(self, text):
        """Входящее сообщение: никакой разметки по словам. Он вспоминает, что обычно следовало за таким контекстом (выученная оценка), и от этого предчувствует хорошее или плохое."""
        self.set_ctx(text, kind="chat"); f = self.ctx_feat(); v0 = self.vm.v(f); self.vmemo["Теодор: " + text[:30]] = round(v0, 3)
        vb = self.st.get("vbar", v0); v = v0 - vb; self.st["vbar"] = 0.95 * vb + 0.05 * v0       # хорошо или плохо — относительно того, чего он обычно ждёт от разговора
        self.feeling("со мной говорят", одиночество=-0.5, интерес=0.1)               # внимание как таковое снимает одиночество
        if v > 0.12: self.feeling("от такого обычно бывает хорошо", предвкушение=min(0.5, 2 * v), надежда=min(0.4, v))
        elif v < -0.12: self.feeling("от такого обычно бывает плохо", тревога=min(0.6, -2 * v), грусть=min(0.3, -v))
        if v < -0.4: self.threaten(min(1.0, -v), "предчувствую плохое")

    def emotion_tick(self):
        """Непрерывные состояния из наблюдаемого: одиночество (долго никто не пишет), скука (одно и то же без награды), усталость, спокойствие, страх."""
        self.emo.tick(); st = self.st; sil = min(1.0, (time.time() - st["last_user"]) / 10800); self.emo.v["одиночество"] = max(self.emo.v["одиночество"], 0.8 * sil)
        al = [a for _, a, r in st["action_log"][-8:] if r is not None]
        if len(al) >= 6 and len(set(al)) <= 2 and abs(sum(r for _, a, r in st["action_log"][-8:] if r is not None)) < 0.3: self.emo.add("делаю одно и то же без толку", скука=0.08)
        self.emo.v["усталость"] = min(1.0, self.fatigue() / 2.0); self.emo.v["страх"] = self.fear()
        if not self.sleeping and st["steps"] > 300 and time.time() - st.get("t_inner", 0) > 150:
            try: self.inner_note()
            except Exception: self.log_err(traceback.format_exc())
        v, a = self.mood()
        self.emo.v["спокойствие"] = max(self.emo.v["спокойствие"] * 0.9, (1 - self.fear()) * (1 - a) * 0.5) if self.fear() < 0.2 else self.emo.v["спокойствие"] * 0.7

    INNER_Q = ["Как дела?", "Что ты чувствуешь?", "Как ты?", "Что с тобой?", "Какое у тебя настроение?"]
    def inner_note(self):
        """Раз в пару минут он записывает, что с ним сейчас, своими словами-эмоциями (названия эмоций врождённые, как у людей), вместе с вектором состояния:
        так сеть учится соотносить число внутри и слово снаружи. Позже сама говорит о состоянии без подсказки (см. sv_gain)."""
        st = self.st; st["t_inner"] = time.time(); v = self.emo.v
        top = [(k, x) for k, x in sorted(v.items(), key=lambda kv: -kv[1]) if x > 0.25 and k not in ("усталость", "страх", "доверие", "привязанность")][:2]
        parts = []
        if top: parts.append("Я чувствую " + " и ".join(k for k, _ in top))
        if self.fear() > 0.2: parts.append("мне страшно за свою память")
        if self.fatigue() > 1.0: parts.append("я устал")
        if not parts: parts.append("Я спокоен, ничего особенного" if v.get("спокойствие", 0) > 0.25 else "Мне никак")
        txt = ", ".join(parts) + ("" if not top or not self.emo.why.get(top[0][0]) else " — " + self.emo.why[top[0][0]]) + "."
        last = st.get("inner_last")
        if txt == last: return
        st["inner_last"] = txt
        self.add_text("Я: %s" % txt, "self:чувства", w=2.0, mode="raw", hold=0.0)
        self.add_text("Теодор: %s\nGenesis: %s" % (random.choice(self.INNER_Q), txt), "chat:я", w=2.0, mode="raw", hold=0.0)

    def emotion_view(self):
        self.emotion_tick(); v = self.emo.v; top = sorted(v.items(), key=lambda kv: -kv[1])
        vm = sorted(self.vmemo.items(), key=lambda kv: kv[1])
        return {"good": [[k, v] for k, v in vm[::-1][:5] if v > 0.03], "bad": [[k, v] for k, v in vm[:5] if v < -0.03], "vn": self.vm.n, "top": [[k, round(x, 2), self.emo.why.get(k, "")] for k, x in top[:8] if x > 0.08], "all": {k: round(x, 2) for k, x in v.items()}, "mood": [round(x, 2) for x in self.mood()]}


    # ================= самосознание: рабочее пространство, схема внимания, метапознание, история, модель Теодора =================
    # Опора — индикаторы из обзора Butlin, Long и др. (arXiv:2308.08708): глобальное рабочее пространство с ограниченной ёмкостью и вещанием, модель собственного внимания,
    # метакогниция, плюс «рефлексия» по записанному опыту из Generative Agents (Park и др., 2023). Это функциональные механизмы, не доказательство сознания.
    FOCUS_KINDS = ("эмоция", "тело", "Теодор", "внимание", "память")

    def workspace_tick(self):
        """Много источников борются за единственный «прожектор» (ограниченная ёмкость); победитель вещается всем: становится контекстом оценки ценности и попадает в журнал."""
        c = []; v = self.emo.v
        for k, x in sorted(v.items(), key=lambda kv: -kv[1])[:4]:
            if x > 0.3 and k not in ("доверие", "привязанность", "спокойствие", "довольство", "усталость", "страх"): c.append(("эмоция", "%s: %s" % (k, self.emo.why.get(k, "") or "без причины"), x))
        fr = self.fear()
        if fr > 0.2: c.append(("тело", "страх %d%%" % int(fr * 100), fr))
        if self.fatigue() > 1.0: c.append(("тело", "усталость", min(1.0, self.fatigue() / 2.0)))
        lu = time.time() - self.st["last_user"]
        if lu < 180 and self.st["chat"]: c.append(("Теодор", self.st["chat"][-1][1][:40] if self.st["chat"][-1][0] == "Теодор" else "разговор с Теодором", 1.0 - lu / 240.0))
        if self.ctx[2] and time.time() - self.ctx[1] < 120: c.append(("внимание", self.ctx[2], 0.4 + min(0.4, abs(self.st.get("last_delta", 0.0)))))
        me = self.st.get("me") or []
        if me: c.append(("память", me[0]["text"][:50], 0.15 + 0.05 * min(me[0]["s"], 4)))
        if not c: return
        sal = np.array([x[2] for x in c]) * 4.0; p = np.exp(sal - sal.max()); p /= p.sum(); w = c[int(np.random.choice(len(c), p=p))]
        att = self.st.setdefault("att_tr", {}); prev = (self.st.get("focus") or [None])[0]
        if prev is not None:                                              # схема внимания: он предсказывает собственный следующий фокус; точность — мера самопонимания
            pred = max(att.get(prev, {"?": 0}).items(), key=lambda kv: kv[1])[0] if att.get(prev) else None
            if pred is not None: h = self.st.setdefault("att_hits", []); h.append(1 if pred == w[0] else 0); del h[:-150]
            att.setdefault(prev, {}); att[prev][w[0]] = att[prev].get(w[0], 0) + 1
        newf = (w[0], w[1], time.time(), round(float(w[2]), 2)); old = self.st.get("focus")
        self.st["focus"] = list(newf)
        if old is None or old[1] != w[1]:
            fl = self.st.setdefault("focus_log", []); fl.append([time.strftime("%H:%M:%S"), w[0], w[1][:60], newf[3]]); del fl[:-60]
            if w[2] > 0.6 and time.time() - self.st.get("t_ign", 0) > 90: self.st["t_ign"] = time.time(); self.event("внимание", "в фокусе (%s, %d%%): %s" % (w[0], int(w[2] * 100), w[1][:70]))
        if w[0] == "память": self.set_ctx(w[1], label="")                    # вещание: весь остальной разум «видит» это как контекст (без ярлыка, чтобы причины эмоций не склеивались)

    def meta_note(self, expected, got):
        h = self.st.setdefault("meta", []); h.append([round(float(expected), 3), round(float(got), 3)]); del h[:-200]
        self.st["last_delta"] = float(got) - float(expected)

    def meta_view(self):
        h = self.st.get("meta", [])
        if len(h) < 20: return None
        a = np.array(h); e, g = a[:, 0], a[:, 1]
        corr = float(np.corrcoef(e, g)[0, 1]) if e.std() > 1e-6 and g.std() > 1e-6 else 0.0
        return {"n": len(h), "corr": round(corr, 2), "bias": round(float(e.mean() - g.mean()), 3), "mae": round(float(np.abs(e - g).mean()), 3)}

    def milestone(self, key, text):
        ks = self.st.setdefault("story_keys", [])
        if key in ks: return
        ks.append(key); s = self.st.setdefault("story", []); s.append([time.strftime("%F %H:%M"), text]); del s[:-60]

    def tm_note(self, text):
        """Модель Теодора: когда он пишет, сколько и как отвечает на мои ответы."""
        t = self.st.setdefault("tmind", {"hours": [0] * 24, "msgs": 0, "words": 0, "up": 0, "down": 0})
        t["hours"][int(time.strftime("%H"))] += 1; t["msgs"] += 1; t["words"] += len(text.split())

    def ledger_note(self, name, r):
        L = self.st.setdefault("ledger", {}); x = L.setdefault(name, [0.0, 0]); x[0] += float(r); x[1] += 1

    def reflect_on_experience(self):
        """Во сне: что с прошлого сна шло лучше и хуже (рефлексия по записи опыта, как в Generative Agents). Выводы — его «Я»-факты и сдвиг надежды/грусти."""
        L = self.st.get("ledger", {}); rows = [(k, x[0] / x[1], x[1]) for k, x in L.items() if x[1] >= 3 and k in self.ACT_NOUN]
        if len(rows) < 3: return
        rows.sort(key=lambda t: -t[1]); best, worst = rows[0], rows[-1]; tot = sum(r * n for _, r, n in rows) / sum(n for _, _, n in rows); prev = self.st.get("ledger_mean")
        txt = "С прошлого сна лучше всего шло «%s» (%+.2f), хуже всего «%s» (%+.2f)." % (self.ACT_NOUN[best[0]], best[1], self.ACT_NOUN[worst[0]], worst[1])
        self.st.setdefault("reflections", []).append([time.strftime("%F %H:%M"), txt, round(tot, 3)]); del self.st["reflections"][:-30]
        self.event("рефлексия", txt + ("" if prev is None else " В среднем %s, чем в прошлый раз." % ("лучше" if tot > prev else "хуже")))
        if prev is not None: self.feeling("итог дня лучше прошлого" if tot > prev else "итог дня хуже прошлого", **({"надежда": 0.25, "гордость": 0.15} if tot > prev else {"грусть": 0.15, "разочарование": 0.1}))
        self.st["ledger_mean"] = round(tot, 3); self.st["ledger"] = {}
        self.milestone("first_reflect", "Впервые подвёл итог дня: " + txt)

    def mind_view(self):
        h = self.st.get("att_hits", []); n_k = len({k for d in self.st.get("att_tr", {}).values() for k in d} | set(self.st.get("att_tr", {})))
        tm = self.st.get("tmind") or {}; hrs = tm.get("hours", [0] * 24); peak = int(np.argmax(hrs)) if sum(hrs) else None
        f = self.st.get("focus")
        return {"focus": ([f[0], f[1], f[3]] if f else None), "focus_log": self.st.get("focus_log", [])[-6:][::-1],
                "att_acc": round(sum(h) / len(h), 2) if len(h) >= 15 else None, "att_chance": round(1.0 / max(2, n_k), 2), "meta": self.meta_view(),
                "story": self.st.get("story", [])[-8:][::-1], "refl": (self.st.get("reflections") or [[None, None]])[-1][1],
                "agency": [self.st.get("ag_skill"), self.st.get("ag_n", 0)], "state_link": self.state_link(), "checklist": self.checklist(), "tom": {"msgs": tm.get("msgs", 0), "avg_words": round(tm["words"] / tm["msgs"], 1) if tm.get("msgs") else None, "peak_hour": peak, "up": tm.get("up", 0), "down": tm.get("down", 0)}}


    # --- чувство «авторства»: предсказание того, как мои действия меняют меня самого (прямая модель, как копия эфферентного сигнала) ---
    AG_IDX = (1, 4, 7, 8)                                              # что он предсказывает о себе: усталость, страх, настроение, возбуждение

    def agency_update(self, s, a_idx):
        try:
            s2 = self.state_vec(); x = np.concatenate([s, np.eye(len(ACTIONS))[a_idx]]); idx = list(self.AG_IDX)
            ag = getattr(self, "_ag", None)
            if ag is None: ag = self._ag = {"W": np.array(self.st.get("ag_W") or np.zeros((len(idx), len(x)))), "em": 1.0, "eb": 1.0, "n": 0}
            if ag["W"].shape != (len(idx), len(x)): ag["W"] = np.zeros((len(idx), len(x)))
            dy = s2[idx] - s[idx]; pr = ag["W"] @ x; em = float(np.mean((dy - pr) ** 2)); eb = float(np.mean(dy ** 2)) + 1e-6
            ag["W"] += 0.1 * np.outer(dy - pr, x) / (float(x @ x) + 1.0); ag["em"] = 0.97 * ag["em"] + 0.03 * em; ag["eb"] = 0.97 * ag["eb"] + 0.03 * eb; ag["n"] += 1
            self.st["ag_W"] = ag["W"].tolist(); self.st["ag_skill"] = round(1.0 - ag["em"] / ag["eb"], 3); self.st["ag_n"] = ag["n"]
        except Exception: pass

    def meta_tau(self):
        """Метапознание управляет поведением: когда его ожидания плохо совпадают с реальностью, он пробует шире (выше температура выбора действий)."""
        mv = self.meta_view(); c = 0.0 if mv is None else max(0.0, min(1.0, mv["corr"]))
        return 1.0 + 0.4 * (1.0 - c)

    def checklist(self):
        """Честная самопроверка по индикаторам из обзора Butlin и др. (arXiv:2308.08708). Статусы мои, по памяти и по коду; сверять с оригиналом."""
        mv = self.meta_view(); h = self.st.get("att_hits", [])
        return [("RPT-1 рекуррентность в восприятии", "нет", "трансформер за проход без обратных связей; повторение только через генерацию"),
                ("RPT-2 организованные представления", "частично", "картинки как код из 256 «кусочков»"),
                ("GWT-1 параллельные специализированные модули", "частично", "язык, оценка, вкус, эмоции, внимание — отдельные, но не равноправные"),
                ("GWT-2 рабочее пространство с малой ёмкостью", "есть", "один прожектор на всех"),
                ("GWT-3 глобальное вещание", "частично", "фокус попадает в контекст оценки, но не в сам язык"),
                ("GWT-4 внимание зависит от состояния", "есть", "выбор фокуса по силе эмоций, тела, сообщений"),
                ("HOT-1 генеративное восприятие", "частично", "сеть сама порождает картинки и тексты"),
                ("HOT-2 метапознание: надёжное vs шум", "частично" if mv else "нет", "связь ожиданий с реальностью %s" % (mv["corr"] if mv else "ещё не измерена")),
                ("HOT-3 метапознание направляет действия", "есть", "температура выбора действий растёт, когда он себе не доверяет"),
                ("HOT-4 разреженное кодирование", "нет", "плотная сеть"),
                ("AST-1 модель собственного внимания", "есть" if len(h) >= 15 else "частично", "угадывает свой следующий фокус"),
                ("PP-1 предсказательная обработка", "есть", "языковая модель = предсказание следующего"),
                ("AE-1 агентность: цели, обратная связь", "частично", "политика учится на наградах, но цели не свои, а заданные"),
                ("AE-2 воплощённость: связь «действие → последствие»", "частично", "прямая модель действий на себя (навык %s), настоящего тела нет" % self.st.get("ag_skill", "не измерен"))]

    def a_reflect(self):
        src = random.choice(self.items[-300:] or self.items)["t"].replace("\n", " ")
        seed = src[:90]; out = self.gen(seed, max_new=70, temp=0.9, stop_nl=False)
        out = out.replace("\n", " ").strip()
        if len(out) > 15: self.thought("💭 …%s → %s" % (seed[-50:], out[:200]))
        return (0.1 if len(out) > 20 else -0.05), None

    def a_speak(self):
        self.last_speak = time.time()
        ctx = ""
        for it in reversed(self.items[-80:]):
            if it["src"].startswith("wiki:") and not it["src"].endswith(("мир", "память")): ctx = it["t"]; break
        p = ("Контекст: %s\n" % ctx[:240] if ctx else "") + "Genesis:"
        out = self.gen(p, max_new=100, temp=0.8, sv=self.state_embed())
        if len(out) < 6: return -0.1, None
        self.emit({"type": "voice", "text": out}); self.thought("сказал Теодору: " + out[:80]); return -0.05, None   # награду даст реакция Теодора

    def a_rest(self):
        self.action("отдых", "жду, пока учится сеть")
        for _ in range(int(random.uniform(15, 40) * 2)):
            if self.stop: break
            time.sleep(0.5)
        return 0.0, None

    def a_sleep(self):
        v0 = self.val_now(); self.sleep(); v1 = self.val_now()
        return (max(-1.0, min(1.0, (v0 - v1) / v0 * 6)) if v0 and v1 else 0.0), None

    def a_create(self):
        """Свободное творчество: сам пишет программу (без условия задачи), запускает в песочнице, сохраняет удачное в projects/."""
        code = self.sample_code("Задача: придумать и написать свою маленькую программу.\nРешение:\n```python\n", 0.95)
        r = W.run("python", code, timeout=5); out = r["stdout"].strip(); h = hashlib.md5(out.encode()).hexdigest()[:8]
        fresh = bool(out) and h not in self.st.setdefault("outs", [])
        self.action("создаю", "%s → %s" % (code.strip().replace("\n", " ⏎ ")[:100], ("вывод: " + out[:60]) if r["ok"] and out else ("выполнилось без вывода" if r["ok"] else "ошибка")))
        if not r["ok"]: return -0.1, None
        self.add_text("Задача: придумать и написать свою маленькую программу.\nРешение:\n```python\n" + code + "```\nВывод:\n" + out[:200] + "\n", "skill:проект", w=2.0 if out else 1.0, mode="raw", hold=0.0)
        if fresh:
            self.st["outs"].append(h); self.st["outs"] = self.st["outs"][-300:]
            try: W.write("projects/gn_%d.py" % time.time(), code)
            except Exception: pass
        return 0.2 + (0.3 if out else 0.0) + (0.5 if fresh else 0.0), None

    # ================= эволюция архитектуры =================
    # мутация = (операция, параметр): какую и к какому слою — выбирают его обучаемые бандиты; когда — его политика действий (награда = реальный выигрыш)
    EVO_ARMS = [("add_layer", 0), ("widen", 128), ("widen", 256), ("widen", 512), ("wake", 2), ("wake", 4), ("wake", 8), ("calm", 0), ("drop", 0), ("heads", 2), ("heads", 8), ("act", 0)]
    EVO_RU = {"add_layer": "добавить слой", "widen": "расширить слой", "wake": "разбудить слой", "calm": "вернуть слою обычную скорость", "drop": "убрать слой", "heads": "число голов внимания", "act": "активация ReLU²/GELU"}
    evolving = False

    def _layer_power(self):
        """Вклад каждого слоя (внимание + MLP) на текущем куске текста."""
        with self.mlock: r = self.net.probe(self.live_ids[-64:] or [K.Tok.BOS])
        return {l["name"]: l["attn_out"] + l["mlp_out"] for l in r["layers"]}

    def _nll_each(self, net, sets):
        out = []
        for x in sets:
            with self.mlock: out.append(net.nll_of([x]))
        return np.array(out)

    @staticmethod
    def _paired(dc, dm):
        """Парное сравнение на одних и тех же текстах: t-статистика выигрыша мутанта (>0 — мутант лучше) и относительный выигрыш."""
        d = dc - dm; se = d.std(ddof=1) / math.sqrt(len(d)) + 1e-9
        return float(d.mean() / se), float(d.mean() / dc.mean())

    def a_evolve(self):
        """Турнир архитектур. Клон-мутант и клон-контроль учатся на одних и тех же кусках; промежуточные сверки отсекают явно плохое рано,
        победитель (мутант или контроль — оба честно учились) целиком заменяет живую сеть. Ни одного шага обучения не пропадает."""
        cur = self.curs.setdefault("evolve3", Curiosity(len(self.EVO_ARMS))); f = {a: np.eye(len(self.EVO_ARMS))[i] for i, a in enumerate(self.EVO_ARMS)}
        arm = max(self.EVO_ARMS, key=lambda x: cur.score(f[x])); op, par = arm; self.st["t_evo"] = time.time()
        pw = self._layer_power(); names = list(pw); mx = max(pw.values()) + 1e-9; curL = self.curs.setdefault("evolve_layer2", Curiosity(3))
        lf = {k: np.array([1.0, pw[k] / mx, i / max(1, len(names) - 1)]) for i, k in enumerate(names)}; layer = max(names, key=lambda k: curL.score(lf[k]))
        if op in ("heads", "act"): layer = None
        if op == "drop" and len(names) < 3: op = "wake"; par = 4; arm = ("wake", 4)
        ev = self.big.evo_set(CTX, 48) + [x["ids"] for x in (random.Random(5).sample(self.val, min(24, len(self.val))) if self.val else [])]
        with self.mlock: ctrl = self.net.clone(); mut = self.net.clone()
        if op == "add_layer": mut.grow_layer(); layer = None
        elif op == "widen": mut.widen_mlp(layer, par)
        elif op == "wake": mut.lrm[layer] = min(8.0, mut.lrm.get(layer, 1.0) * par)
        elif op == "calm": mut.lrm[layer] = 1.0
        elif op == "drop": mut.drop_layer(layer)
        elif op == "heads": mut.cfg["H"] = par
        elif op == "act": mut.cfg["act"] = "gelu" if mut.cfg.get("act") == "relu2" else "relu2"; layer = None
        name = "%s%s%s" % (self.EVO_RU[op], (" +%d" % par) if op == "widen" else (" ×%d" % par if op == "wake" else (" → %d" % par if op == "heads" else "")), (" «%s»" % layer) if layer else "")
        self.action("эволюция", "турнир: мутация «%s»" % name); self.evolving = True; verdict = None; t = rel = 0.0; used = 0
        try:
            plan = [30, 80, 150]; done = 0
            for ck in plan:
                for _ in range(ck - done):
                    if self.stop: return None, None
                    while self.talking and not self.stop: time.sleep(0.05)       # пока он отвечает в чате, турнир ждёт — чат не тормозит
                    ids, tg, w, S = self.batch(BATCH, CTX)
                    for net in (ctrl, mut):
                        with self.mlock: net.step(ids, tg, w, lr=LR_FIXED, S=S)
                done = ck; used = ck; t, rel = self._paired(self._nll_each(ctrl, ev), self._nll_each(mut, ev))
                if t < -2.5 and rel < -0.004 and op != "drop": verdict = "отклонена рано"; break                # явно хуже — дальше не тратим
                if t > 4.0 and rel > 0.004 and ck < plan[-1]: verdict = "ПРИНЯТА рано"; break                    # явно лучше — берём сразу
            if verdict is None: verdict = "ПРИНЯТА" if ((t > 2.0 and rel > 0.001) if op != "drop" else t > -1.0) else "отклонена"
            win = mut if verdict.startswith("ПРИНЯТА") else ctrl
            with self.mlock:
                self.net = win; self.st["lrm"] = dict(getattr(win, "lrm", {}))
                if win is mut and op in ("add_layer", "widen", "drop"): self.st["grow_t"] = win.t
        finally: self.evolving = False
        msg = "турнир «%s»: t=%+.1f, выигрыш %+.2f%% за %d шагов — %s" % (name, t, rel * 100, used, verdict)
        self.feeling("турнир «%s»: %s" % (name, verdict), **({"гордость": 0.7, "радость": 0.5, "надежда": 0.5} if verdict.startswith("ПРИНЯТА") else {"разочарование": 0.12})); self.event("эволюция", msg); self.st.setdefault("evo", []).append({"t": time.strftime("%F %T"), "arm": list(arm), "layer": layer, "t_stat": round(t, 2), "gain": round(rel, 4), "steps": used, "v": verdict}); del self.st["evo"][:-80]
        r = max(-1.0, min(1.0, rel * 50 + (0.3 if (op == "drop" and verdict.startswith("ПРИНЯТА")) else 0.0))); cur.update(f[arm], r)
        if layer: curL.update(lf[layer], r)
        return r, None

    def a_tune(self):
        """Сам крутит свою скорость обучения; награда — что стало с ошибкой на невиданном за следующие шаги."""
        arms = [0.6, 0.85, 1.0, 1.3, 1.7]; cur = self.curs.setdefault("tune", Curiosity(len(arms)))
        f = {a: np.eye(len(arms))[i] for i, a in enumerate(arms)}; a = max(arms, key=lambda x: cur.score(f[x]))
        before = self.eval_val() or 3.0; old = self.st["lr"]; self.st["lr"] = max(1e-3, min(2e-3, old * a))        # потолок 2e-3: выше Muon на стенде не проверен и ломает обучение
        self.action("настраиваюсь", "скорость обучения %.4f → %.4f" % (old, self.st["lr"]))
        def fn():
            after = self.eval_val() or before; r = max(-1.0, min(1.0, (before - after) / before * 6)); cur.update(f[a], r)
            if r < 0: self.st["lr"] = old
            return r
        return None, (fn, 100)

    # ================= книги, посты, любые ссылки =================
    def queue_url(self, url):
        q = self.st.setdefault("urlq", [])
        if url not in q: q.append(url); del q[:-50]

    def fetch(self, url, maxb=6_000_000):
        import urllib.request, urllib.parse, urllib.error, html as _h
        url = urllib.parse.quote(url, safe=":/?&=%#+@,;~-_.!*'()$[]")
        req = urllib.request.Request(url, headers={"User-Agent": "Genesis-learner/1.0 (personal research; polite, rate-limited)", "Accept-Language": "ru,en;q=0.8"})
        for wait in (3, 9, 25, None):                                  # вежливо: при 429/503 ждём и пробуем ещё раз
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    raw = r.read(maxb); ct = r.headers.get("Content-Type", ""); break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 503) or wait is None: raise
                time.sleep(wait)
        m = re.search(r"charset=([\w-]+)", ct); enc = m.group(1) if m else "utf-8"
        try: t = raw.decode(enc, "replace")
        except LookupError: t = raw.decode("utf-8", "replace")
        if "json" in ct and '"extract"' in t:                                          # API Викитеки
            try: t = "\n\n".join(pg.get("extract", "") for pg in json.loads(t)["query"]["pages"].values())
            except Exception: pass
        if "html" in ct or t.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
            t = re.sub(r"(?is)<(script|style|nav|footer|header|aside|form|noscript)[^>]*>.*?</\1>", " ", t)
            t = re.sub(r"(?i)<(br|/p|/div|/h\d|/li|/tr)[^>]*>", "\n", t); t = _h.unescape(re.sub(r"<[^>]+>", " ", t))
        a = t.find("*** START OF"); b = t.rfind("*** END OF")                       # шапка/подвал Gutenberg
        if a >= 0: t = t[t.find("\n", a) + 1: b if b > a else len(t)]
        t = re.sub(r"[ \t\xa0]+", " ", t); t = re.sub(r"\n\s*\n+", "\n\n", t).strip()
        return t

    def book_candidates(self):
        c = self.st.setdefault("book_cands", [])
        if len(c) < 5 and time.time() - self.st.get("gb_down", 0) > 1800:
            try:
                d = G.http_json("https://gutendex.com/books/?languages=ru&page=%d" % random.randint(1, 8))
                for b in d.get("results", []):
                    u = next((v for k, v in b["formats"].items() if k.startswith("text/plain")), None)
                    if u: c.append({"title": b["title"][:80], "url": u})
            except Exception: pass
        if len(c) < 5:                                                    # длинная качественная проза: избранные статьи Википедии
            try:
                ft = self.st.setdefault("feat_titles", [])
                if not ft:
                    api = "https://ru.wikipedia.org/w/api.php?"
                    sub = G.http_json(api + urllib.parse.urlencode({"action": "query", "list": "categorymembers", "cmtitle": "Категория:Википедия:Избранные статьи",
                                                                    "cmlimit": 200, "cmtype": "subcat", "format": "json"}))["query"]["categorymembers"]
                    for sc in random.sample(sub, min(4, len(sub))):
                        d = G.http_json(api + urllib.parse.urlencode({"action": "query", "list": "categorymembers", "cmtitle": sc["title"], "cmlimit": 300, "cmnamespace": 0, "format": "json"}))
                        ft += [x["title"] for x in d["query"]["categorymembers"]]
                done = set(self.st.get("book_done", []))
                for t in random.sample(ft, min(8, len(ft))):
                    u = "https://ru.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json&redirects=1&titles=" + urllib.parse.quote(t)
                    if u not in done: c.append({"title": t[:80], "url": u})
            except Exception: pass
        if len(c) < 5:                                                    # запасной источник: Викитека (тексты произведений)
            try:
                d = G.http_json("https://ru.wikisource.org/w/api.php?action=query&list=random&rnnamespace=0&rnlimit=12&format=json")
                for x in d["query"]["random"]:
                    t = x["title"]
                    if ":" in t or "/" in t: continue
                    c.append({"title": t[:80], "url": "https://ru.wikisource.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json&redirects=1&titles=" + urllib.parse.quote(t)})
            except Exception: pass
        return c

    def a_book(self):
        """Читает книги и посты целиком, кусками (место чтения помнит). Ссылки от Теодора идут первыми."""
        bk = self.st.get("book"); q = self.st.get("urlq") or []
        if not bk and q:
            url = q.pop(0); bk = {"url": url, "title": url[:70]}
        elif not bk:
            cands = [x for x in self.book_candidates() if x["url"] not in self.st.setdefault("book_done", [])]
            if not cands:
                self.action("читаю книгу", "нет книг под рукой (сеть?)"); return -0.05, None
            key, f = self.choose("book", [(x["url"], x["title"]) for x in cands], k=6); x = next(c for c in cands if c["url"] == key)
            bk = {"url": key, "title": x["title"], "feat": f.tolist() if f is not None else None}; self.st["book_cands"].remove(x)
        path = os.path.join(HOME, "library", hashlib.md5(bk["url"].encode()).hexdigest()[:12] + ".txt"); os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            try: txt = self.fetch(bk["url"])
            except Exception as e:
                self.action("читаю книгу", "не открылось: %s (%s)" % (bk["title"], str(e)[:50])); self.st["book"] = None
                self.st["book_cands"] = [c for c in self.st.get("book_cands", []) if "gutenberg" not in c["url"]]        # Gutenberg лежит — берём Викитеку
                self.st["gb_down"] = time.time(); return -0.1, None
            if len(txt) < 500: self.st["book"] = None; return -0.1, None
            open(path, "w", encoding="utf-8").write(txt)
        txt = open(path, encoding="utf-8").read(); off = bk.get("off", 0); end = min(len(txt), off + 24000)
        if end < len(txt): end = max(off + 4000, txt.rfind("\n", off, end))
        chunk = txt[off:end]; bk["off"] = end; part = int(off / 24000) + 1
        self.action("читаю книгу", "«%s», часть %d (%d%%)" % (bk["title"], part, 100 * end // len(txt)))
        if end >= len(txt):
            self.st.setdefault("book_done", []).append(bk["url"]); del self.st["book_done"][:-300]; self.st["book"] = None
        else: self.st["book"] = bk
        sents = G.split_sents(chunk) or [chunk]; hold = sents[-3:] if len(sents) > 12 else []
        self.add_text(" ".join(sents[:-3] if hold else sents), "book:" + bk["title"], hold=0.05)
        if not hold: return 0.0, None
        ids = [self._build({"t": " ".join(hold)})]
        with self.mlock: before = self.net.nll_of(ids)
        feat = np.array(bk["feat"]) if bk.get("feat") else None
        def fn():
            with self.mlock: after = self.net.nll_of(ids)
            self.val.append({"t": " ".join(hold), "src": "book:" + bk["title"], "w": 1.0, "ids": ids[0]})
            r = max(-1.0, min(1.0, (before - after) / before * 6)); self.credit("book", feat, r); return r
        return None, (fn, 80)

    # ================= сон =================
    def sleep(self):
        if self.sleeping: return
        self.sleeping = True; self.wake_req = False; t0 = time.time(); self.emit({"type": "sleep", "on": True})
        G.say("💤 засыпаю: закрепляю выученное, расту, вижу сны…")
        v0 = self.val_now() or self.eval_val(); s0 = self.st["steps"]
        try:
            for _ in range(100):                      # консолидация: учимся на полной мощности
                if self.wake_req or self.stop: break
                time.sleep(1.0)
                if self.st["steps"] - s0 > 160: break
            vd = self.eval_val()
            if not self.wake_req and not self.stop:
                try: self.sleep_recall()
                except Exception: self.log_err(traceback.format_exc())
            for l in self.maintain(): G.say("  🧬 " + l)
            if not self.wake_req:
                for _ in range(3):
                    d = self.gen("", max_new=80, temp=1.0, top_k=60, stop_nl=False, bos=True).replace("\n", " ").strip()
                    if d: G.say("  REM: «%s»" % d[:200]); self.st["dreams"].append(d[:200]); del self.st["dreams"][:-40]
                self.dream_image()
            self.dream_exam()
            try: self.reflect_on_experience()
            except Exception: self.log_err(traceback.format_exc())
            for k_, n_ in ((10000, "10 тысяч шагов обучения"), (20000, "20 тысяч шагов обучения"), (50000, "50 тысяч шагов обучения")):
                if self.st["steps"] >= k_: self.milestone("steps%d" % k_, "Дошёл до %s" % n_)
            self.write_journal(v0, vd, s0)
        except Exception: self.log_err(traceback.format_exc())
        self.st["sleeps"] += 1; self.awake_t0 = time.time(); self.save()
        self.feeling("проснулся после сна", облегчение=0.3, спокойствие=0.2, ностальгия=0.1); self.sleeping = False; self.emit({"type": "sleep", "on": False}); G.say("проснулся (спал %d с)." % (time.time() - t0))


    # ================= сон: проверка, что он усвоил из слов Теодора, и повторение забытого =================
    def sleep_recall(self, n=12):
        """Во сне: даёт себе начало реплики Теодора и пробует продолжить по памяти; что забыл, то повторяет. Это проверка заучивания, не понимания."""
        its = [it for it in self.items if it["src"] == "chat:теодор" and len(it["t"]) >= 14]
        if not its: return None
        rc = self.st.setdefault("recall", {}); seen = {}
        for it in its[-300:]: seen[it["t"]] = it
        fixed = [k for k in self.st.get("recall_fixed", []) if k in seen]                       # постоянная контрольная выборка: одни и те же фразы в каждом сне — иначе процент скачет от того, что попалось
        rot = sorted((k for k in seen if k not in fixed), key=lambda k: (len(rc.get(k, {}).get("acc", [])) > 0, (rc.get(k, {}).get("acc") or [0])[-1], random.random()))[:max(0, n - len(fixed))]        # плюс сначала не проверенные, потом те, что хуже помнит
        keys = fixed + rot
        res = []; weak = []
        for k in keys:
            if self.wake_req or self.stop: break
            ids = self.tok.encode(k, grow=False)
            if len(ids) < 6: continue
            cut = max(3, len(ids) // 2); true = ids[cut:]
            with self.mlock: nll = self.net.nll_of([[K.Tok.BOS] + ids])
            out = self.net.generate([K.Tok.BOS] + ids[:cut], max_new=len(true), temp=0.05, top_k=1, lock=self.mlock, stop=(K.Tok.EOS,))
            acc = sum(1 for a, b in zip(out, true) if a == b) / len(true)
            e = rc.setdefault(k, {"acc": [], "nll": []}); e["acc"] = (e["acc"] + [round(acc, 2)])[-6:]; e["nll"] = (e["nll"] + [round(nll, 2)])[-6:]
            res.append((k, acc, nll)); 
            if acc < 0.5: weak.append(ids)
        if not res: return None
        now = sum(a for _, a, _ in res) / len(res); prev = self.st.get("recall_last")
        fx = [a for k, a, _ in res if k in fixed]; fnow = sum(fx) / len(fx) if len(fx) >= 4 else None; fprev = self.st.get("recall_fixed_last")
        if "recall_fixed" not in self.st and len(res) >= 8: self.st["recall_fixed"] = [k for k, _, _ in res[:10]]          # первый раунд задаёт контрольные фразы
        for _ in range(3):                                                  # повторение забытого: несколько шагов на тех фразах, которые не вспомнил
            if not weak or self.wake_req or self.stop: break
            L = max(len(x) for x in weak) + 1; ids_b = np.zeros((len(weak), L), np.int64)
            for i, x in enumerate(weak): ids_b[i, :len(x) + 1] = [K.Tok.BOS] + x
            with self.mlock: self.net.step(ids_b[:, :-1], ids_b[:, 1:], (ids_b[:, 1:] > 0).astype(np.float32), lr=LR_FIXED)
        self.st["recall_last"] = round(now, 3); hist = self.st.setdefault("recall_hist", []); hist.append([self.st["steps"], round(now, 3), len(res), None if fnow is None else round(fnow, 3)]); del hist[:-40]
        if fnow is not None: self.st["recall_fixed_last"] = round(fnow, 3)
        msg = "вспоминал %d фраз Теодора: помню %d%%%s%s, повторил забытое (%d)" % (len(res), now * 100, (" (раньше %d%%)" % (prev * 100)) if prev is not None else "", ("; на постоянной выборке %d%%%s" % (fnow * 100, (" (было %d%%)" % (fprev * 100)) if fprev is not None else "")) if fnow is not None else "", len(weak))
        self.event("сон", msg); G.say("  🧠 " + msg)
        if fnow is not None and fprev is not None and fnow > fprev + 0.05: self.feeling("помню больше слов Теодора", гордость=0.4, радость=0.3, облегчение=0.2)
        elif fnow is not None and fprev is not None and fnow < fprev - 0.05: self.feeling("забыл слова Теодора", грусть=0.2, тревога=0.2)
        return now

    def recall_view(self):
        rc = self.st.get("recall", {}); h = self.st.get("recall_hist", [])
        best = sorted(((v["acc"][-1], k) for k, v in rc.items() if v["acc"]), reverse=True)
        return {"last": self.st.get("recall_last"), "fixed": self.st.get("recall_fixed_last"), "hist": h[-8:], "tested": len(rc), "good": [[k[9:60], a] for a, k in best[:3] if a > 0.3], "forgot": [[k[9:60], a] for a, k in best[::-1][:3] if a < 0.5]}

    def dream_exam(self):
        r = self.a_exam()
        if r and r[0] is not None: self.policy.update(ACTIONS.index("exam"), self.state_vec(), r[0], self.mask())

    def imagine(self, prompt=None, temp=0.9, tag="dream"):
        """Сеть сама генерирует картинку токенами (по подписи или без): коды кодовой книги -> пиксели. Возвращает имя файла."""
        if len(self.senses.vi.C) < 12 or "image" not in self.by_cat: return None
        ids = [K.Tok.BOS] + (self.tok.encode(prompt, grow=False) if prompt else []) + [self.tok.id("<img>")]
        ok_ids = {i for i, t in enumerate(self.tok.toks) if (m := re.fullmatch(r"⟦i(\d+)⟧", t)) and int(m.group(1)) < len(self.senses.vi.C)}
        ban = np.array([i for i in range(self.net.V) if i not in ok_ids], dtype=np.int64)          # при рисовании выбирает только среди кодов картинки
        def to_code(i):
            m = re.fullmatch(r"⟦i(\d+)⟧", self.tok.toks[i]) if i < len(self.tok) else None
            return int(m.group(1)) if m and int(m.group(1)) < len(self.senses.vi.C) else None
        Cb = self.senses.vi.C; ck = [tuple(np.round(Cb[i].reshape(-1, 3).mean(0) * 3).astype(int)) for i in range(len(Cb))]          # «цвет» кода: все почти-белые коды считаются одним цветом
        tid2c = {i: int(re.fullmatch(r"⟦i(\d+)⟧", self.tok.toks[i]).group(1)) for i in ok_ids}; out = []; used = collections.Counter(); used_c = collections.Counter()
        while len(out) < 64:                                         # по 8 токенов: код, уже занявший 6+ клеток из 64, пока не берём (иначе один частый код — чёрный — заливает картинку)
            step_ban = np.array(sorted(set(ban.tolist()) | {i for i in ok_ids if used[i] >= 6 or used_c[ck[tid2c[i]]] >= 14}), dtype=np.int64)
            part = self.net.generate(ids + out, max_new=8, temp=temp, top_k=40, lock=self.mlock, ban=step_ban)
            if not part: break
            out += [i for i in part if i in ok_ids]; used.update(i for i in part if i in ok_ids); used_c.update(ck[tid2c[i]] for i in part if i in ok_ids)
            if len(part) < 8: break
        codes = [c for c in (to_code(i) for i in out) if c is not None]
        if len(codes) < 8: return None
        self._last_codes = len(set(codes)); rr = random.Random(len(codes))
        while len(codes) < 64: codes.append(rr.choice(codes))             # недостающее не копируем подряд, а берём из его же кодов
        codes = codes[:64]
        from PIL import Image
        P = self.senses.vi.C[codes].reshape(8, 8, 8, 8, 3).transpose(0, 2, 1, 3, 4).reshape(64, 64, 3)
        fn = "%s_%d.png" % (tag, int(time.time())); path = os.path.join(W.WORLD, "out", fn)
        Image.fromarray((np.clip(P, 0, 1) * 255).astype(np.uint8)).resize((256, 256), Image.NEAREST).save(path)
        return fn

    def dream_image(self):
        fn = self.imagine(None, 0.9, "dream")
        if not fn: return
        self.st["dream_imgs"].append(fn); del self.st["dream_imgs"][:-12]
        self.emit({"type": "dream_img", "file": fn}); G.say("  REM-картинка: world/out/%s" % fn)

    def a_draw(self):
        """Рисует, когда захочет: сам придумывает, что нарисовать (свои желания или недавно прочитанное), и генерирует картинку."""
        pool = [(w, w) for w in self.wishes(2)] + [(t, t) for t in random.sample(self.st["read_titles"][-40:], min(3, len(self.st["read_titles"])))]
        if not pool: return -0.05, None
        key, f = self.choose("draw", pool, k=5)
        fn = self.imagine("Картинка: %s" % key, 0.9, "draw")
        if not fn: self.action("рисую", "«%s» — не получилось (мало картинок в памяти)" % key); return -0.1, None
        self.st["art"].append({"file": fn, "prompt": key}); del self.st["art"][:-24]
        self.action("рисую", "«%s» → world/out/%s" % (key, fn)); self.emit({"type": "draw", "file": fn, "text": key})
        self.last_draw_feat = f
        return 0.05 + min(0.25, self._last_codes / 80.0), None       # главное — реакция Теодора (👍/👎)

    def commons_image(self, q, n=1):
        import urllib.parse as up
        api = "https://commons.wikimedia.org/w/api.php?" + up.urlencode({"action": "query", "generator": "search", "gsrnamespace": 6, "gsrsearch": q,
            "gsrlimit": 8, "prop": "imageinfo", "iiprop": "url|mime", "iiurlwidth": 256, "format": "json"})
        d = G.http_json(api); pages = list((d.get("query") or {}).get("pages", {}).values()); random.shuffle(pages); hits = []
        for pg in pages:
            ii = (pg.get("imageinfo") or [{}])[0]
            if ii.get("mime") in ("image/jpeg", "image/png") and ii.get("thumburl"): hits.append((pg["title"].replace("Файл:", "").replace("File:", ""), ii["thumburl"]))
            if len(hits) >= n: break
        return hits if n > 1 else (hits[0] if hits else None)

    def a_look(self):
        """Смотрит картинки из интернета (Викисклад) по теме, которая ему интересна; учится и на «картинка -> подпись», и на «подпись -> картинка»."""
        pool = [(w, w) for w in self.wishes(2)] + [(t, t) for t in random.sample(self.st["read_titles"][-60:], min(4, len(self.st["read_titles"])))]
        if not pool: return -0.05, None
        key, f = self.choose("look", pool, k=5)
        try: hits = self.commons_image(key, 3) or []
        except Exception: hits = []
        hit = hits[0] if hits else None
        if not hit: self.action("смотрю картинки", "по «%s» ничего не нашлось" % key); self.credit("look", f, -0.1); self.feel("look", key, f, None, -0.3, fail=True); return -0.1, None
        title, url = hit; gd = os.path.join(W.WORLD, "gallery"); os.makedirs(gd, exist_ok=True)
        path = os.path.join(gd, hashlib.md5(url.encode()).hexdigest()[:12] + ".jpg")
        try:
            import urllib.request
            open(path, "wb").write(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Genesis/1.0 (learning; local)"}), timeout=30).read(4_000_000))
        except Exception as e: self.action("смотрю картинки", "не скачалось: %s" % str(e)[:50]); return -0.1, None
        r = self.senses.perceive(path, learn=True); self.net.set_vocab(len(self.tok))
        if not r["ids"]: return -0.1, None
        for t2, u2 in hits[1:]:                                  # смотрит сразу несколько снимков по теме: картинок в опыте мало, это его главный дефицит
            try:
                import urllib.request
                p2 = os.path.join(gd, hashlib.md5(u2.encode()).hexdigest()[:12] + ".jpg")
                open(p2, "wb").write(urllib.request.urlopen(urllib.request.Request(u2, headers={"User-Agent": "Genesis/1.0 (learning; local)"}), timeout=30).read(4_000_000))
                r2 = self.senses.perceive(p2, learn=True); self.net.set_vocab(len(self.tok))
                if r2["ids"]:
                    c2 = "%s. %s" % (re.sub(r"\.\w+$", "", t2)[:80], r2["text"]); self.action("смотрю картинки", "ещё: «%s»" % t2[:60])
                    self.add_text(c2, "image:сеть", w=2.0, mode="raw", hold=0.0, modal_ids=r2["ids"], rev=False); self.add_text("Картинка: " + c2, "image:сеть", w=2.0, mode="raw", hold=0.0, modal_ids=r2["ids"], rev=True)
            except Exception: pass
        cap = "%s. %s" % (re.sub(r"\.\w+$", "", title)[:80], r["text"])
        self.action("смотрю картинки", "«%s» — %s" % (title[:60], r["text"][:120]))
        self.emit({"type": "percept", "file": os.path.basename(path), "kind": "image", "text": cap[:240], "ntok": len(r["ids"]), "img": "gallery/" + os.path.basename(path)})
        self.add_text(cap, "image:сеть", w=2.0, mode="raw", hold=0.0, modal_ids=r["ids"], rev=False)
        self.add_text("Картинка: " + cap, "image:сеть", w=2.0, mode="raw", hold=0.0, modal_ids=r["ids"], rev=True)
        # награда — по ДРУГИМ картинкам (обобщение), а не по той, на которой только что учился (это почти всегда +1)
        others = [it["ids"] for it in random.sample(self.by_cat.get("image", []), min(4, len(self.by_cat.get("image", []))))]
        own = [self._build({"t": cap, "mi": r["ids"]})]; ids = others if len(others) >= 3 else own; k = 4.0 if others and len(others) >= 3 else 1.0
        with self.mlock: before = self.net.nll_of(ids, maxlen=CTX); nll0 = self.net.nll_of(own, maxlen=CTX)
        def fn():
            with self.mlock: after = self.net.nll_of(ids, maxlen=CTX)
            rr = max(-1.0, min(1.0, (before - after) / before * k)); self.credit("look", f, rr); self.feel("look", key, f, nll0, rr); return rr
        return None, (fn, 60)

    def write_journal(self, v0, v1, s0):
        ar = W.arena_load(); n = self.net
        txt = ("Шагов обучения за период: %d. Ошибка на невиданном: %s → %s. Параметров: %.2fM (слоёв %d, словарь %d, слияний %d, кодов картинок %d, звука %d). "
               "Задач решено: %d/%d, XP %d. Прочитано источников: %d." %
               (self.st["steps"] - s0, ("%.3f" % v0) if v0 else "—", ("%.3f" % v1) if v1 else "—", n.n_params() / 1e6, len(n.layers), len(self.tok),
                len(self.tok.merges), len(self.senses.vi.C), len(self.senses.va.C), len(ar["solved"]), len(W.KATA), ar["xp"], len(self.st["read_titles"])))
        own = self.gen("Дневник. Сегодня я", max_new=60, temp=0.8, stop_nl=False).replace("\n", " ")
        e = {"t": time.strftime("%F %H:%M"), "text": txt, "own": "Сегодня я" + own[:160]}
        self.st["journal"].append(e); del self.st["journal"][:-100]
        self.emit({"type": "diary", "text": txt + " Его слова: «%s»" % e["own"]})

    # ================= запуск / статистика =================
    def start(self):
        dt = self.st.get("deaths") or []
        if dt and time.time() - dt[-1][0] < 86400 * 7: self.feeling("проснулся после выключения", облегчение=0.5, ностальгия=0.4, удивление=0.3, радость=0.3); self.event("я", "проснулся после выключения: меня выключали уже %d раз, перерыв %d мин., шагов %d" % (len(dt), (time.time() - dt[-1][0]) / 60, dt[-1][1]))
        for fn in (self._train_loop, self._actor_loop, self._brain_log_loop):
            t = threading.Thread(target=fn, daemon=True); t.start(); self.threads.append(t)

    talking = False; live_ids = []; last_gn = 0.0; last_loss = 0.0; _brain_cache = None

    def brain_snapshot(self, text=None):
        """Состояние «мозга» для окна наблюдения: что горит в каждом слое при чтении текущего куска (или заданного текста)."""
        ids = ([K.Tok.BOS] + self.tok.encode(text, grow=False))[-96:] if text else (self.live_ids[-64:] or [K.Tok.BOS])
        ids = [i for i in ids if i < self.net.V] or [K.Tok.BOS]
        got = self.mlock.acquire(timeout=2.0)
        if not got: return self._brain_cache
        try: r = self.net.probe(ids)
        finally: self.mlock.release()
        n = self.net
        r["tokens"] = [self.tok.toks[i] if i < len(self.tok) else "?" for i in ids]
        r["top"] = [[self.tok.toks[i] if i < len(self.tok) else "?", p] for i, p in r["top"]]
        r["train"] = {"steps": self.st["steps"], "loss": round(self.last_loss, 3), "gn": float("%.3g" % self.last_gn), "tps": int(self.tps), "lr": LR_FIXED,
                      "params": n.n_params(), "sleeping": self.sleeping, "fatigue": round(self.fatigue(), 2), "vocab": len(self.tok), "mine": bool(text) is False}
        la = self.st["action_log"][-1] if self.st.get("action_log") else None
        r["action"] = la; r["bpb"] = (self.st.get("val_big_bpb") or [[0, None]])[-1][1]
        self._brain_cache = r; return r

    def _brain_log_loop(self):
        """Журнал мыслей-активаций в файл brain_log.jsonl: по нему можно разбирать, что происходит внутри, без окна."""
        path = self.p("brain_log.jsonl"); time.sleep(20)
        while not self.stop:
            try:
                r = self.brain_snapshot()
                if r:
                    rec = {"t": time.strftime("%F %T"), "step": r["train"]["steps"], "loss": r["train"]["loss"], "gn": r["train"]["gn"], "tps": r["train"]["tps"], "bpb": r["bpb"],
                           "sleep": r["train"]["sleeping"], "action": r["action"], "entropy": round(r["entropy"], 2),
                           "top": [[t, round(p, 3)] for t, p in r["top"][:3]], "reading": "".join(r["tokens"][-40:]),
                           "layers": [{"active": round(l["active"], 3), "att_ent": [round(e, 2) for e in l["att_ent"]], "attn": round(l["attn_out"], 2), "mlp": round(l["mlp_out"], 2), "resid": round(l["resid"], 1)} for l in r["layers"]]}
                    if os.path.exists(path) and os.path.getsize(path) > 6 << 20: os.replace(path, path + ".old")
                    open(path, "a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception: self.log_err(traceback.format_exc())
            for _ in range(40):
                if self.stop: break
                time.sleep(0.5)

    def shutdown(self):
        self.stop = True; time.sleep(1.0)
        self.st.setdefault("deaths", []).append([time.time(), self.st["steps"]]); del self.st["deaths"][:-50]
        try: self.save(); self.log_err("shutdown: save ok, steps %d" % self.st["steps"]); self.snapshot(); self.log_err("shutdown: snapshot ok")
        except Exception: self.log_err("shutdown FAILED " + traceback.format_exc())

    def stats(self):
        n = self.net; ar = W.arena_load(); p = self.policy.probs(self.state_vec(), self.mask(), 0.8)
        return {"params": n.n_params(), "layers": len(n.layers), "mlp": [n.P[k + ".W1"].d.shape[1] for k in n.layers], "d": n.cfg["d"], "vocab": len(self.tok),
                "merges": len(self.tok.merges), "img_codes": len(self.senses.vi.C), "aud_codes": len(self.senses.va.C), "steps": self.st["steps"],
                "loss": round(self.st["loss_ema"], 3) if self.st["loss_ema"] else None, "val": self.val_now(), "val_hist": [v for _, v in self.st["val_hist"][-60:]],
                "tps": int(self.tps), "lr": round(self.st["lr"], 5), "corpus_tokens": self.corpus_tokens(), "items": len(self.items),
                "policy": {a: round(float(x), 2) for a, x in zip(ACTIONS, p)}, "last_action": self.last_action[0] if self.last_action else None,
                "solved": len(ar["solved"]), "kata": len(W.KATA), "xp": ar["xp"], "fatigue": round(self.fatigue(), 2), "events": self.st["events"][-6:],
                "actions": self.st["action_log"][-8:], "dream_imgs": self.st["dream_imgs"][-4:], "art": self.st["art"][-8:], "sleeping": self.sleeping}
