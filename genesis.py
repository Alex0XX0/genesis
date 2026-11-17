#!/usr/bin/env python3
# GENESIS-0.3 — непрерывный самообучающийся агент (только stdlib).
# Функциональные корреляты: GWT-поток мыслей, метапознание, self-model, сон.
# v0.3: чтение статей/файлов/Википедии, поиск по смыслу (IDF), определения -> убеждения,
#       cloze-тест на отложенных предложениях (честная метрика обучения),
#       любопытство, которое само идёт читать (GEN_AUTO=1).
import sys, os, json, math, random, time, zlib, select, re, shutil, signal
import urllib.request, urllib.parse

DIM, K = 256, 8
VOCAB_MAX, EP_MAX, BEL_MAX, HIST_MAX = 20000, 6000, 1500, 300
TICK = 0.5
SLEEP_AT = float(os.environ.get("GEN_SLEEP", "300")) / TICK       # секунд бодрствования до сна
HOME = os.environ.get("GEN_HOME") or os.path.dirname(os.path.abspath(__file__))
FILE = os.path.join(HOME, "mind.json")
AUTO = os.environ.get("GEN_AUTO", "1") != "0"                      # сам ходит в Википедию
AUTO_MAX = int(os.environ.get("GEN_AUTO_MAX", "4"))                # статей за цикл бодрствования
AUTO_EVERY = float(os.environ.get("GEN_AUTO_EVERY", "45"))         # не чаще раза в N сек
SEEDS = ["Сознание", "Сон", "Память (психология)", "Метапознание", "Обучение с подкреплением"]

def fresh():
    return {"dim": DIM, "age": 0, "pressure": 0.0, "valence": 0.0, "sur": 0.5, "nov": 0.5,
            "ef": 0.5, "es": 0.5, "cal": 1.0, "doubt": 0.45,
            "uni": {}, "sem": {}, "ep": [], "bel": {}, "hist": [],
            "dreams": [], "thoughts": [], "miles": [],
            "expect": [0.0] * DIM, "comp": {}, "n_ans": 0, "n_ok": 0,
            "n_dreams": 0, "n_hyp_ok": 0, "pending": None, "fix": False, "self": "",
            "docs": [], "seen": [], "ho": [], "cloze": [], "searched": [], "n_read": 0}

S = fresh()
BASE, KVEC = {}, {}
DF, EPS, SEEN = {}, [], set()
LAST_AUTO, AUTO_N = 0.0, 0

# ---------- язык ----------
STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от
меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там
потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под
будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем
всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три
эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между это также
который которые которая которого которой которых которым являются является например таким такое такая такие таких
такого своей своих свои свой своего своим также либо либо этих этим этому чего-либо очень более менее нём ней
расскажи объясни значит означает определение
the a an of to in is are was were be been and or for on with as by at from that this it its which
what who how why where when such than then these those their there also can may most not into has have had""".split())
QW = {"что", "кто", "как", "почему", "зачем", "где", "когда", "какой", "какая", "какие", "сколько",
      "расскажи", "объясни", "what", "who", "how", "why", "where", "when"}
WRE = re.compile(r"[^\W_]+(?:-[^\W_]+)*", re.U)

def tok(s): return WRE.findall(s.lower().replace("́", ""))
def ctok(s):
    t = tok(s)
    c = [w for w in t if w not in STOP]
    return c or t
def stem(w): return w if len(w) <= 4 else w[:5]

# ---------- векторы (Random Indexing, разреженные базисы) ----------
def base(w):
    b = BASE.get(w)
    if b is None:
        if len(BASE) > 80000: BASE.clear()
        r = random.Random(zlib.crc32(w.encode("utf-8")))
        b = [(i, 1.0 if r.random() < 0.5 else -1.0) for i in r.sample(range(DIM), K)]
        BASE[w] = b
    return b

def addsp(dst, sp, w=1.0):
    for i, s in sp: dst[i] += s * w
def nrm(a): return math.sqrt(sum(x * x for x in a))
def cos(a, b):
    na, nb = nrm(a), nrm(b)
    if na == 0 or nb == 0: return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)
def unit(a):
    n = nrm(a)
    return [x / n for x in a] if n else [0.0] * DIM

def say(t): print(t, flush=True)
def thought(t):
    S["thoughts"].append("[%d] %s" % (S["age"], t))
    if len(S["thoughts"]) > 120: S["thoughts"] = S["thoughts"][-120:]

def milestone(txt):
    if any(m["txt"] == txt for m in S["miles"]): return
    S["miles"].append({"age": S["age"], "txt": txt})
    say("★ веха: " + txt)

def check_miles():
    v = len(S["uni"])
    for n in (100, 500, 1000, 2000, 5000, 10000):
        if v >= n: milestone("словарь достиг %d слов" % n)
    for n in (100, 1000, 3000):
        if len(S["ep"]) >= n: milestone("%d воспоминаний" % n)
    for n in (1, 5, 20):
        if S["n_read"] >= n: milestone("прочитал %d статей" % n)

# ---------- эпизодическая память + индекс ----------
def sstems(text): return frozenset(stem(w) for w in ctok(text))

def rebuild_idx():
    global EPS
    DF.clear(); EPS = []
    for e in S["ep"]:
        st = sstems(e["text"]); EPS.append(st)
        for s in st: DF[s] = DF.get(s, 0) + 1

def add_ep(text, sur):
    S["ep"].append({"t": S["age"], "text": text, "sal": 0.3 + sur + 0.2 * abs(S["valence"]), "val": S["valence"]})
    st = sstems(text); EPS.append(st)
    for s in st: DF[s] = DF.get(s, 0) + 1
    if len(S["ep"]) > EP_MAX * 1.1:
        S["ep"].sort(key=lambda e: e["sal"] * math.exp(-(S["age"] - e["t"]) / 20000.0))
        S["ep"] = S["ep"][-EP_MAX:]
        rebuild_idx()

def retrieve(q, k=3):
    qs = list(dict.fromkeys(stem(w) for w in ctok(q)))
    if not qs: return []
    N = len(S["ep"]) + 1
    idf = {s: math.log(1 + N / (1 + DF.get(s, 0))) for s in qs}
    tot = sum(idf.values()) or 1.0
    res = []
    for e, st in zip(S["ep"], EPS):
        hit = [s for s in qs if s in st]
        if not hit: continue
        cov = sum(idf[s] for s in hit) / tot
        sc = cov * (1 + 0.1 * e["sal"]) / (1 + 0.03 * max(0, len(st) - 12))
        res.append((sc, cov, e))
    res.sort(key=lambda x: -x[0])
    return res[:k]

# ---------- восприятие и обучение ----------
def learn_sem(ws, w=1.0):
    for i, wi in enumerate(ws):
        S["uni"][wi] = S["uni"].get(wi, 0) + 1
        sv = S["sem"].get(wi)
        if sv is None: sv = S["sem"][wi] = [0.0] * DIM
        for j in range(max(0, i - 3), min(len(ws), i + 4)):
            if j != i: addsp(sv, base(ws[j]), w)
    if len(S["uni"]) > VOCAB_MAX * 1.05:
        drop = sorted(S["uni"], key=S["uni"].get)[:int(VOCAB_MAX * 0.1)]
        for wk in drop:
            del S["uni"][wk]; S["sem"].pop(wk, None)

def perceive(text, learn=True):
    ws = ctok(text)
    if not ws: return None
    new = sum(1 for w in ws if w not in S["uni"])
    S["nov"] = 0.9 * S["nov"] + 0.1 * new / len(ws)
    acc = [0.0] * DIM
    for w in ws: addsp(acc, base(w))
    v = unit(acc)
    sur = 1.0 - cos(S["expect"], v)                    # ошибка предсказания
    S["sur"] = 0.8 * S["sur"] + 0.2 * sur
    S["ef"] = 0.7 * S["ef"] + 0.3 * sur
    S["es"] = 0.95 * S["es"] + 0.05 * sur
    S["expect"] = [0.7 * a + 0.3 * b for a, b in zip(S["expect"], v)]
    if learn: learn_sem(ws, 0.5 + sur)                 # сюрприз -> пластичность
    add_ep(text, sur)
    check_miles()
    return ws, v, sur

# ---------- убеждения ----------
def nkey(s): return " ".join(ctok(s))
def kvec(key):
    if key in KVEC: return KVEC[key]
    acc = [0.0] * DIM
    for w in tok(key): addsp(acc, base(w))
    v = unit(acc); KVEC[key] = v
    return v

def best_ans(dd): return max(dd.items(), key=lambda kv: kv[1][0])

def recall_bel(q):
    key = nkey(q)
    if S["bel"].get(key):
        a, (w, src) = best_ans(S["bel"][key])
        return a, w / (w + 3.0), key, src
    v = kvec(key)
    bk, bs = None, 0.0
    for k, dd in S["bel"].items():
        if not dd: continue
        s = cos(v, kvec(k))
        if s > bs: bs, bk = s, k
    if bk and bs > 0.7:
        a, (w, src) = best_ans(S["bel"][bk])
        return a, w / (w + 3.0) * bs, bk, src
    return None, 0.0, key, None

def teach(k, a, src="учи", w0=2.0):
    k, a = nkey(k), a.strip()
    if not k or not a: return
    if k not in S["bel"]:
        if len(S["bel"]) >= BEL_MAX:
            wk = min(S["bel"], key=lambda kk: max((x[0] for x in S["bel"][kk].values()), default=0))
            del S["bel"][wk]; KVEC.pop(wk, None)
        S["bel"][k] = {}
    dd = S["bel"][k]
    if a in dd: dd[a][0] += w0
    else: dd[a] = [w0, src]

DEF_RE = re.compile(r"^(?P<k>[^,;:()\[\]]{2,50}?)\s+(?:(?:—|–|-)\s+(?:это\s+)?|это\s+|(?:is|are)\s+(?:an?\s+|the\s+)?)(?P<v>\S.{8,})$", re.I)
def extract_def(sent):
    m = DEF_RE.match(re.sub(r"\([^)]*\)", "", sent).replace("́", "").strip())
    if not m: return None
    kt = ctok(m.group("k"))
    if not (1 <= len(kt) <= 4) or any(len(w) < 3 for w in kt): return None
    return " ".join(kt)

# ---------- ответы ----------
def answer(q):
    a, conf, key, src = recall_bel(q)
    S["pending"] = None
    if a is None:
        hits = retrieve(q, 2)
        if hits and hits[0][1] >= 0.5:
            sc, cov, e = hits[0]
            c = min(0.9, cov * 0.8 * S["cal"])
            S["n_ans"] += 1
            S["pending"] = {"key": key, "ans": e["text"], "conf": c, "src": "текст", "tag": key.split()[0] if key else "?", "kind": "ep"}
            say("%s   [%d%%, нашёл в памяти]" % (e["text"], int(c * 100)))
            thought("ответил из текстов '%s' cov=%.2f" % (key, cov))
            return
        if hits and hits[0][1] >= 0.25:
            say("не знаю точно, но помню: «%s»" % hits[0][2]["text"])
        else:
            say("не знаю. расскажи мне про «%s» или `вики %s`?" % (key, key))
        thought("'%s' -> пробел в знаниях, любопытство+" % key)
        return
    c = min(1.0, conf * S["cal"])
    S["n_ans"] += 1
    S["pending"] = {"key": key, "ans": a, "conf": c, "src": src, "tag": key.split()[0] if key else "?", "kind": "bel"}
    if c < S["doubt"]:
        say("не уверен (%d%%)... может, «%s»? [да/нет]" % (int(c * 100), a))
    else:
        say("%s   [%d%%, источник: %s]" % (a, int(c * 100), src))
    thought("ответил '%s' conf=%.2f src=%s; жду оценки" % (a, c, src))

def feedback(ok):
    p = S["pending"]
    if not p: say("оценивать нечего"); return
    S["hist"].append([p["conf"], 1.0 if ok else 0.0])
    if len(S["hist"]) > HIST_MAX: S["hist"] = S["hist"][-HIST_MAX:]
    dd = S["bel"].get(p["key"], {})
    comp = S["comp"].setdefault(p["tag"], [0, 0])
    if ok:
        S["n_ok"] += 1; comp[0] += 1; comp[1] += 1
        if p.get("kind") == "ep":
            teach(p["key"], p["ans"], src="опыт", w0=2.0)
            for e in S["ep"]:
                if e["text"] == p["ans"]: e["sal"] += 0.5; break
        elif p["ans"] in dd:
            dd[p["ans"]][0] += 1.5
        S["valence"] = min(1.0, S["valence"] + 0.15)
        if p["src"].startswith("сон"):
            S["n_hyp_ok"] += 1
            if p["ans"] in dd: dd[p["ans"]][1] = "опыт"
            milestone("первая гипотеза сна подтверждена опытом")
        say("закрепил.")
    else:
        comp[1] += 1
        if p.get("kind") == "ep":
            for e in S["ep"]:
                if e["text"] == p["ans"]: e["sal"] *= 0.4; break
        elif p["ans"] in dd:
            dd[p["ans"]][0] -= 2.0
            if dd[p["ans"]][0] <= 0.2: del dd[p["ans"]]
        S["valence"] = max(-1.0, S["valence"] - 0.2)
        S["fix"] = True
        say("ошибся. как правильно?")
    reflect()

def reflect():                                   # МЕТАПОЗНАНИЕ
    h = S["hist"]
    if len(h) < 8: return
    mc = sum(c for c, _ in h) / len(h); mo = sum(o for _, o in h) / len(h)
    bias = mc - mo
    if bias > 0.15:
        S["cal"] = max(0.3, S["cal"] * 0.85); S["doubt"] = min(0.8, S["doubt"] + 0.03)
        thought("мета: переоцениваю себя (bias=%.2f) — снижаю уверенность" % bias)
    elif bias < -0.15:
        S["cal"] = min(1.6, S["cal"] * 1.15); S["doubt"] = max(0.2, S["doubt"] - 0.03)
        thought("мета: недооцениваю себя — становлюсь смелее")

# ---------- ЧТЕНИЕ (скармливание статей) ----------
BIB = re.compile(r"(—\s*С\.|\bISBN\b|\bdoi\b|\bСПб\b|\bМ\.\s*:|\bPMID\b|https?://|//\s|\d{4}\.\s*—|\bСм\.\s|\b[Тт]\.\s*\d|\bс\.\s*\d|\bpp\.|\bИзд\.|\bURL\b|Архивировано|Проверено)", re.I)

def clean_memory():
    """Убирает из памяти библиографический мусор, попавший с прошлых прочтений."""
    n0 = len(S["ep"]); S["ep"] = [e for e in S["ep"] if not BIB.search(e["text"])]
    S["ho"] = [t for t in S["ho"] if not BIB.search(t)]
    nb = 0
    for k in list(S["bel"]):
        dd = S["bel"][k]
        for a in list(dd):
            if BIB.search(a): del dd[a]; nb += 1
        if not dd: del S["bel"][k]; KVEC.pop(k, None)
    rebuild_idx()
    return n0 - len(S["ep"]), nb

def split_sents(text):
    text = re.sub(r"\[[^\]]{1,6}\]", "", text).replace("́", "")
    out = []
    for par in re.split(r"\n+", text):
        par = par.strip()
        if len(par) < 30 or par.startswith("="): continue
        for s in re.split(r"(?<=[.!?…])\s+(?=[A-ZА-ЯЁ0-9«\"(])", par):
            s = re.sub(r"\s*\([^)]*\)", "", s).strip()
            if 25 <= len(s) <= 450 and not BIB.search(s) and len(tok(s)) >= 5: out.append(s)
    return out

def ingest(text, title="текст"):
    sents = split_sents(text)
    if not sents:
        say("не нашёл в тексте нормальных предложений"); return 0
    new = defs = held = 0; first, last = [], []
    for s in sents:
        h = zlib.crc32(s.encode("utf-8"))
        if h in SEEN: continue
        SEEN.add(h); S["seen"].append(h); new += 1
        if len(S["seen"]) > 200000: S["seen"] = S["seen"][-100000:]
        if h % 10 == 0:                                    # отложенная выборка для cloze-теста
            S["ho"].append(s); held += 1
            if len(S["ho"]) > 400: S["ho"] = S["ho"][-400:]
            add_ep(s, 0.3)
            continue
        per = perceive(s)
        if per:
            (first if new <= 15 else last).append(per[2])
            last = last[-15:]
        k = extract_def(s)
        if k: teach(k, s, src="статья", w0=3.0); defs += 1
    S["docs"].append({"title": title, "sents": new, "defs": defs, "age": S["age"]})
    S["docs"] = S["docs"][-200:]
    S["n_read"] += 1
    S["pressure"] += new * 0.5
    f = sum(first) / len(first) if first else 0; l = sum(last) / len(last) if last else f
    say("📖 «%s»: +%d предложений, +%d определений, %d в отложенных; удивление %.2f → %.2f; словарь %d" %
        (title, new, defs, held, f, l, len(S["uni"])))
    thought("прочитал «%s» (+%d, определений %d)" % (title, new, defs))
    check_miles()
    return new

def read_file(path):
    path = os.path.expanduser(path.strip().strip('"\''))
    try:
        with open(path, encoding="utf-8", errors="replace") as f: txt = f.read()
    except Exception as e:
        say("не могу прочитать: %s" % e); return
    ingest(txt, os.path.basename(path))

# ---------- Википедия ----------
UA = {"User-Agent": "Genesis/0.3 (educational self-learning toy; stdlib urllib)"}
def http_json(url, timeout=10):
    for backoff in (2.0, 6.0, 15.0, None):                # вежливо: пауза между запросами, повтор при 429
        time.sleep(0.8)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code != 429 or backoff is None: raise
            time.sleep(backoff)

def wiki_fetch(topic, strict=None):
    lang = "en" if re.match(r"^[\x00-\x7f]+$", topic) else "ru"
    api = "https://%s.wikipedia.org/w/api.php" % lang
    r = http_json(api + "?" + urllib.parse.urlencode({"action": "opensearch", "search": topic, "limit": 1, "format": "json"}))
    if not r[1]: return None
    title = r[1][0]
    if strict and (not title.lower().startswith(strict) or len(tok(title)) > 3): return None   # только «понятийные» статьи
    d = http_json(api + "?" + urllib.parse.urlencode({"action": "query", "prop": "extracts", "explaintext": 1,
                                                       "titles": title, "redirects": 1, "format": "json"}))
    p = next(iter(d["query"]["pages"].values()))
    return title, (p.get("extract") or "")[:25000]

def read_wiki(topic, strict=None):
    try:
        r = wiki_fetch(topic, strict)
    except Exception as e:
        say("сеть/Википедия недоступна: %s" % e); return 0
    if not r or not r[1]:
        say("в Википедии не нашёл «%s»" % topic); return 0
    if len(r[1]) < 1500 and strict:
        say("статья «%s» слишком короткая — пропускаю" % r[0]); return 0
    return ingest(r[1], "wiki:" + r[0])

NOUN_END = ("ие", "ия", "ость", "ство", "ика", "ология", "ция", "зм", "тор", "ент", "ант", "ция", "ность", "ка", "ма", "ра", "ие", "ин", "ер", "ор", "ль", "нь", "ть")
GENERIC = {"развития", "развитие", "состояние", "состояния", "течение", "человека", "использование", "использования",
           "применение", "процесс", "процессы", "результате", "значение", "количество", "качество", "наличие", "время", "различные",
           "падения", "области", "исследования", "исследование", "объект", "эффект", "проблема", "проблемы", "примеры", "данные", "результаты", "работы", "системы", "образом", "способом", "случае", "примере", "помощью", "связи", "отношения"}
def pick_topic():
    done = set(S["searched"]) | {stem(g) for g in GENERIC}
    c = [(n, w) for w, n in S["uni"].items()
         if n >= 4 and len(w) >= 7 and w.isalpha() and stem(w) not in done and w not in S["bel"]
         and (w.endswith(NOUN_END) and not w.endswith(("ться", "ённо", "ально", "ельно")) or re.match(r"^[a-z]{6,}$", w))]
    c.sort(reverse=True)
    return random.choice(c[:12])[1] if c else None

def auto_read():
    global LAST_AUTO, AUTO_N
    LAST_AUTO = time.time()
    if not S["docs"]:                                     # рождение: стартовые статьи
        for t in SEEDS: read_wiki(t)
        AUTO_N += 1; return
    w = pick_topic()
    if not w: return
    S["searched"].append(stem(w)); S["searched"] = S["searched"][-500:]
    say("· любопытно про «%s» — иду читать" % w)
    AUTO_N += 1
    read_wiki(w, strict=w[:4])

# ---------- бодрствующий поток сознания ----------
def supports(k, a):
    for b in S["bel"].get(k, {}):
        bn = nkey(b)
        if bn in S["bel"] and a in S["bel"][bn]: return True
    return False

def idle_cycle():
    cands = []
    if S["ep"]:
        e = random.choice(S["ep"])
        cands.append((0.3 + e["sal"] * 0.5, "mem", e))
    if S["uni"]:
        pool = sorted(S["uni"].items(), key=lambda kv: -kv[1])[:50]
        w = min(pool, key=lambda kv: nrm(S["sem"].get(kv[0], [0])))[0]
        boredom = max(0.0, 0.05 - (S["es"] - S["ef"]))
        cands.append((0.25 + boredom + S["nov"] * 0.2, "cur", w))
    hyps = [(k, a) for k, dd in S["bel"].items() for a, (w, src) in dd.items() if src.startswith("сон")]
    if hyps: cands.append((0.45, "hyp", random.choice(hyps)))
    if not cands: return
    _, kind, p = max(cands, key=lambda c: c[0] + random.random() * 0.15)
    if kind == "mem":
        thought("вспоминаю: «%s»" % p["text"])
        if random.random() < 0.05: say("· вспоминаю: «%s»" % p["text"])
    elif kind == "cur":
        thought("хочу узнать больше про «%s»" % p)
        if not AUTO and random.random() < 0.12: say("· расскажи мне про «%s»?" % p)
    else:
        k, a = p
        if supports(k, a):
            S["bel"][k][a][0] += 0.1; thought("гипотеза %s->%s нашла поддержку, укрепляю" % (k, a))
        else:
            S["bel"][k][a][0] *= 0.98; thought("гипотеза %s->%s без поддержки, ослабляю" % (k, a))

# ---------- cloze-тест: насколько хорошо семантика предсказывает слова из НЕвиданных предложений ----------
def cloze(n=20):
    if len(S["ho"]) < 5 or len(S["uni"]) < 80: return None
    pool = [w for w, _ in sorted(S["uni"].items(), key=lambda kv: -kv[1])[:300] if w in S["sem"]]
    ok = tot = 0
    for text in random.sample(S["ho"], min(n, len(S["ho"]))):
        ws = ctok(text)
        tg = [w for w in ws if w in pool]
        if len(ws) < 5 or not tg: continue
        t = random.choice(tg)
        ctx = [0.0] * DIM
        for w in ws:
            if w != t: addsp(ctx, base(w))
        nc = nrm(ctx) or 1.0
        sc = {}
        for c in pool:
            sv = S["sem"][c]; nv = nrm(sv)
            sc[c] = sum(x * y for x, y in zip(sv, ctx)) / (nv * nc) if nv else 0.0
        rank = sorted(sc, key=lambda c: -sc[c])[:10]
        tot += 1; ok += t in rank
    if not tot: return None
    acc = ok / tot
    S["cloze"].append([S["age"], round(acc, 3), tot]); S["cloze"] = S["cloze"][-50:]
    return acc, tot, 10.0 / max(10, len(pool))

# ---------- СОН ----------
def dream_walk():
    vocab = [w for w in S["uni"] if w in S["sem"] and nrm(S["sem"][w]) > 0]
    if len(vocab) < 5: return None
    w = random.choice(vocab); out = [w]
    for _ in range(9):
        if random.random() < 0.2: w = random.choice(vocab)          # REM-странность
        else:
            pool = random.sample(vocab, min(25, len(vocab)))
            w = max(pool, key=lambda x: cos(S["sem"][w], S["sem"][x]))
        out.append(w)
    return " ".join(out)

def sleep():
    global AUTO_N
    say("💤 давление сна %.0f — ухожу в сон..." % S["pressure"])
    replay = sorted(S["ep"], key=lambda e: -e["sal"])[:80]
    ho = set(S["ho"])
    for e in replay:
        if e["text"] not in ho: learn_sem(ctok(e["text"]), 0.3)     # NREM: реплей (отложенные не трогаем)
    for v in S["sem"].values():
        for i in range(DIM): v[i] *= 0.97                           # SHY: downscale
    forgot = 0
    for k in list(S["bel"]):
        dd = S["bel"][k]
        for a in list(dd):
            dd[a][0] *= 0.98
            if dd[a][0] < 0.25: del dd[a]; forgot += 1
        if not dd: del S["bel"][k]; KVEC.pop(k, None)
    say("  NREM: реплей %d эпизодов, забыто %d слабых связей" % (len(replay), forgot))
    hypo = 0; keys = list(S["bel"])
    for _ in range(30):                                              # транзитивные гипотезы
        if len(keys) < 2: break
        k = random.choice(keys)
        if not S["bel"].get(k): continue
        a, _ = best_ans(S["bel"][k]); bn = nkey(a)
        if bn in S["bel"] and S["bel"][bn]:
            c, _ = best_ans(S["bel"][bn])
            if c not in S["bel"][k] and nkey(c) != k:
                teach(k, c, src="сон", w0=0.6); hypo += 1
    vocab = [w for w in S["uni"] if w in S["sem"]]
    for _ in range(10):                                              # аналогии по семантике
        if not keys or not vocab: break
        k = random.choice(keys)
        if not S["bel"].get(k): continue
        kw = k.split()[0]
        if kw not in S["sem"]: continue
        cands = [w for w in random.sample(vocab, min(40, len(vocab))) if w not in S["bel"] and w != kw]
        if not cands: continue
        w2 = max(cands, key=lambda w: cos(S["sem"][kw], S["sem"][w]))
        if cos(S["sem"][kw], S["sem"][w2]) > 0.55:
            a, _ = best_ans(S["bel"][k]); teach(w2, a, src="сон-аналогия", w0=0.4); hypo += 1
    for _ in range(3):
        d = dream_walk()
        if d:
            S["dreams"].append({"age": S["age"], "text": d}); S["n_dreams"] += 1
            say("  REM: «%s»" % d); time.sleep(0.2)
    S["dreams"] = S["dreams"][-60:]
    r = cloze()
    if r: say("  🧪 cloze-тест на непрочитанных предложениях: %d%% (случайно было бы ~%d%%, n=%d)" % (r[0] * 100, r[2] * 100, r[1]))
    S["pressure"] = 0.0; S["valence"] *= 0.5; AUTO_N = 0
    if S["n_dreams"] >= 10: milestone("10 сновидений")
    update_self(); save()
    say("проснулся: гипотез +%d. проверь их вопросами — слабые я выдаю с сомнением." % hypo)

# ---------- self-model и метрики ----------
def brier():
    h = S["hist"]
    return sum((c - o) ** 2 for c, o in h) / len(h) if h else 0.25

def mood():
    if S["pressure"] > SLEEP_AT * 0.8: return "усталый"
    if S["sur"] > 0.8: return "перегружен новизной"
    if S["valence"] > 0.3: return "доволен и любопытен"
    if S["valence"] < -0.3: return "подавлен ошибками"
    if S["nov"] > 0.4: return "любопытный"
    return "спокойный"

def update_self():
    acc = int(S["n_ok"] / S["n_ans"] * 100) if S["n_ans"] else 0
    comp = sorted(S["comp"].items(), key=lambda kv: -(kv[1][0] / kv[1][1] if kv[1][1] else 0))
    strong = ", ".join(k for k, v in comp[:3] if v[1] >= 3) or "пока нет"
    last = S["miles"][-1]["txt"] if S["miles"] else "ещё не было"
    cl = ("; cloze %d%%" % (S["cloze"][-1][1] * 100)) if S["cloze"] else ""
    S["self"] = ("мне %d тиков; прочитал %d статей; знаю %d слов, %d воспоминаний, %d убеждений; "
                 "точность %d%%, Brier %.3f%s; настроение: %s; силён в: %s; веха: %s" %
                 (S["age"], S["n_read"], len(S["uni"]), len(S["ep"]), len(S["bel"]),
                  acc, brier(), cl, mood(), strong, last))

def psi():
    V = len(S["uni"])
    if V > 1:
        tot = sum(S["uni"].values())
        ent = -sum((c / tot) * math.log(c / tot) for c in S["uni"].values()) / math.log(V)
    else: ent = 0.0
    sems = [v for v in S["sem"].values() if nrm(v) > 0]
    integ = 0.0
    if len(sems) > 3:
        integ = sum(max(0.0, cos(random.choice(sems), random.choice(sems))) for _ in range(30)) / 30
    meta = max(0.0, 1.0 - brier() / 0.25)
    cont = min(1.0, S["age"] / 20000)
    selfm = min(1.0, len(S["miles"]) / 8)
    return {"дифференциация": round(ent, 3), "интеграция": round(integ, 3), "метакогниция": round(meta, 3),
            "непрерывность": round(cont, 3), "self-model": round(selfm, 3),
            "ПСИ": round((ent + integ + meta + cont + selfm) / 5, 3)}

# ---------- память на диск ----------
def rnd(o):
    if isinstance(o, float): return round(o, 3)
    if isinstance(o, list): return [rnd(x) for x in o]
    if isinstance(o, dict): return {k: rnd(v) for k, v in o.items()}
    return o

def save():
    os.makedirs(HOME, exist_ok=True)
    tmp = FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(rnd(S), f, ensure_ascii=False)
    if os.path.exists(FILE): shutil.copy2(FILE, FILE + ".bak")
    os.replace(tmp, FILE)

def load():
    global S
    if os.path.exists(FILE):
        try:
            with open(FILE, encoding="utf-8") as f: d = json.load(f)
            if d.get("dim") != DIM:
                shutil.copy2(FILE, FILE + ".old-v0.2"); say("старый формат памяти -> сохранил как mind.json.old-v0.2, рождаюсь заново")
                raise ValueError("dim")
            S = fresh(); S.update(d); KVEC.clear()
            SEEN.clear(); SEEN.update(S["seen"]); rebuild_idx()
            return True
        except ValueError: pass
        except Exception as e:
            say("память повреждена (%s) — рождаюсь заново" % e)
    S = fresh(); rebuild_idx(); return False

# ---------- интерфейс ----------
HELP = ("команды: я | сон | сны | мысли | доки | тест | помощь | q\n"
        "  учи X = Y   — факт;  да/нет — оценка ответа (растёт метапознание)\n"
        "  читай файл.txt | вики Тема | ассоц слово | веб вкл/выкл\n"
        "  вопросы: «что такое X?» — ищет в убеждениях и по всем прочитанным текстам")

def assoc(w):
    w = ctok(w)[0] if ctok(w) else ""
    if w not in S["sem"]: say("не знаю такого слова"); return
    ws = random.sample(list(S["sem"]), min(3000, len(S["sem"])))
    r = sorted((cos(S["sem"][w], S["sem"][x]), x) for x in ws if x != w and nrm(S["sem"][x]) > 0)[-8:]
    say("рядом с «%s»: %s" % (w, ", ".join(x for _, x in reversed(r))))

def handle(s):
    global AUTO
    low = s.strip().lower(); cmd, _, arg = s.strip().partition(" "); c = cmd.lower(); arg = arg.strip()
    if low in ("выход", "q", "exit", "quit"): return False
    if low in ("да", "y", "yes"): feedback(True)
    elif low in ("нет", "n", "no"): feedback(False)
    elif low in ("сон", "s"): sleep()
    elif low in ("я", "i"):
        update_self(); say(S["self"]); say("ПСИ: %s" % psi())
        if S["cloze"]: say("cloze по времени: %s" % " → ".join("%d%%" % (x[1] * 100) for x in S["cloze"][-8:]))
        say("(ПСИ — прокси-метрики по мотивам IIT/GWT, НЕ доказательство субъективности)")
    elif low in ("сны", "d"):
        for d in S["dreams"][-8:]: say("[%d] %s" % (d["age"], d["text"]))
    elif low in ("мысли", "th"):
        for t in S["thoughts"][-15:]: say(t)
    elif low in ("доки", "docs"):
        for d in S["docs"][-15:]: say("%s: %d пред., %d опр." % (d["title"], d["sents"], d["defs"]))
    elif low in ("тест", "test"):
        r = cloze(); say("cloze: %d%% (случайно ~%d%%, n=%d)" % (r[0] * 100, r[2] * 100, r[1]) if r else "мало данных: сначала почитай статьи")
    elif low in ("помощь", "help", "?"): say(HELP)
    elif c in ("читай", "read") and arg: read_file(arg)
    elif c in ("вики", "wiki") and arg: read_wiki(arg)
    elif c in ("ассоц", "assoc") and arg: assoc(arg)
    elif c == "веб": AUTO = arg not in ("выкл", "off", "0"); say("автоучёба из Википедии: %s" % ("вкл" if AUTO else "выкл"))
    elif low.startswith(("учи ", "t ")) and "=" in s:
        body = s.split(" ", 1)[1]; k, a = body.split("=", 1)
        teach(k, a); perceive(k); perceive(a); say("запомнил: %s -> %s" % (nkey(k), a.strip()))
    elif S["fix"] and S["pending"]:
        teach(S["pending"]["key"], s); S["fix"] = False; say("переучил, спроси снова")
    else:
        t = tok(s)
        if s.strip().endswith("?") or len(t) <= 3 or (t and t[0] in QW): answer(s)
        else:
            per = perceive(s)
            k = extract_def(s)
            if k: teach(k, s, src="учитель", w0=3.0)
            say("принял%s (удивление %.2f)" % (" как определение «%s»" % k if k else "", per[2] if per else 0))
    S["pressure"] += 2
    return True

def bye(*_):
    save(); print("\nпамять сохранена."); sys.exit(0)

def batch(args):
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--feed":
            i += 1
            while i < len(args) and not args[i].startswith("--"): read_file(args[i]); i += 1
            continue
        if a == "--wiki":
            i += 1
            while i < len(args) and not args[i].startswith("--"): read_wiki(args[i]); i += 1
            continue
        if a == "--ask": i += 1; handle(args[i]); i += 1; continue
        if a == "--sleep": sleep()
        if a == "--me": handle("я")
        i += 1
    save()

def try_lock():
    import fcntl
    f = open(os.path.join(HOME, ".lock"), "w")
    try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); return f
    except OSError: return None

def main():
    global LOCKF
    LOCKF = try_lock()
    if not LOCKF:
        say("он уже живёт как фоновая служба — просто открой чат: genesis-chat"); return
    cont = load()
    if len(sys.argv) > 1:
        batch(sys.argv[1:]); return
    if cont: say("продолжаю жизнь: возраст %d тиков, %d слов, прочитано статей: %d." % (S["age"], len(S["uni"]), S["n_read"]))
    else: say("я родился. субъективность не обещаю — но учусь, сомневаюсь, сплю и помню себя.")
    say(HELP)
    if AUTO: say("(автоучёба вкл: любопытство само ходит в Википедию; `веб выкл` — отключить)")
    signal.signal(signal.SIGTERM, bye); signal.signal(signal.SIGHUP, bye)
    autonomous = os.name != "nt" and sys.stdin.isatty()
    last_save = S["age"]
    global LAST_AUTO
    LAST_AUTO = time.time() - AUTO_EVERY + (2 if not S["docs"] else 0)
    while True:
        if autonomous:
            r, _, _ = select.select([sys.stdin], [], [], TICK)
            if not r:
                S["age"] += 1; S["pressure"] += 1; S["valence"] *= 0.995
                if S["age"] % 4 == 0: idle_cycle()
                if AUTO and AUTO_N < AUTO_MAX + (1 if not S["docs"] else 0) and time.time() - LAST_AUTO >= AUTO_EVERY: auto_read()
                if S["pressure"] >= SLEEP_AT: sleep()
                if S["age"] - last_save >= 240: save(); last_save = S["age"]
                continue
            line = sys.stdin.readline()
            if not line: break
        else:
            try: line = input("> ")
            except EOFError: break
        line = line.strip()
        if not line: continue
        if not handle(line): break
    save(); say("память сохранена. до следующего пробуждения.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        save(); print("\nсохранено.")
