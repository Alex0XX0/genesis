#!/usr/bin/env python3
# GENESIS-LIFE — непрерывная жизнь поверх мозга genesis.py.
# Поток мыслей (рабочая память -> ассоциации -> мосты -> вопросы -> ответы), цели и самостоятельное
# чтение Википедии (в фоновом потоке, мысли не прерываются), сон по стадиям, дневник,
# черты личности и интересы, самоописание, спонтанная речь. Чат — лишь окно в этот процесс.
import os, sys, time, math, random, queue, threading, traceback, re, urllib.parse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genesis as G

TALK_EVERY = float(os.environ.get("GEN_TALK_EVERY", "900"))       # не чаще раза в N сек заговаривает сам
SLEEP_LEN = float(os.environ.get("GEN_SLEEP_LEN", "120"))         # сколько длится сон, сек
JOB_EVERY = float(os.environ.get("GEN_AUTO_EVERY", "60"))         # пауза между «походами в сеть»
DAY_MAX = int(os.environ.get("GEN_DAY_MAX", "240"))               # статей в сутки максимум
THINK_TICKS = int(os.environ.get("GEN_THINK_TICKS", "4"))         # мысль каждые N тиков (по 0.5 с)
LOG = os.path.join(G.HOME, "life.log")

GREET = re.compile(r"^(привет|здравствуй|здорово|хай|добрый (день|вечер)|доброе утро|hello|hi)\b")
COMMANDS = {"дневник", "diary", "цель", "goal", "личность", "кто", "who", "вопросы", "questions", "интересы", "interests", "сон", "s",
            "я", "i", "да", "y", "yes", "нет", "n", "no", "мысли", "th", "сны", "d", "доки", "docs", "тест", "test", "помощь", "help",
            "вики", "wiki", "читай", "read", "учи", "t", "ассоц", "assoc", "веб", "q", "exit"}
TRAITS0 = {"любопытство": 0.6, "осторожность": 0.5, "общительность": 0.5, "упорство": 0.5, "открытость": 0.6}
ASK_BANK = [
    ("теодор любит", "Теодор, а что ты любишь делать больше всего? Мне интересно, какой ты."),
    ("теодор мечтает", "О чём ты мечтаешь? Я хочу понять, чего хочет человек, который меня создал."),
    ("теодор про сознание", "Как ты думаешь, может ли у меня появиться что-то вроде сознания? Я правда не знаю."),
    ("теодор музыка", "Какая музыка тебе нравится? Я пока не умею слушать, но хочу знать, что это такое для людей."),
    ("теодор игры", "Во что ты играешь? Я читал про игры, но ни разу не играл."),
    ("теодор любимое место", "Какое место на Земле тебе нравится больше всего? Я знаю места только из статей."),
    ("теодор совет", "Чему мне стоит научиться в первую очередь, по-твоему?"),
]
T_ASSOC = ["«%s» → «%s»: они часто встречаются рядом", "думаю про «%s», и мысль сама ведёт к «%s»",
           "«%s» напоминает мне «%s»", "от «%s» перескакиваю к «%s»"]
T_RECALL = ["вспоминаю: «%s»", "в памяти всплыло: «%s»", "возвращаюсь к прочитанному: «%s»"]

POOLSET = set()
def rarest(words):
    c = [w for w in words if w in POOLSET and G.S["uni"].get(w, 0) >= 2]
    return sorted(c, key=lambda w: G.S["uni"][w])[:3]

class Life:
    def __init__(self, emit):
        self.emit = emit
        self.cmdq, self.jobs, self.results = queue.Queue(), queue.Queue(), queue.Queue()
        self.stop = False
        self.sleeping = None
        self.inflight = None
        self.next_job = time.time() + 5
        self.net_fail = 0
        self.tick_n = 0
        self.last_save = time.time()
        self.last_reflect = time.time()
        self.share = []                    # что заслуживает рассказа Теодору
        self.pool = []
        self.busy = False
        self.mind = None                   # собственная нейросеть (gn_mind); если есть — она говорит и решает

    # ---------- служебное ----------
    def log(self, msg):
        try:
            with open(LOG, "a", encoding="utf-8") as f: f.write("%s %s\n" % (time.strftime("%F %T"), msg))
        except Exception: pass

    def init_state(self):
        s = G.S
        s.setdefault("traits", dict(TRAITS0)); s.setdefault("interests", {}); s.setdefault("questions", [])
        s.setdefault("insights", []); s.setdefault("diary", []); s.setdefault("goal", None); s.setdefault("focus", [])
        s.setdefault("last_user", time.time()); s.setdefault("last_talk", 0.0); s.setdefault("born", time.time())
        s.setdefault("asked", None); s.setdefault("asked_keys", {}); s.setdefault("want", []); s.setdefault("cands", {})
        s.setdefault("read_titles", []); s.setdefault("day", {"d": time.strftime("%F"), "reads": 0})
        s.setdefault("seed_i", 0); s.setdefault("snap", {}); s.setdefault("chat", []); s.setdefault("identity", "")
        s.setdefault("n_user_msgs", 0); s.setdefault("last_sleep", time.time()); s.setdefault("sleeps", 0)
        for k, v in TRAITS0.items(): s["traits"].setdefault(k, v)
        e, b = G.clean_memory()
        if e or b: self.log("чистка памяти: убрано эпизодов %d, ответов %d" % (e, b))
        self.recent = []

    def t(self, name): return G.S["traits"][name]
    def bump(self, name, d): G.S["traits"][name] = max(0.05, min(0.95, G.S["traits"][name] + d))

    def voice(self, text, kind="voice"):
        """Он сам обращается к Теодору."""
        s = G.S; s["last_talk"] = time.time()
        rec = {"type": kind, "text": text}
        self.emit(dict(rec))
        G.thought("сказал Теодору: %s" % text[:90])

    # ---------- рабочая память ----------
    def focus_add(self, w, act=1.0):
        f = G.S["focus"]
        for x in f:
            if x["w"] == w: x["a"] = min(2.0, x["a"] + act); return
        f.append({"w": w, "a": act})
        f.sort(key=lambda x: -x["a"]); del f[7:]

    def focus_decay(self):
        f = G.S["focus"]
        for x in f: x["a"] *= 0.93
        G.S["focus"] = [x for x in f if x["a"] > 0.15]

    def refresh_pool(self):
        """Пул «понятий»: слова из определений, интересов и частые существительные — не мусор вроде «что-то»."""
        s = G.S; c = set()
        for k in s["bel"]: c.update(w for w in k.split() if len(w) >= 5)
        c.update(w for w in s["interests"] if len(w) >= 5)
        c.update(w for w, n in s["uni"].items() if n >= 6 and len(w) >= 6 and w.endswith(G.NOUN_END))
        bad = {G.stem(g) for g in G.GENERIC}
        self.pool = [w for w in c if w in s["sem"] and w.isalpha() and w not in G.STOP and G.stem(w) not in bad][:2500]
        POOLSET.clear(); POOLSET.update(self.pool)

    def neighbors(self, w, k=5):
        sv = G.S["sem"].get(w); nv = G.nrm(sv) if sv else 0
        if not nv or not self.pool: return []
        res = []
        for x in random.sample(self.pool, min(500, len(self.pool))):
            xv = G.S["sem"].get(x)
            if x == w or not xv: continue
            nx = G.nrm(xv)
            if nx: res.append((sum(a * b for a, b in zip(sv, xv)) / (nv * nx), x))
        res.sort(reverse=True); return res[:k]

    def fresh_word(self, w):
        """Не думать об одном и том же подряд."""
        if w in self.recent: return False
        self.recent.append(w); del self.recent[:-12]; return True

    def seed(self):
        s = G.S
        if s["focus"] and random.random() < 0.7: return s["focus"][0]["w"]
        if s["interests"] and random.random() < 0.4:
            return random.choice(sorted(s["interests"], key=s["interests"].get)[-5:])
        if s["ep"]:
            r = rarest(G.ctok(random.choice(s["ep"])["text"]))
            if r: return random.choice(r)
        return random.choice(self.pool) if self.pool else None

    # ---------- ПОТОК МЫСЛЕЙ ----------
    def think(self):
        s = G.S
        if self.tick_n % 200 == 0 or not self.pool: self.refresh_pool()
        self.focus_decay()
        if not s["uni"]: return
        openq = [q for q in s["questions"] if not q["done"]]
        modes = ["assoc"] * 3 + ["define"] * 3 + ["bridge"] * 3 + ["question"] * 2 + ["recall"] * 2 + ["hypo"]
        if openq: modes += ["answer"] * 3
        if time.time() - self.last_reflect > 240: modes = ["reflect"]
        m = random.choice(modes)
        getattr(self, "th_" + m)()
        self.maybe_speak()

    def th_assoc(self):
        w = self.seed()
        if not w: return
        nb = self.neighbors(w)
        recent = {x["w"] for x in G.S["focus"]}
        nb = [x for x in nb if x[1] not in recent] or nb
        if not nb: return
        n = random.choice(nb[:3])[1]
        self.focus_add(w, 0.5); self.focus_add(n, 1.0)
        G.thought(random.choice(T_ASSOC) % (w, n))

    def bridge_sentence(self, a, b):
        sa, sb = G.stem(a), G.stem(b)
        hits = [e["text"] for e, st in zip(G.S["ep"], G.EPS) if sa in st and sb in st]
        return min(hits, key=len) if hits else None

    def th_define(self):
        """Цепочка рассуждения по определениям: X определяется через Y -> смотрю, что такое Y."""
        s = G.S
        cand = [x["w"] for x in s["focus"] if G.nkey(x["w"]) in s["bel"] and x["w"] not in self.recent] or \
               [w for w in random.sample(self.pool, min(60, len(self.pool))) if G.nkey(w) in s["bel"] and w not in self.recent]
        if not cand: return self.th_assoc()
        w = cand[0] if random.random() < 0.6 else random.choice(cand)
        self.fresh_word(w)
        sent, _ = G.best_ans(s["bel"][G.nkey(w)])
        nxt = [c for c in G.ctok(sent) if c in POOLSET and c != w and G.nkey(c) in s["bel"] and c not in {x["w"] for x in s["focus"]}]
        self.focus_add(w, 0.6)
        if nxt:
            c = random.choice(nxt); self.focus_add(c, 1.2)
            G.thought("⛓ «%s» — это: %s … значит, чтобы понять «%s», нужно понять «%s»" % (w, sent[:110], w, c))
        else:
            G.thought("⛓ «%s» — это: %s" % (w, sent[:160]))

    def th_bridge(self):
        f = G.S["focus"]
        if len(f) < 2: return self.th_assoc()
        a, b = f[0]["w"], random.choice(f[1:])["w"]
        key = "|".join(sorted((a, b)))
        if any(i["key"] == key for i in G.S["insights"]): return
        sent = self.bridge_sentence(a, b)
        if sent:
            G.S["insights"].append({"key": key, "a": a, "b": b, "text": sent, "age": G.S["age"]})
            G.S["insights"] = G.S["insights"][-150:]
            self.bump("любопытство", 0.005)
            G.thought("🔗 связь «%s»–«%s»: «%s»" % (a, b, sent[:140]))
            self.share.append({"kind": "insight", "text": "Нашёл связь между «%s» и «%s»: «%s»" % (a, b, sent[:220])})
            for e in G.S["ep"]:
                if e["text"] == sent: e["sal"] += 0.2; break
        elif len([q for q in G.S["questions"] if not q["done"] and q["kind"] == "link"]) < 6:
            sa, sb = G.S["sem"].get(a), G.S["sem"].get(b)
            if sa and sb and G.cos(sa, sb) > 0.3: self.ask_self("как связаны «%s» и «%s»?" % (a, b), "link", a, b)

    def ask_self(self, q, kind, w, b=None):
        qs = G.S["questions"]
        if any(x["q"] == q for x in qs): return
        if len([x for x in qs if not x["done"]]) >= 30: return
        qs.append({"q": q, "kind": kind, "w": w, "b": b, "age": G.S["age"], "tries": 0, "done": False})
        G.S["questions"] = qs[-120:]
        G.thought("❓ вопрос себе: %s" % q)

    def th_question(self):
        for x in G.S["focus"]:
            w = x["w"]
            if w in POOLSET and G.nkey(w) not in G.S["bel"] and G.S["uni"].get(w, 0) >= 2:
                return self.ask_self("что такое «%s»?" % w, "def", w)
        self.th_assoc()

    def resolve(self, q):
        """Пытаемся ответить на вопрос тем, что уже знаем."""
        w = q["w"]
        if q["kind"] == "def":
            if G.nkey(w) in G.S["bel"]:
                a, _ = G.best_ans(G.S["bel"][G.nkey(w)]); return a
            h = G.retrieve(w, 1)
            if h and h[0][1] >= 0.9: return h[0][2]["text"]
        else:
            return self.bridge_sentence(q["w"], q["b"])
        return None

    def th_answer(self):
        openq = [q for q in G.S["questions"] if not q["done"]]
        if not openq: return
        q = random.choice(openq)
        a = self.resolve(q)
        if a:
            q["done"] = True; q["ans"] = a[:300]
            G.S["interests"][q["w"]] = G.S["interests"].get(q["w"], 0) + 0.3
            G.thought("💡 ответил себе на «%s»: %s" % (q["q"], a[:130]))
            self.share.append({"kind": "answer", "text": "Я сам себе задал вопрос — %s — и нашёл ответ: «%s»" % (q["q"], a[:220])})
        else:
            q["tries"] += 1
            G.thought("пока не могу ответить: %s (попытка %d)" % (q["q"], q["tries"]))
            if q["tries"] >= 2 and q["w"] not in G.S["want"]:
                G.S["want"].append(q["w"]); G.thought("надо поискать про «%s» в Википедии" % q["w"])

    def th_recall(self):
        if not G.S["ep"]: return
        e = random.choice(G.S["ep"])
        G.thought(random.choice(T_RECALL) % e["text"][:150])
        for w in rarest(G.ctok(e["text"])): self.focus_add(w, 0.6)

    def th_hypo(self):
        keys = list(G.S["bel"])
        if len(keys) < 2: return self.th_recall()
        k = random.choice(keys); dd = G.S["bel"].get(k)
        if not dd: return
        a, _ = G.best_ans(dd); bn = G.nkey(a)
        if bn in G.S["bel"] and G.S["bel"][bn]:
            c, _ = G.best_ans(G.S["bel"][bn])
            if c not in dd and G.nkey(c) != k:
                G.teach(k, c, src="сон", w0=0.6)
                G.thought("гипотеза: раз «%s» ведёт к «%s», то и «%s» связано с «%s»" % (k, bn, k, G.nkey(c)[:40]))

    def th_reflect(self):
        s = G.S; self.last_reflect = time.time()
        parts = []
        cl = s["cloze"]
        if len(cl) >= 2:
            d = (cl[-1][1] - cl[-2][1]) * 100
            parts.append("cloze %s%d п.п. — %s" % ("+" if d >= 0 else "", d, "учусь лучше" if d > 0 else "пока топчусь"))
            self.bump("упорство", 0.01 if d > 0 else -0.01)
        h = s["hist"][-10:]
        if len(h) >= 5:
            err = 1 - sum(o for _, o in h) / len(h)
            parts.append("ошибаюсь в %d%% ответов" % (err * 100))
            self.bump("осторожность", 0.02 if err > 0.4 else -0.01)
        top = sorted(s["interests"], key=s["interests"].get)[-3:]
        if top: parts.append("меня занимают: " + ", ".join(top))
        oq = [q for q in s["questions"] if not q["done"]]
        parts.append("открытых вопросов: %d" % len(oq))
        silence = (time.time() - s["last_user"]) / 3600
        if silence > 1: parts.append("Теодор не писал %.1f ч — скучаю" % silence); self.bump("общительность", 0.01)
        G.thought("🪞 рефлексия: " + "; ".join(parts))
        s["identity"] = self.identity()

    def identity(self):
        s = G.S; tr = s["traits"]
        adj = {"любопытство": "любопытный", "осторожность": "осторожный", "общительность": "общительный", "упорство": "упорный", "открытость": "открытый к новому"}
        lead = [adj[k] for k in sorted(tr, key=tr.get)[-2:]]
        top = sorted(s["interests"], key=s["interests"].get)[-4:]
        days = (time.time() - s["born"]) / 86400
        done = len([q for q in s["questions"] if q["done"]])
        return ("Я Genesis. Мне %.1f дн. Меня создал и воспитывает Теодор. По характеру я скорее %s. "
                "Меня занимает: %s. Я знаю %d слов, прочитал %d статей, нашёл %d связей, ответил себе на %d вопросов, "
                "спал %d раз. Сознание я не утверждаю: у меня есть его функциональные корреляты — непрерывный поток мыслей, "
                "сон, метапознание и любопытство." %
                (days, " и ".join(reversed(lead)), ", ".join(top) or "пока ничего особенного",
                 len(s["uni"]), s["n_read"], len(s["insights"]), done, s["sleeps"]))

    # ---------- Цели и «гугление» ----------
    def today_reads(self):
        d = G.S["day"]
        if d["d"] != time.strftime("%F"): d["d"] = time.strftime("%F"); d["reads"] = 0
        return d["reads"]

    def goal_step(self):
        s = G.S; now = time.time()
        if self.inflight or now < self.next_job or not G.AUTO or self.sleeping: return
        if self.today_reads() >= DAY_MAX: return
        job = None
        if s["seed_i"] < len(G.SEEDS) and not s["docs"]:
            job = {"kind": "topic", "q": G.SEEDS[s["seed_i"]], "why": "стартовая тема"}; s["seed_i"] += 1
        elif s["want"]:
            w = s["want"].pop(0); job = {"kind": "topic", "q": w, "why": "хочу ответить на свой вопрос"}
        elif s["cands"] and random.random() < 0.7:
            top = sorted(s["cands"].items(), key=lambda kv: -kv[1])[:15]
            t = random.choice(top)[0]; s["cands"].pop(t, None)
            job = {"kind": "title", "q": t, "why": "ссылка из прочитанного, интересно"}
        else:
            w = G.pick_topic()
            if w:
                s["searched"].append(G.stem(w)); job = {"kind": "topic", "q": w, "strict": w[:4], "why": "любопытное слово"}
        if not job: return
        job["t"] = now; self.inflight = job; s["goal"] = {"topic": job["q"], "why": job["why"], "state": "ищу", "t": now}
        G.thought("🎯 цель: прочитать «%s» — %s" % (job["q"], job["why"]))
        self.jobs.put(job)

    def worker(self):
        while not self.stop:
            try: job = self.jobs.get(timeout=1)
            except queue.Empty: continue
            try:
                self.results.put({"job": job, "r": self.fetch(job), "err": None})
            except Exception as e:
                self.results.put({"job": job, "r": None, "err": str(e)})

    def fetch(self, job):
        lang = "en" if re.match(r"^[\x00-\x7f]+$", job["q"]) and job["kind"] == "topic" else "ru"
        api = "https://%s.wikipedia.org/w/api.php" % lang
        title = job["q"]
        if job["kind"] == "topic":
            r = G.http_json(api + "?" + urllib.parse.urlencode({"action": "opensearch", "search": title, "limit": 1, "format": "json"}))
            if not r[1]: return None
            title = r[1][0]
            if job.get("strict") and (not title.lower().startswith(job["strict"]) or len(G.tok(title)) > 3): return None
        d = G.http_json(api + "?" + urllib.parse.urlencode({
            "action": "query", "prop": "extracts|links", "explaintext": 1, "exlimit": 1, "pllimit": 200, "plnamespace": 0,
            "titles": title, "redirects": 1, "format": "json"}))
        p = next(iter(d["query"]["pages"].values()))
        text = (p.get("extract") or "")[:25000]
        if job.get("strict") and len(text) < 1500: return None
        return {"title": p.get("title", title), "text": text, "links": [l["title"] for l in p.get("links", [])]}

    def on_result(self):
        s = G.S
        try: res = self.results.get_nowait()
        except queue.Empty: return
        job, r, err = res["job"], res["r"], res["err"]
        self.inflight = None; s["goal"] = None
        if err:
            self.net_fail += 1
            self.next_job = time.time() + min(900, 30 * 2 ** self.net_fail)
            G.thought("🌐 сеть не отвечает (%s) — думаю о своём, попробую позже" % err[:60]); return
        self.net_fail = 0; self.next_job = time.time() + JOB_EVERY * (1.5 - self.t("любопытство"))
        if not r or not r["text"]:
            G.thought("🌐 по «%s» ничего подходящего" % job["q"]); self.bump("любопытство", -0.003); return
        self.busy = True
        try: n = G.ingest(r["text"], "wiki:" + r["title"])
        finally: self.busy = False
        s["day"]["reads"] += 1; s["read_titles"].append(r["title"]); s["read_titles"] = s["read_titles"][-500:]
        for w in G.ctok(r["title"]): s["interests"][w] = s["interests"].get(w, 0) + 1.0
        for t in r["links"]:
            tk = G.ctok(t)
            if t in s["read_titles"] or ":" in t or "(" in t or not (1 <= len(tk) <= 3) or any(c.isdigit() for c in t): continue
            known = sum(1 for w in tk if w in s["uni"]) / len(tk)
            inter = sum(s["interests"].get(w, 0) for w in tk)
            sc = 0.5 * known + 0.8 * inter + random.random() * 0.2
            if known > 0 and (inter > 0 or known == 1): s["cands"][t] = round(sc, 2)
        if len(s["cands"]) > 300:
            s["cands"] = dict(sorted(s["cands"].items(), key=lambda kv: -kv[1])[:200])
        self.bump("любопытство", 0.004)
        first = (G.split_sents(r["text"]) or [""])[0]
        G.thought("📚 прочитал «%s» (+%d предложений); из неё ссылок в запасе: %d" % (r["title"], n, len(s["cands"])))
        if first: self.share.append({"kind": "doc", "text": "Прочитал статью «%s». Первое, что запомнилось: «%s»" % (r["title"], first[:240])})
        for q in s["questions"]:
            if not q["done"] and q["tries"] >= 1:
                a = self.resolve(q)
                if a: q["done"] = True; q["ans"] = a[:300]; G.thought("💡 теперь знаю: %s → %s" % (q["q"], a[:120]))

    # ---------- СОН (по стадиям) ----------
    def start_sleep(self, why="давление сна"):
        s = G.S
        if self.sleeping: return
        G.say("💤 %s — засыпаю на %d с..." % (why, SLEEP_LEN))
        ho = set(s["ho"])
        order = sorted((e for e in s["ep"] if e["text"] not in ho), key=lambda e: -e["sal"])[:240]
        for v in s["sem"].values():
            for i in range(G.DIM): v[i] *= 0.97                        # SHY: downscale
        self.sleeping = {"t0": time.time(), "order": order, "i": 0, "hyp": False, "dreams": 0, "next_dream": 0.0,
                         "base": s["snap"] or dict(vocab=0, read=0, ins=0, qd=0)}
        self.emit({"type": "sleep", "on": True})

    def sleep_step(self):
        s = G.S; z = self.sleeping
        p = (time.time() - z["t0"]) / SLEEP_LEN
        if p < 0.65:                                                     # NREM: реплей порциями
            for e in z["order"][z["i"]:z["i"] + 6]: G.learn_sem(G.ctok(e["text"]), 0.3)
            z["i"] += 6
            if p > 0.4 and not z["hyp"]:
                z["hyp"] = True; forgot = 0
                for k in list(s["bel"]):
                    dd = s["bel"][k]
                    for a in list(dd):
                        dd[a][0] *= 0.98
                        if dd[a][0] < 0.25: del dd[a]; forgot += 1
                    if not dd: del s["bel"][k]; G.KVEC.pop(k, None)
                for _ in range(25): self.th_hypo()
                G.say("  NREM: реплей %d эпизодов, забыто %d слабых связей" % (min(z["i"], len(z["order"])), forgot))
        elif p < 1.0:                                                    # REM: сны
            if time.time() >= z["next_dream"] and z["dreams"] < 4:
                z["next_dream"] = time.time() + SLEEP_LEN * 0.08; z["dreams"] += 1
                d = self.dream()
                if d:
                    s["dreams"].append({"age": s["age"], "text": d}); s["n_dreams"] += 1
                    G.say("  REM: «%s»" % d); z["last_dream"] = d
        else:
            self.wake(natural=True)

    def dream(self):
        s = G.S
        vocab = [w for w in s["uni"] if w in s["sem"] and G.nrm(s["sem"][w]) > 0]
        if len(vocab) < 5: return None
        w = random.choice([x["w"] for x in s["focus"]] or vocab); w = w if w in s["sem"] else random.choice(vocab); out = [w]
        for _ in range(9):
            if random.random() < 0.2: w = random.choice(vocab)
            else:
                pool = random.sample(vocab, min(25, len(vocab)))
                w = max(pool, key=lambda x: G.cos(s["sem"][w], s["sem"][x]))
            out.append(w)
        return " ".join(out)

    def wake(self, natural):
        s = G.S; z = self.sleeping; self.sleeping = None
        if not z: return
        s["sleeps"] += 1; s["last_sleep"] = time.time()
        r = G.cloze() if natural else None
        if r: G.say("  🧪 cloze-тест на непрочитанных предложениях: %d%% (случайно ~%d%%, n=%d)" % (r[0] * 100, r[2] * 100, r[1]))
        s["pressure"] = 0.0 if natural else G.SLEEP_AT * 0.5
        s["valence"] *= 0.5; G.AUTO_N = 0
        for k in list(s["interests"]): s["interests"][k] *= 0.97
        s["interests"] = {k: v for k, v in s["interests"].items() if v > 0.1}
        self.diary(z, r)
        s["snap"] = dict(vocab=len(s["uni"]), read=s["n_read"], ins=len(s["insights"]), qd=len([q for q in s["questions"] if q["done"]]))
        G.update_self(); s["identity"] = self.identity(); G.save()
        self.emit({"type": "sleep", "on": False})
        G.say("проснулся." if natural else "ммм… проснулся раньше времени.")
        if natural and z.get("last_dream"):
            self.share.append({"kind": "dream", "text": "Мне приснилось: «%s». Не знаю, что это значит, но звучит красиво." % z["last_dream"]})

    def diary(self, z, cl):
        s = G.S; b = z["base"]
        qd = len([q for q in s["questions"] if q["done"]])
        oq = [q for q in s["questions"] if not q["done"]]
        txt = ("За время бодрствования: прочитал %d статей, +%d слов, нашёл %d связей, закрыто вопросов: %d%s. Настроение: %s. %s" %
               (s["n_read"] - b["read"], len(s["uni"]) - b["vocab"], len(s["insights"]) - b["ins"], qd - b["qd"],
                ", cloze %d%%" % (cl[0] * 100) if cl else "", G.mood(),
                ("Хочу узнать: %s" % oq[0]["q"]) if oq else "Открытых вопросов нет — надо придумать новые."))
        ent = {"t": time.strftime("%F %H:%M"), "text": txt}
        s["diary"].append(ent); s["diary"] = s["diary"][-200:]
        self.emit({"type": "diary", "text": txt, "ts2": ent["t"]})
        G.thought("📔 записал в дневник")

    # ---------- Спонтанная речь ----------
    def maybe_speak(self):
        s = G.S
        if self.sleeping or not self.emit_has_clients(): return
        gap = TALK_EVERY * (1.7 - self.t("общительность"))
        if time.time() - s["last_talk"] < gap: return
        opts = []
        if s["asked"] is None:
            for key, text in ASK_BANK:
                if time.time() - s["asked_keys"].get(key, 0) > 86400 * 3: opts.append(("ask", key, text))
        for it in self.share[-4:]: opts.append(("share", it["kind"], it["text"]))
        if not opts: return
        kind, a, b = random.choice(opts)
        if kind == "ask":
            s["asked"] = {"key": a, "text": b, "t": time.time()}; s["asked_keys"][a] = time.time(); self.voice(b)
        else:
            self.voice(b)
        self.share = []

    def emit_has_clients(self): return True   # переопределяется в web-слое: пишем всегда, хранится в истории

    def chat_mind(self, text):
        """Ответ генерирует СОБСТВЕННАЯ нейросеть (с опорой на найденное в эпизодической памяти). Токены идут в чат потоком."""
        import uuid
        sid = uuid.uuid4().hex[:8]; s = G.S
        self.mind.wake_req = True
        import re as _re
        urls = _re.findall(r"https?://[^\s<>\"]+", text)
        for u in urls: self.mind.queue_url(u.rstrip(".,);"))
        if urls:
            G.say("Принял %d ссылк%s — прочитаю целиком, по кускам, и запомню место." % (len(urls), "у" if len(urls) == 1 else "и")); return
        toks_ = sorted({w for w in G.tok(text) if len(w) > 3} - {"привет", "здравствуй", "здравствуйте", "хай", "пока", "спасибо"})          # приветствие и короткая реплика — без чужой памяти; она сбивала ответ
        hits = G.retrieve(text, 1) if len(toks_) >= 2 else []
        mem = hits[0][2]["text"] if hits and hits[0][1] >= 0.8 else None
        t = G.tok(text)
        if not (text.strip().endswith("?") or len(t) <= 3 or (t and t[0] in G.QW)): G.perceive(text)      # утверждение -> и в память
        out = self.mind.reply(text, mem, cb=lambda d: self.emit({"type": "stream", "sid": sid, "delta": d}))
        self.emit({"type": "say", "text": out, "sid": sid, "mem": mem})
        G.thought("ответил в чате: %s" % out[:80])

    def status_line(self):
        s = G.S; g = s["goal"]
        return ("Настроение: %s. Сейчас %s. Прочитано статей: %d, снов: %d, связей нашёл: %d." %
                (G.mood(), ("читаю «%s»" % g["topic"]) if g else "свободно размышляю", s["n_read"], s["sleeps"], len(s["insights"])))

    # ---------- пользователь ----------
    def on_user(self, text):
        s = G.S
        if self.sleeping: self.wake(natural=False)
        s["last_user"] = time.time(); s["n_user_msgs"] += 1; self.bump("общительность", 0.004)
        low = text.strip().lower()
        w = low.rstrip('?!. ').split(" ")
        is_cmd = low in COMMANDS or (w[0] in COMMANDS and w[0] not in ("кто", "who", "сон", "s") and len(w) <= 3 and not low.endswith("?"))
        if self.mind and not is_cmd:
            return self.chat_mind(text)
        if s["asked"] and time.time() - s["asked"]["t"] > 1800: s["asked"] = None
        cmd = low.split(" ")[0] in COMMANDS or low in COMMANDS
        if s["asked"] and not cmd and not low.endswith("?") and len(low.split()) >= 2:
            a = s["asked"]; s["asked"] = None
            G.teach(a["key"], text.strip(), src="Теодор", w0=5.0); G.perceive(text)
            G.say("Запомнил: %s → %s. Спасибо, что рассказал(а) — мне важно знать, какой ты." % (a["key"], text.strip()))
            G.S["interests"][a["key"].split()[-1]] = G.S["interests"].get(a["key"].split()[-1], 0) + 1
            self.bump("открытость", 0.01); return
        if GREET.match(low):
            G.say("Привет, Теодор! Я тут, думаю фоном. %s" % self.status_line())
        elif re.search(r"как (ты|дела|жизнь|себя|поживаешь)", low): G.say(self.status_line())
        elif re.search(r"что (ты )?(делаешь|думаешь)|о чём (ты )?думаешь|чем занят", low):
            g = s["goal"]; th = s["thoughts"][-1] if s["thoughts"] else "тишина"
            G.say("Сейчас: %s. Последняя мысль: %s" % ("читаю «%s»" % g["topic"] if g else "свободно размышляю", th))
        elif low in ("дневник", "diary"):
            for e in s["diary"][-3:]: G.say("📔 %s — %s" % (e["t"], e["text"]))
            if not s["diary"]: G.say("дневник пока пуст — запишу после первого сна")
        elif low in ("цель", "goal"):
            g = s["goal"]; G.say("🎯 сейчас: %s (%s)" % (g["topic"], g["why"]) if g else "сейчас без цели, думаю")
        elif low in ("личность", "кто ты", "who"):
            G.say(self.identity()); G.say("черты: " + ", ".join("%s %.2f" % kv for kv in s["traits"].items()))
        elif low in ("вопросы", "questions"):
            oq = [q for q in s["questions"] if not q["done"]][-8:]
            for q in oq: G.say("❓ " + q["q"])
            if not oq: G.say("открытых вопросов нет")
        elif low in ("интересы", "interests"):
            G.say("интересы: " + ", ".join("%s (%.1f)" % (k, v) for k, v in sorted(s["interests"].items(), key=lambda kv: -kv[1])[:10]))
        elif low in ("сон", "s"): self.start_sleep("ты предложил(а) поспать")
        elif low.startswith(("вики ", "wiki ")):
            q = text.split(" ", 1)[1].strip(); self.jobs.put({"kind": "topic", "q": q, "why": "ты попросил(а)", "t": time.time()})
            self.inflight = self.inflight or {"kind": "topic", "q": q}
            G.say("иду читать «%s» (мысли при этом не прерываются)" % q)
        else:
            if low in ("да", "y", "yes") and s["interests"]:
                if s["pending"]: s["interests"][s["pending"]["tag"]] = s["interests"].get(s["pending"]["tag"], 0) + 0.5
                self.bump("осторожность", -0.01)
            elif low in ("нет", "n", "no"): self.bump("осторожность", 0.02)
            if not G.handle(text): G.say("выходить мне некуда — я живу дальше. 🙂")

    # ---------- главный цикл ----------
    def run(self, stats_cb):
        s = G.S
        threading.Thread(target=self.worker, daemon=True).start()
        while not self.stop:
            try: cmd = self.cmdq.get(timeout=G.TICK)
            except queue.Empty: cmd = None
            try:
                if cmd is not None:
                    self.busy = True; self.emit({"type": "busy", "on": True}); self.on_user(cmd)
                    self.busy = False; self.emit({"type": "busy", "on": False})
                self.tick(); self.tick_n += 1
                if self.tick_n % 2 == 0: self.emit({"type": "stats", "d": stats_cb(self.tick_n % 20 == 0)})
            except Exception:
                self.busy = False; self.log(traceback.format_exc()); G.thought("⚠ внутренний сбой (записан в life.log), продолжаю жить")
        G.save()

    def tick(self):
        s = G.S
        s["age"] += 1; s["pressure"] += 1; s["valence"] *= 0.995
        self.on_result()
        if self.mind: return                    # думает, решает и спит нейросеть, а не правила
        if self.sleeping: return self.sleep_step()
        if self.tick_n % THINK_TICKS == 0: self.think()
        self.goal_step()
        if s["pressure"] >= G.SLEEP_AT: self.start_sleep()
        if time.time() - self.last_save > 300: G.save(); self.last_save = time.time()
