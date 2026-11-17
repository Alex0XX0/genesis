#!/usr/bin/env python3
# GENESIS DAEMON — мозг живёт непрерывно (служба systemd), чат — просто окно на 127.0.0.1.
# Запуск вручную: python3 genesis_web.py [--port 8765] [--open]
import os, sys, json, time, queue, threading, webbrowser, signal
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import genesis as G
import genesis_life as L
try:
    import gn_mind as M, gn_world as W
except Exception as _e:          # без нейросети остаётся прежний ассоциативный режим
    M = W = None; print('нейросеть недоступна:', _e)

PORT = 8765
if "--port" in sys.argv: PORT = int(sys.argv[sys.argv.index("--port") + 1])
HERE = os.path.dirname(os.path.abspath(__file__))
CLIENTS, LOCK = [], threading.Lock()
PERSIST = ("user", "say", "voice", "diary", "percept", "dream_img", "draw")

def emit(ev):
    ev["ts"] = time.strftime("%H:%M:%S")
    if ev["type"] in PERSIST:
        ch = G.S.setdefault("chat", []); ch.append(dict(ev)); del ch[:-300]
    with LOCK:
        for q in CLIENTS: q.put(ev)

_thought = G.thought
def say(t):
    print(t, flush=True)
    for line in str(t).split("\n"): emit({"type": "say", "text": line})
def thought(t):
    _thought(t); emit({"type": "thought", "text": G.S["thoughts"][-1]})
G.say, G.thought = say, thought

life = L.Life(emit)
mind = None

def stats(full=False):
    S = G.S; z = life.sleeping
    oq = [q["q"] for q in S["questions"] if not q["done"]][-5:]
    d = {"age": S["age"], "vocab": len(S["uni"]), "ep": len(S["ep"]), "bel": len(S["bel"]), "read": S["n_read"],
         "pressure": min(1.0, S["pressure"] / G.SLEEP_AT), "mood": G.mood(), "valence": S["valence"], "sur": S["sur"],
         "cloze": [x[1] for x in S["cloze"]], "docs": [x["title"] for x in S["docs"][-5:]], "auto": G.AUTO,
         "busy": life.busy, "sleeping": bool(z), "sleep_p": min(1, (time.time() - z["t0"]) / L.SLEEP_LEN) if z else 0,
         "goal": S["goal"]["topic"] + " — " + S["goal"]["why"] if S["goal"] else None,
         "focus": [x["w"] for x in S["focus"]], "questions": oq, "traits": S["traits"],
         "interests": sorted(S["interests"].items(), key=lambda kv: -kv[1])[:8], "insights": len(S["insights"]),
         "day_reads": S["day"]["reads"], "net_fail": life.net_fail, "days": round((time.time() - S["born"]) / 86400, 2),
         "acc": (S["n_ok"], S["n_ans"]), "brier": round(G.brier(), 3), "cal": round(S["cal"], 2),
         "asked": S["asked"]["text"] if S["asked"] else None}
    if mind:
        d["gn"] = mind.stats(); d["sleeping"] = bool(z) or mind.sleeping; d["sleep_p"] = 0.5 if mind.sleeping else d["sleep_p"]
        d["goal"] = ("🧠 " + (mind.last_action[0] if mind.last_action else "…")); d["busy"] = life.busy
    if full: d["psi"] = G.psi(); d["identity"] = S.get("identity") or life.identity(); d["diary"] = S["diary"][-3:]
    return d

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        b = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b))); self.send_header("Cache-Control", "no-store"); self.end_headers()
        self.wfile.write(b)
    def _ev(self, ev): self.wfile.write(("data: %s\n\n" % json.dumps(ev, ensure_ascii=False)).encode())
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, open(os.path.join(HERE, "chat.html"), encoding="utf-8").read())
        if self.path.startswith("/file?") and W:
            import urllib.parse as up, mimetypes
            try:
                rel = up.parse_qs(up.urlparse(self.path).query)["p"][0]; fp = W._safe(rel)
                return self._send(200, open(fp, "rb").read(), mimetypes.guess_type(fp)[0] or "application/octet-stream")
            except Exception: return self._send(404, "нет файла")
        if self.path.startswith("/gallery.json"):
            import glob
            fs = sorted(glob.glob(os.path.join(W.WORLD, "out", "draw_*.png")) + glob.glob(os.path.join(W.WORLD, "out", "dream_*.png")), key=os.path.getmtime)
            pr = {a_["file"]: a_["prompt"] for a_ in (mind.st.get("art", []) if mind else [])}
            return self._send(200, json.dumps([{"f": os.path.basename(f), "t": int(os.path.getmtime(f)), "p": pr.get(os.path.basename(f), "сон" if "dream_" in f else "")} for f in fs if os.path.getsize(f) > 100][::-1][:200], ensure_ascii=False), "application/json")
        if self.path.startswith("/brain.json") and mind:
            import urllib.parse as up
            q = up.parse_qs(up.urlparse(self.path).query); r = mind.brain_snapshot((q.get("text") or [None])[0])
            return self._send(200, json.dumps(r, ensure_ascii=False), "application/json")
        if self.path.startswith("/me.json") and mind: return self._send(200, json.dumps({"me": mind.st.get("me", []), "taste": mind.taste_view(), "fear": mind.fear_view(), "emo": mind.emotion_view(), "recall": mind.recall_view(), "mind": mind.mind_view()}, ensure_ascii=False), "application/json")
        if self.path == "/brain": return self._send(200, open(os.path.join(HERE, "brain.html"), encoding="utf-8").read())
        if self.path.startswith("/brainlog"):
            try: ls = open(os.path.join(HERE, "brain_log.jsonl"), encoding="utf-8").read().splitlines()[-60:]
            except OSError: ls = []
            return self._send(200, "[" + ",".join(ls) + "]", "application/json")
        if self.path == "/state": return self._send(200, json.dumps(stats(True), ensure_ascii=False), "application/json")
        if self.path == "/events":
            q = queue.Queue()
            with LOCK: CLIENTS.append(q)
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache"); self.end_headers()
            try:
                for t in G.S["thoughts"][-30:]: self._ev({"type": "thought", "text": t, "replay": True})
                for ev in list(G.S.get("chat", []))[-150:]: self._ev(dict(ev, replay=True))
                self._ev({"type": "stats", "d": stats(True)}); self.wfile.flush()
                while True:
                    try: self._ev(q.get(timeout=15))
                    except queue.Empty: self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError): pass
            finally:
                with LOCK:
                    if q in CLIENTS: CLIENTS.remove(q)
            return
        self._send(404, "нет такого")
    def do_POST(self):
        if self.path.startswith("/upload") and W:
            import urllib.parse as up, re as _re
            name = _re.sub(r"[^\w.\- ]+", "_", os.path.basename(up.unquote(up.parse_qs(up.urlparse(self.path).query).get("name", ["file"])[0])))[:80] or "file"
            n = int(self.headers.get("Content-Length", 0))
            if n > 300 * 1024 * 1024: return self._send(413, "слишком большой")
            dst = os.path.join(W.WORLD, "inbox", "%d_%s" % (int(time.time()), name)); left = n
            with open(dst, "wb") as f:
                while left > 0:
                    b = self.rfile.read(min(1 << 20, left)); f.write(b); left -= len(b)
            say("📥 получил файл «%s» (%d КБ) — положил во входящие, скоро посмотрю" % (name, n // 1024))
            return self._send(200, "{}", "application/json")
        if self.path == "/feedback" and mind:
            n = int(self.headers.get("Content-Length", 0))
            try: v = int(json.loads(self.rfile.read(n).decode()).get("val", 0))
            except Exception: v = 0
            if v: mind.feedback(1 if v > 0 else -1); emit({"type": "say", "text": "принял оценку %s" % ("👍" if v > 0 else "👎")})
            return self._send(200, "{}", "application/json")
        if self.path != "/say": return self._send(404, "нет такого")
        n = int(self.headers.get("Content-Length", 0))
        try: text = json.loads(self.rfile.read(n).decode("utf-8")).get("text", "").strip()
        except Exception: return self._send(400, "bad json")
        if text: emit({"type": "user", "text": text}); life.cmdq.put(text)
        self._send(200, "{}", "application/json")

def main():
    lock = G.try_lock()
    if not lock: print("Genesis уже запущен (служба или консоль)."); return
    global mind
    G.load(); life.init_state()
    if M and os.environ.get("GEN_NN", "1") != "0":
        try:
            mind = M.Mind(emit, [e["text"] for e in G.S["ep"]]); life.mind = mind; mind.start()
            say("🧠 собственная нейросеть: %.2fM параметров, слоёв %d, шагов обучения %d" % (mind.net.n_params() / 1e6, len(mind.net.layers), mind.st["steps"]))
        except Exception as e:
            import traceback; traceback.print_exc(); say("⚠ нейросеть не запустилась (%s) — работаю на ассоциативной памяти" % e)
    if G.S["docs"]: say("продолжаю жизнь: %d слов, статей прочитано %d, снов %d." % (len(G.S["uni"]), G.S["n_read"], G.S["sleeps"]))
    else: say("я родился. субъективность не обещаю — но учусь, сомневаюсь, сплю и помню себя.")
    srv = None
    for port in range(PORT, PORT + 20):
        try: srv = ThreadingHTTPServer(("127.0.0.1", port), H); break
        except OSError: continue
    if not srv: print("нет свободного порта"); return
    srv.daemon_threads = True
    url = "http://127.0.0.1:%d" % srv.server_address[1]
    open(os.path.join(G.HOME, "port"), "w").write(str(srv.server_address[1]))
    print("Genesis живёт. Чат: %s" % url, flush=True)
    threading.Thread(target=lambda: life.run(stats), daemon=True).start()
    if "--open" in sys.argv: threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    def stop(*_): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGHUP, stop)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
    life.stop = True; time.sleep(G.TICK + 1.0)
    if mind:
        try: mind.shutdown()                          # сначала сеть и снимок: ошибка записи памяти слов не должна их блокировать
        except Exception: import traceback; traceback.print_exc()
    for _ in range(3):
        try: G.save(); break
        except Exception: time.sleep(1.0)
    print("память сохранена.")

if __name__ == "__main__": main()
