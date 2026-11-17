#!/usr/bin/env python3
# GN-SENSES — восприятие. Картинка / звук / видео (любой формат: декодирует ffmpeg) превращаются
# в последовательности дискретных кодов тем же потоком, что и текст. Коды даёт ОНЛАЙН-VQ:
# кодовая книга учится на лету и растёт (новый код рождается, когда увиденное не похоже на известное).
# Плюс «факты восприятия» (OCR, цвета, длительность, ритм, лица) — как текстовая подпись для связывания модальностей.
import os, subprocess, json, math, shutil, tempfile, re
import numpy as np

def _run(cmd, timeout=60, binary=False):
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        return r.stdout if binary else r.stdout.decode("utf-8", "replace")
    except Exception: return b"" if binary else ""

class VQ:
    def __init__(self, dim, cap=192, name="i"):
        self.dim, self.cap, self.name = dim, cap, name
        self.C = np.zeros((0, dim), np.float32); self.cnt = np.zeros(0, np.float32); self.ema = None

    def encode(self, X, learn=True):
        X = np.asarray(X, np.float32); out = []
        for x in X:
            if len(self.C):
                d = ((self.C - x) ** 2).sum(1); j = int(d.argmin()); dist = float(math.sqrt(d[j]))
                self.ema = dist if self.ema is None else (0.995 * self.ema + 0.005 * dist)
            else: j, dist = -1, 1e9
            if learn and len(self.C) < self.cap and (len(self.C) < 8 and dist > 1e-3 or dist > 1.6 * (self.ema or 0) + 1e-6):
                self.C = np.vstack([self.C, x[None]]); self.cnt = np.append(self.cnt, 1.0); j = len(self.C) - 1   # рождение нового кода
            elif learn and j >= 0:
                self.cnt[j] += 1; self.C[j] += (x - self.C[j]) * max(1.0 / self.cnt[j], 0.02)
            out.append(j)
        return out

    def save(self, path):
        tmp = path + ".tmp.npz"; np.savez(tmp, C=self.C, cnt=self.cnt, ema=np.array(self.ema or 0.0)); os.replace(tmp, path)
    @classmethod
    def load(cls, path, dim, cap, name):
        v = cls(dim, cap, name)
        if os.path.exists(path):
            z = np.load(path); v.C, v.cnt, v.ema = z["C"], z["cnt"], float(z["ema"]) or None
        return v

COLORS = [("чёрный", (0, 0, 0)), ("белый", (255, 255, 255)), ("серый", (128, 128, 128)), ("красный", (220, 30, 30)), ("зелёный", (40, 170, 60)),
          ("синий", (40, 70, 210)), ("жёлтый", (240, 220, 40)), ("оранжевый", (240, 140, 30)), ("фиолетовый", (140, 60, 190)),
          ("розовый", (240, 130, 180)), ("коричневый", (120, 75, 40)), ("голубой", (110, 190, 240))]

def mel_matrix(sr=16000, n_fft=512, n_mels=24):
    f = lambda m: 700 * (10 ** (m / 2595) - 1); h = lambda x: 2595 * np.log10(1 + x / 700)
    pts = f(np.linspace(h(60), h(sr / 2), n_mels + 2)); bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    M = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for i in range(n_mels):
        a, b, c = bins[i], bins[i + 1], bins[i + 2]
        for k in range(a, b): M[i, k] = (k - a) / max(b - a, 1)
        for k in range(b, c): M[i, k] = (c - k) / max(c - b, 1)
    return M
MEL = mel_matrix()

class Senses:
    def __init__(self, tok, home):
        self.tok, self.home = tok, home
        self.vi = VQ.load(os.path.join(home, "vq_img.npz"), 192, 256, "i")
        self.va = VQ.load(os.path.join(home, "vq_aud.npz"), 24, 128, "a")

    def save(self):
        self.vi.save(os.path.join(self.home, "vq_img.npz")); self.va.save(os.path.join(self.home, "vq_aud.npz"))

    # ---------- общий вход ----------
    def perceive(self, path, learn=True):
        """-> {"kind","text","ids"}. ids — токены для сети; text — факты восприятия."""
        if not os.path.isfile(path): return {"kind": "none", "text": "файла нет", "ids": []}
        mime = _run(["file", "-b", "--mime-type", path]).strip()
        ext = os.path.splitext(path)[1].lower()
        if mime.startswith("image/") or ext in (".heic", ".avif", ".svg"): return self.image(path, learn)
        if mime.startswith("video/"): return self.video(path, learn)
        if mime.startswith("audio/") or ext in (".mp3", ".ogg", ".opus", ".m4a", ".flac", ".wav", ".aac", ".wma"): return self.audio(path, learn)
        if mime == "application/pdf":
            t = _run(["pdftotext", "-l", "6", "-layout", path, "-"])[:3000]
            return {"kind": "pdf", "text": "PDF-документ. Текст: " + re.sub(r"\s+", " ", t).strip(), "ids": []}
        if mime.startswith("text/") or ext in (".py", ".js", ".md", ".json", ".csv", ".html", ".c", ".rs", ".go", ".sh", ".txt"):
            try: t = open(path, encoding="utf-8", errors="replace").read(4000)
            except Exception: t = ""
            return {"kind": "text", "text": t, "ids": []}
        # неизвестный формат: пробуем ffmpeg как мультимедиа
        if _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path]).strip():
            return self.video(path, learn)
        return {"kind": "unknown", "text": "Неизвестный формат: " + _run(["file", "-b", path]).strip()[:200], "ids": []}

    # ---------- изображения ----------
    def _load_rgb(self, path, size):
        from PIL import Image
        try: im = Image.open(path); im.load()
        except Exception:
            tmp = tempfile.mktemp(suffix=".png"); _run(["ffmpeg", "-v", "error", "-y", "-i", path, "-frames:v", "1", tmp])
            if not os.path.exists(tmp): _run(["convert", path + "[0]", tmp])
            if not os.path.exists(tmp): return None
            im = Image.open(tmp); im.load(); os.remove(tmp)
        im = im.convert("RGB"); w, h = im.size; s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size))
        return np.asarray(im, np.float32) / 255.0, (w, h)

    def _img_codes(self, arr, patch, learn):
        n = arr.shape[0] // patch
        P = arr.reshape(n, patch, n, patch, 3).transpose(0, 2, 1, 3, 4).reshape(n * n, -1)
        return self.vi.encode(P, learn), n

    def _facts_image(self, path, arr, dims):
        px = (arr.reshape(-1, 3) * 255)
        names = []
        for p in px[:: max(1, len(px) // 400)]:
            names.append(min(COLORS, key=lambda c: sum((a - b) ** 2 for a, b in zip(c[1], p)))[0])
        top = [n for n, _ in __import__("collections").Counter(names).most_common(3)]
        g = arr.mean(2); edge = float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())
        ocr = re.sub(r"\s+", " ", _run(["tesseract", path, "-", "-l", "rus+eng", "--psm", "6"], 40)).strip()[:300]
        faces = 0
        try:
            import cv2
            im = cv2.imread(path)
            if im is not None:
                cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = len(cas.detectMultiScale(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), 1.2, 5))
        except Exception: pass
        return ("Изображение %dx%d. Основные цвета: %s. Яркость %.0f%%. Детальность %s. Лиц: %d.%s" %
                (dims[0], dims[1], ", ".join(top), g.mean() * 100, "высокая" if edge > 0.12 else "средняя" if edge > 0.05 else "низкая",
                 faces, (" Текст на картинке: «%s»." % ocr) if len(ocr) > 3 else ""))

    def image(self, path, learn=True):
        r = self._load_rgb(path, 64)
        if r is None: return {"kind": "image", "text": "не смог открыть изображение", "ids": []}
        arr, dims = r; codes, n = self._img_codes(arr, 8, learn); T = self.tok
        ids = [T.id("<img>")]
        for i in range(n):
            ids += [T.code("i", c) for c in codes[i * n:(i + 1) * n] if T.code("i", c) is not None] + [T.id("<row>")]
        ids.append(T.id("</img>"))
        return {"kind": "image", "text": self._facts_image(path, arr, dims), "ids": ids}

    # ---------- аудио ----------
    def _decode_audio(self, path, seconds=12):
        raw = _run(["ffmpeg", "-v", "error", "-i", path, "-t", str(seconds), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], 60, True)
        return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0 if raw else np.zeros(0, np.float32)

    def _mel_frames(self, y, hop=640, n_fft=512):
        if len(y) < n_fft: return np.zeros((0, 24), np.float32)
        win = np.hanning(n_fft).astype(np.float32); fr = []
        for s in range(0, len(y) - n_fft, hop):
            sp = np.abs(np.fft.rfft(y[s:s + n_fft] * win)) ** 2; fr.append(np.log(MEL @ sp + 1e-6))
        return (np.array(fr, np.float32) + 14.0) / 6.0

    def _facts_audio(self, path, y, M):
        dur = float(_run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path]).strip() or len(y) / 16000)
        if not len(y): return "Аудио без звука или не читается."
        rms = float(np.sqrt((y ** 2).mean())); sil = float((np.abs(y) < 0.01).mean())
        spec = np.abs(np.fft.rfft(y[:16000 * 8])); fr = np.fft.rfftfreq(len(y[:16000 * 8]), 1 / 16000); cen = float((spec * fr).sum() / (spec.sum() + 1e-9))
        env = np.abs(y[: len(y) // 400 * 400]).reshape(-1, 400).mean(1); env = env - env.mean(); bpm = 0
        if len(env) > 80:
            ac = np.correlate(env, env, "full")[len(env) - 1:]; lo, hi = int(40 / 60 * 0 + 8), min(len(ac) - 1, 80)
            lag = lo + int(np.argmax(ac[lo:hi])); bpm = 60.0 / (lag * 400 / 16000) if lag else 0
        return ("Аудио, %.1f с. Громкость %s. Тишины %.0f%%. Тембр %s (центроид %.0f Гц).%s" %
                (dur, "высокая" if rms > 0.12 else "средняя" if rms > 0.03 else "тихая", sil * 100, "яркий" if cen > 2500 else "средний" if cen > 1200 else "тёмный",
                 cen, (" Ритм ≈ %.0f уд/мин." % bpm) if 50 < bpm < 200 else ""))

    def audio(self, path, learn=True):
        y = self._decode_audio(path); M = self._mel_frames(y); T = self.tok
        codes = self.va.encode(M, learn) if len(M) else []
        ids = [T.id("<aud>")] + [x for x in (T.code("a", c) for c in codes) if x is not None] + [T.id("</aud>")]
        return {"kind": "audio", "text": self._facts_audio(path, y, M), "ids": ids}

    # ---------- видео ----------
    def video(self, path, learn=True, frames=6):
        info = _run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height,r_frame_rate:format=duration", "-of", "json", path])
        try: j = json.loads(info)
        except Exception: j = {}
        dur = float(j.get("format", {}).get("duration", 0) or 0); T = self.tok
        vs = [s for s in j.get("streams", []) if s.get("codec_type") == "video"]; has_a = any(s.get("codec_type") == "audio" for s in j.get("streams", []))
        ids = [T.id("<vid>")]; facts = []
        d = tempfile.mkdtemp()
        try:
            for k in range(frames):
                t = dur * (k + 0.5) / frames if dur else k; fp = os.path.join(d, "f%d.png" % k)
                _run(["ffmpeg", "-v", "error", "-y", "-ss", "%.2f" % t, "-i", path, "-frames:v", "1", fp], 40)
                if not os.path.exists(fp): continue
                r = self._load_rgb(fp, 32)
                if r is None: continue
                codes, n = self._img_codes(r[0], 8, learn)
                ids += [x for x in (T.code("i", c) for c in codes) if x is not None] + [T.id("<row>")]
                if k == 0: facts.append(self._facts_image(fp, self._load_rgb(fp, 64)[0], r[1]))
            a = self.audio(path, learn) if has_a else None
            if a: ids += a["ids"][:80]
        finally: shutil.rmtree(d, ignore_errors=True)
        ids.append(T.id("</vid>"))
        w, h = (vs[0].get("width"), vs[0].get("height")) if vs else (0, 0)
        txt = "Видео %dx%d, %.1f с%s. Кадр: %s%s" % (w or 0, h or 0, dur, ", со звуком" if has_a else ", без звука", facts[0] if facts else "—", (" " + a["text"]) if a else "")
        return {"kind": "video", "text": txt, "ids": ids}
