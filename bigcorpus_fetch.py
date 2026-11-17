#!/usr/bin/env python3
# Качает начало дампов Википедии, чистит разметку и пишет текстовые шарды в corpus_big/txt (≈1 ГБ всего).
import os, re, sys, bz2, subprocess, html
H = os.path.dirname(os.path.abspath(__file__)); RAW = H + "/corpus_big/raw"; TXT = H + "/corpus_big/txt"
SRC = [("ru", "https://dumps.wikimedia.org/ruwiki/latest/ruwiki-latest-pages-articles-multistream.xml.bz2", 250),
       ("simple", "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles-multistream.xml.bz2", 45)]
SHARD = 64 * 1024 * 1024
TPL = re.compile(r"\{\{[^{}]*\}\}"); LNK = re.compile(r"\[\[(?:[^\[\]|]*\|)?([^\[\]|]*)\]\]"); FILE = re.compile(r"\[\[(?:Файл|File|Image|Изображение|Категория|Category):[^\[\]]*(?:\[\[[^\[\]]*\]\][^\[\]]*)*\]\]", re.I)
REF = re.compile(r"<ref[^>]*?/>|<ref[^>]*>.*?</ref>", re.S); TAG = re.compile(r"<[^>]+>"); EXT = re.compile(r"\[https?://\S+(?: ([^\]]*))?\]")
def clean(t):
    t = html.unescape(t); t = REF.sub("", t); t = re.sub(r"<!--.*?-->", "", t, flags=re.S); t = re.sub(r"\{\|.*?\|\}", "", t, flags=re.S)
    for _ in range(4): t = TPL.sub("", t)
    t = FILE.sub("", t); t = LNK.sub(r"\1", t); t = EXT.sub(lambda m: m.group(1) or "", t); t = TAG.sub("", t); t = html.unescape(t)
    t = re.sub(r"'{2,}", "", t); t = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", t, flags=re.M)
    out = []
    for p in t.split("\n"):
        p = p.strip()
        if len(p) < 60 or p[0] in "*#:;|!{}" or p.lower().startswith(("категория", "category")): continue
        out.append(p)
    return "\n".join(out)
def run(tag, url, mb):
    f = "%s/%s.bz2" % (RAW, tag)
    if not os.path.exists(f) or os.path.getsize(f) < mb * 1024 * 1024 * 0.99:
        subprocess.run(["curl", "-s", "-L", "-r", "0-%d" % (mb * 1024 * 1024 - 1), "-o", f, url], check=True)
    d = bz2.BZ2Decompressor(); buf = ""; n = 0; shard = 0; out = open("%s/%s_%03d.txt" % (TXT, tag, shard), "w"); size = 0
    with open(f, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk: break
            try:
                while chunk or d.needs_input is False:
                    buf += d.decompress(chunk).decode("utf-8", "ignore"); chunk = b""
                    if d.eof: left = d.unused_data; d = bz2.BZ2Decompressor(); chunk = left
                    if not chunk: break
            except (OSError, EOFError): break
            while True:
                a = buf.find("<text"); 
                if a < 0: buf = buf[-200:]; break
                b = buf.find("</text>", a)
                if b < 0: buf = buf[a:]; break
                body = buf[buf.find(">", a) + 1:b]; buf = buf[b + 7:]
                if body.lstrip().lower().startswith("#redirect") or "#перенаправление" in body[:40].lower(): continue
                t = clean(body)
                if len(t) > 400: out.write(t + "\n\n"); size += len(t); n += 1
                if size > SHARD: out.close(); shard += 1; size = 0; out = open("%s/%s_%03d.txt" % (TXT, tag, shard), "w")
            if n and n % 2000 < 3: print(tag, n, "статей", flush=True)
    out.close(); print(tag, "готово:", n, "статей", flush=True)
for s in SRC: run(*s)
print("ALL DONE", flush=True)
