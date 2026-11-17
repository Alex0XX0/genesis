#!/usr/bin/env python3
# GN-WORLD — его мир: песочница для кода (bwrap + cgroup-лимиты, без сети, только своя папка) и «школа»
# с задачами на программирование (скрытые тесты). Система хозяина и важные файлы ему недоступны.
import os, json, time, subprocess, shutil, uuid, textwrap

HOME = os.environ.get("GEN_HOME") or os.path.dirname(os.path.abspath(__file__))
WORLD = os.path.join(HOME, "world")
for d in ("projects", "inbox", "out", "run"): os.makedirs(os.path.join(WORLD, d), exist_ok=True)
MEM = os.environ.get("GEN_SBX_MEM", "768M"); CPU = os.environ.get("GEN_SBX_CPU", "150%")

def _clip(b, n=6000):
    s = b.decode("utf-8", "replace") if isinstance(b, bytes) else b
    return s if len(s) <= n else s[:n] + "\n…[обрезано]"

_BW = [None]
def _bwrap_ok():
    """bwrap нужен user-namespace; в службе systemd AppArmor его режет — тогда идём через запасную Python-песочницу."""
    if _BW[0] is None:
        try:
            r = subprocess.run(["bwrap", "--unshare-all", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
                                "--symlink", "usr/lib64", "/lib64", "true"], capture_output=True, timeout=10)
            _BW[0] = r.returncode == 0
        except Exception: _BW[0] = False
    return _BW[0]

_GUARD = r"""
import sys, os, runpy, resource
HOME_ = %(home)r; WR_ = (os.path.realpath(os.getcwd()) + os.sep, %(proj)r + os.sep)
RD_ = WR_ + (%(inbox)r + os.sep, "/usr/", "/etc/", "/proc/", "/dev/", "/lib", "/bin/", "/sys/")
resource.setrlimit(resource.RLIMIT_FSIZE, (20 << 20, 20 << 20))
def _in(p, roots):
    try: p = os.path.realpath(os.fsdecode(p)) + (os.sep if os.path.isdir(os.fsdecode(p)) else "")
    except Exception: return False
    return p.startswith(roots)
def _hook(ev, a):
    if ev == "open":
        path, mode, flags = a
        if isinstance(path, int): return
        w = (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC)) or (isinstance(mode, str) and any(c in mode for c in "wax+"))
        if w and not _in(path, WR_): raise PermissionError("писать можно только в свою папку")
        if not w and os.path.realpath(os.fsdecode(path)).startswith(HOME_) and not _in(path, RD_): raise PermissionError("чужие файлы читать нельзя")
    elif ev in ("os.remove", "os.rmdir", "os.rename", "os.truncate", "os.chmod", "os.chown", "os.mkdir", "os.link", "os.symlink", "shutil.rmtree", "shutil.move"):
        for x in a:
            if isinstance(x, (str, bytes)) and not _in(x, WR_): raise PermissionError("менять можно только свою папку")
    elif ev.startswith(("socket.", "subprocess.", "os.system", "os.exec", "os.fork", "os.posix_spawn", "os.spawn", "ctypes.", "os.kill", "os.forkpty", "pty.")):
        raise PermissionError("это в песочнице нельзя: " + ev)
sys.addaudithook(_hook)
runpy.run_path("main.py", run_name="__main__")
"""

def _run_py_guarded(code, rd, timeout):
    open(os.path.join(rd, "main.py"), "w", encoding="utf-8").write(code)
    open(os.path.join(rd, "_guard.py"), "w", encoding="utf-8").write(_GUARD % {"home": os.path.expanduser("~") + os.sep, "proj": os.path.realpath(os.path.join(WORLD, "projects")),
                                                                                 "inbox": os.path.realpath(os.path.join(WORLD, "inbox"))})
    cmd = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=" + MEM, "-p", "MemorySwapMax=0", "-p", "CPUQuota=" + CPU, "-p", "TasksMax=96", "--",
           "timeout", "-s", "KILL", str(timeout), "env", "-i", "PATH=/usr/bin:/bin", "LANG=C.UTF-8", "PYTHONIOENCODING=utf-8", "python3", "-B", "-s", "_guard.py"]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=rd, capture_output=True, timeout=timeout + 5)
        out = {"rc": r.returncode, "stdout": _clip(r.stdout), "stderr": _clip(r.stderr)}
    except subprocess.TimeoutExpired as e:
        out = {"rc": -9, "stdout": _clip(e.stdout or b""), "stderr": "превышено время"}
    ms = int((time.time() - t0) * 1000)
    out.update(ok=out["rc"] == 0, ms=ms, timeout=out["rc"] in (-9, 137, 124) and ms >= timeout * 1000 * 0.9)
    return out

def run(lang, code, timeout=10):
    """Запуск кода в песочнице. -> {"ok","rc","stdout","stderr","ms","timeout"}."""
    rid = uuid.uuid4().hex[:10]; rd = os.path.join(WORLD, "run", rid); os.makedirs(rd)
    ext = {"python": "py", "bash": "sh", "c": "c"}.get(lang)
    if not ext: return {"ok": False, "rc": -1, "stdout": "", "stderr": "язык не поддержан: " + lang, "ms": 0, "timeout": False}
    if not _bwrap_ok():
        if lang != "python":
            shutil.rmtree(rd, ignore_errors=True); return {"ok": False, "rc": -1, "stdout": "", "stderr": "bash/C без bwrap недоступны", "ms": 0, "timeout": False}
        try: return _run_py_guarded(code, rd, timeout)
        finally: shutil.rmtree(rd, ignore_errors=True)
    open(os.path.join(rd, "main." + ext), "w", encoding="utf-8").write(code)
    inner = {"python": "python3 -B main.py", "bash": "bash main.sh",
             "c": "gcc -O1 -o prog main.c -lm && ./prog"}[lang]
    cmd = ["timeout", "-s", "KILL", str(timeout), "bwrap", "--unshare-all", "--die-with-parent", "--new-session",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--ro-bind-try", "/etc/alternatives", "/etc/alternatives", "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache",
           "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--bind", rd, "/work", "--bind", os.path.join(WORLD, "projects"), "/projects",
           "--ro-bind", os.path.join(WORLD, "inbox"), "/inbox", "--chdir", "/work", "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
           "--setenv", "HOME", "/work", "--setenv", "LANG", "C.UTF-8", "--setenv", "PYTHONIOENCODING", "utf-8", "bash", "-c", inner]
    scope = ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=" + MEM, "-p", "MemorySwapMax=0", "-p", "CPUQuota=" + CPU, "-p", "TasksMax=96", "--"]
    t0 = time.time()
    try:
        r = subprocess.run(scope + cmd, capture_output=True, timeout=timeout + 5)
        out = {"rc": r.returncode, "stdout": _clip(r.stdout), "stderr": _clip(r.stderr)}
    except subprocess.TimeoutExpired as e:
        out = {"rc": -9, "stdout": _clip(e.stdout or b""), "stderr": "превышено время"}
    ms = int((time.time() - t0) * 1000)
    out.update(ok=out["rc"] == 0, ms=ms, timeout=out["rc"] in (-9, 137, 124) and ms >= timeout * 1000 * 0.9)
    shutil.rmtree(rd, ignore_errors=True)
    return out

# ---------- файловая часть (только внутри world/) ----------
def _safe(rel):
    p = os.path.realpath(os.path.join(WORLD, rel.lstrip("/")))
    if not (p == WORLD or p.startswith(WORLD + os.sep)): raise ValueError("за пределами мира")
    return p

def ls(rel="."):
    p = _safe(rel)
    return sorted(("%s/" % n if os.path.isdir(os.path.join(p, n)) else n) for n in os.listdir(p) if n != "run")[:200]

def read(rel, n=6000): return open(_safe(rel), encoding="utf-8", errors="replace").read(n)
def write(rel, text):
    p = _safe(rel); os.makedirs(os.path.dirname(p), exist_ok=True)
    if len(text) > 200_000: raise ValueError("слишком большой файл")
    open(p, "w", encoding="utf-8").write(text); return p

# ---------- школа ----------
def _k(id, title, prompt, sol, tests, seed=True):
    return {"id": id, "title": title, "prompt": prompt, "solution": textwrap.dedent(sol).strip() + "\n", "tests": textwrap.dedent(tests).strip() + "\n", "seed": seed}

KATA = [
    _k("sum_list", "Сумма списка", "Напиши функцию sum_list(xs), возвращающую сумму чисел списка.",
       "def sum_list(xs):\n    t = 0\n    for x in xs:\n        t += x\n    return t", "assert sum_list([1,2,3])==6\nassert sum_list([])==0\nassert sum_list([-1,1])==0"),
    _k("is_even", "Чётность", "Напиши функцию is_even(n): True, если n чётное.",
       "def is_even(n):\n    return n % 2 == 0", "assert is_even(4) and not is_even(7) and is_even(0)"),
    _k("rev_str", "Разворот строки", "Напиши функцию rev_str(s), возвращающую строку задом наперёд.",
       "def rev_str(s):\n    return s[::-1]", "assert rev_str('abc')=='cba'\nassert rev_str('')==''"),
    _k("count_vowels", "Гласные", "Напиши функцию count_vowels(s): число гласных (латиница aeiou и кириллица аеёиоуыэюя), без учёта регистра.",
       "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiouаеёиоуыэюя')", "assert count_vowels('Hello')==2\nassert count_vowels('Привет')==2\nassert count_vowels('')==0"),
    _k("factorial", "Факториал", "Напиши функцию factorial(n) для n >= 0.",
       "def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r", "assert factorial(0)==1\nassert factorial(5)==120"),
    _k("max_in", "Максимум", "Напиши функцию max_in(xs), возвращающую наибольший элемент непустого списка, не используя max().",
       "def max_in(xs):\n    m = xs[0]\n    for x in xs:\n        if x > m:\n            m = x\n    return m", "assert max_in([3,9,2])==9\nassert max_in([-5,-2])==-2"),
    _k("is_palindrome", "Палиндром", "Напиши функцию is_palindrome(s): палиндром ли строка без учёта регистра и пробелов.",
       "def is_palindrome(s):\n    t = ''.join(c for c in s.lower() if c != ' ')\n    return t == t[::-1]", "assert is_palindrome('А роза упала на лапу Азора')\nassert not is_palindrome('abc')"),
    _k("gcd", "НОД", "Напиши функцию gcd(a, b) — наибольший общий делитель.",
       "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a", "assert gcd(12,18)==6\nassert gcd(7,13)==1"),
    # экзаменационные (в обучении не показываются)
    _k("fib", "Фибоначчи", "Напиши функцию fib(n): n-е число Фибоначчи (fib(0)=0, fib(1)=1).",
       "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a", "assert [fib(i) for i in range(8)]==[0,1,1,2,3,5,8,13]", False),
    _k("fizzbuzz", "FizzBuzz", "Напиши функцию fizzbuzz(n): список строк от 1 до n; кратные 3 -> 'Fizz', 5 -> 'Buzz', обоим -> 'FizzBuzz', иначе число строкой.",
       "def fizzbuzz(n):\n    r = []\n    for i in range(1, n + 1):\n        if i % 15 == 0: r.append('FizzBuzz')\n        elif i % 3 == 0: r.append('Fizz')\n        elif i % 5 == 0: r.append('Buzz')\n        else: r.append(str(i))\n    return r",
       "assert fizzbuzz(5)==['1','2','Fizz','4','Buzz']\nassert fizzbuzz(15)[-1]=='FizzBuzz'", False),
    _k("word_count", "Подсчёт слов", "Напиши функцию word_count(text): словарь {слово: сколько раз}, слова по пробелам, в нижнем регистре.",
       "def word_count(text):\n    d = {}\n    for w in text.lower().split():\n        d[w] = d.get(w, 0) + 1\n    return d", "assert word_count('a b A')=={'a':2,'b':1}", False),
    _k("primes", "Простые числа", "Напиши функцию primes(n): список простых чисел до n включительно.",
       "def primes(n):\n    r = []\n    for k in range(2, n + 1):\n        if all(k % d for d in range(2, int(k ** 0.5) + 1)):\n            r.append(k)\n    return r", "assert primes(10)==[2,3,5,7]\nassert primes(1)==[]", False),
    _k("flatten", "Выпрямление", "Напиши функцию flatten(xs): превращает вложенные списки в плоский список.",
       "def flatten(xs):\n    r = []\n    for x in xs:\n        if isinstance(x, list): r += flatten(x)\n        else: r.append(x)\n    return r", "assert flatten([1,[2,[3]],4])==[1,2,3,4]", False),
    _k("two_sum", "Два числа", "Напиши функцию two_sum(nums, target): индексы (i, j), i<j, чьи значения в сумме дают target.",
       "def two_sum(nums, target):\n    for i in range(len(nums)):\n        for j in range(i + 1, len(nums)):\n            if nums[i] + nums[j] == target: return (i, j)", "assert two_sum([2,7,11,15],9)==(0,1)", False),
]
KATA_BY_ID = {k["id"]: k for k in KATA}
ARENA_FILE = os.path.join(HOME, "arena.json")

def arena_load():
    try: return json.load(open(ARENA_FILE))
    except Exception: return {"solved": {}, "tries": {}, "xp": 0}

def arena_save(a):
    tmp = ARENA_FILE + ".tmp"; json.dump(a, open(tmp, "w"), ensure_ascii=False); os.replace(tmp, ARENA_FILE)

def kata_text(k, with_solution=True):
    t = "Задача: %s\nРешение:\n```python\n" % k["prompt"]
    return t + (k["solution"] + "```\n" if with_solution else "")

def check(kid, code):
    """Проверка решения скрытыми тестами. -> (ok, details)."""
    k = KATA_BY_ID[kid]
    r = run("python", code + "\n\n" + k["tests"] + "print('GN_OK')\n", timeout=8)
    ok = r["ok"] and "GN_OK" in r["stdout"]
    a = arena_load(); a["tries"][kid] = a["tries"].get(kid, 0) + 1
    if ok and kid not in a["solved"]:
        a["solved"][kid] = {"t": time.time(), "tries": a["tries"][kid]}; a["xp"] += 10 if k["seed"] else 30
    arena_save(a)
    return ok, (r["stderr"] or r["stdout"])[-300:]

if __name__ == "__main__":
    print(run("python", "print(sum(range(10)))"))
