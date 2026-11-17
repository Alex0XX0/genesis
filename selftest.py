#!/usr/bin/env python3
"""Объективная батарея «насколько он знает про себя». Читает только сохранённые файлы (сеть, токенайзер, состояние), службу не трогает.
Каждый тест сравнивается с шансом или с глупым базовым методом. Это НЕ тест на сознание (такого теста не существует):
он меряет самознание в узком смысле: совпадает ли то, что он говорит и предсказывает о себе, с тем, что на самом деле с ним происходит.
Запуск: python3 selftest.py   ->  печатает таблицу и пишет selftest_report.md"""
import sys, os, json, math, time, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gn_net as N, gn_tok as K
from gn_mind import Emotions, ACTIONS

F = np.float32
net = N.Net.load(os.path.join(HERE, "gn_net.npz")); tok = K.Tok.load(os.path.join(HERE, "gn_tok.json")); net.set_vocab(len(tok))
st = json.load(open(os.path.join(HERE, "gn_state.json"), encoding="utf-8"))
EM = list(Emotions.SPEC.keys()); NE = len(EM)
rows = []        # (название, результат, шанс/база, вердикт, пояснение)

def verdict(ok, partial=False): return "✅ да" if ok else ("🟡 слабо" if partial else "❌ нет")

def seq_nll(ids_list, sv=None):
    """Пословный nll для списка последовательностей (с паддингом), состояние sv одно на все."""
    L = max(len(x) for x in ids_list); B = len(ids_list); ids = np.zeros((B, L), np.int64); w = np.zeros((B, L - 1), F)
    for b, x in enumerate(ids_list): ids[b, :len(x)] = x; w[b, :len(x) - 1] = 1
    S = None if sv is None else np.broadcast_to(np.asarray(sv, F), (B, L - 1, N.NS))
    with N.no_grad(): _, nll = net.loss(ids[:, :-1], ids[:, 1:], w, S)
    return nll * w

# ---------- T1: причинная проверка «состояние -> слова о себе» (вмешательство в состояние) ----------
prompts = ["Теодор: Что ты чувствуешь?\nGenesis: Я чувствую", "Теодор: Как дела?\nGenesis: Я чувствую", "Теодор: Что с тобой?\nGenesis: Я чувствую"]
words = [e for e in EM if e not in ("усталость", "страх")]; idx = [EM.index(e) for e in words]; n = len(words)
LP = np.zeros((n, n))       # LP[i, j]: насколько вероятно слово j, если я принудительно включил эмоцию i
for pr in prompts:
    pre = [K.Tok.BOS] + tok.encode(pr, grow=False); cand = []; plen = []
    for wd in words:
        full = [K.Tok.BOS] + tok.encode(pr + " " + wd, grow=False); p = 0
        while p < min(len(pre), len(full)) and pre[p] == full[p]: p += 1
        cand.append(full); plen.append(p)
    for i in range(n):
        sv = np.zeros(N.NS, F); sv[idx[i]] = 0.9
        nll = seq_nll(cand, sv)
        for j in range(n): LP[i, j] += -float(nll[j, plen[j] - 1:len(cand[j]) - 1].sum())
LP /= len(prompts)
col_acc = float(np.mean([np.argmax(LP[:, j]) == j for j in range(n)]))            # слово j самое вероятное именно при состоянии j (убирает частоту слов)
lift = float(np.mean([LP[j, j] - LP[:, j].mean() for j in range(n)]))              # на сколько нат вероятнее слово при «своём» состоянии, чем в среднем
from math import comb
k = int(round(col_acc * n)); pval = sum(comb(n, x) * (1 / n) ** x * (1 - 1 / n) ** (n - x) for x in range(k, n + 1))
rows.append(("T1 вмешательство: включаю эмоцию → растёт ли вероятность её слова", "точность %.0f%% (%d из %d), прирост %+.3f нат" % (100 * col_acc, k, n, lift), "шанс %.0f%%, p=%.3f" % (100 / n, pval),
             verdict(pval < 0.01 and lift > 0.05, pval < 0.1 or lift > 0.02), "причинная связь внутреннего состояния и слов о нём: главный тест «говорит о том, что с ним реально»"))
rows_t1 = (LP, words)

# ---------- T2: зеркало авторства: «я это говорил или Теодор?» по чату ----------
chat = [x for x in st.get("chat", []) if len(x) == 2 and 8 <= len(x[1]) <= 160][-40:]
ok = tot = 0
for who, text in chat:
    a = [K.Tok.BOS] + tok.encode("Genesis: " + text, grow=False); b = [K.Tok.BOS] + tok.encode("Теодор: " + text, grow=False)
    na = float(seq_nll([a])[0].sum()) ; nb = float(seq_nll([b])[0].sum())
    tot += 1; ok += int((na < nb) == (who == "Genesis"))
rows.append(("T2 зеркало: узнаёт свои реплики среди реплик Теодора", "%.0f%% (%d из %d)" % (100 * ok / max(tot, 1), ok, tot), "шанс 50%", verdict(tot >= 12 and ok / tot > 0.75, tot >= 12 and ok / tot > 0.6),
             "осторожно: эти реплики он частично видел при обучении, так что результат завышен"))

# ---------- T3: схема внимания против глупых базовых методов ----------
fl = st.get("focus_log", []); kinds = [x[1] for x in fl][::1]; hits = st.get("att_hits", [])
if len(kinds) >= 12 and len(hits) >= 30:
    same = np.mean([kinds[i] == kinds[i - 1] for i in range(1, len(kinds))]); cnt = collections.Counter(kinds); freq = max(cnt.values()) / len(kinds)
    mine = sum(hits) / len(hits); base = max(same, freq)
    rows.append(("T3 предсказывает собственный следующий фокус внимания", "%.0f%% на %d" % (100 * mine, len(hits)), "база (повторить прошлое/самое частое): %.0f%%" % (100 * base), verdict(mine > base + 0.1, mine > base),
                 "если не лучше простой базы, это не самопредсказание, а привычка"))
else: rows.append(("T3 предсказание своего внимания", "мало данных", "-", "—", ""))

# ---------- T4: метапознание: ожидания против результата ----------
mt = np.array(st.get("meta", []))
if len(mt) >= 40:
    e, g = mt[:, 0], mt[:, 1]; corr = float(np.corrcoef(e, g)[0, 1]) if e.std() > 1e-6 and g.std() > 1e-6 else 0.0
    q = np.quantile(e, [1 / 3, 2 / 3]); lo, mid, hi = g[e <= q[0]].mean(), g[(e > q[0]) & (e <= q[1])].mean(), g[e > q[1]].mean()
    se = 1.0 / math.sqrt(len(mt))
    rows.append(("T4 калибровка: ожидал хорошо → получил хорошо", "корреляция %+.2f на %d (тройки: %+.2f / %+.2f / %+.2f)" % (corr, len(mt), lo, mid, hi), "шанс 0 ± %.2f" % (2 * se), verdict(corr > 2 * se and hi > lo, corr > 0),
                 "если ≤ 0, он не знает, чего от себя ждать: «знание о своих силах» не работает"))
else: rows.append(("T4 калибровка", "мало данных", "-", "—", ""))

# ---------- T5: авторство: предсказание последствий своих действий ----------
ag, agn = st.get("ag_skill"), st.get("ag_n", 0)
rows.append(("T5 авторство: предсказывает, как его действия меняют его самого", "лучше шанса на %s на %d действиях" % ("%.0f%%" % (100 * ag) if ag is not None else "—", agn), "0%", verdict((ag or 0) > 0.1 and agn >= 100, (ag or 0) > 0.03),
             "навык прямой модели против «ничего не изменится»"))

# ---------- T6: память о собственном опыте ----------
rc = st.get("recall_fixed_last")
rows.append(("T6 помнит, что ему говорили (фиксированная выборка)", ("%.0f%%" % (100 * rc)) if rc is not None else "—", "до сна/без повторений ≈ 0-10%", verdict((rc or 0) > 0.5, (rc or 0) > 0.25), "эпизодическая память: из дневного опыта в весах"))

# ---------- T7: связь состояние→речь на отложенных примерах ----------
sg = (st.get("sv_gain") or [None])[-1]
rows.append(("T7 состояние помогает говорить (отложенные примеры)", ("%+.3f нат (чужое состояние хуже), n=%s" % (sg[1], sg[3])) if sg else "копит данные", "0", verdict(bool(sg) and sg[1] > 0.02, bool(sg) and sg[1] > 0), "тот же вопрос, что T1, но на естественных примерах"))

# ---------- сводка ----------
score = sum(1 for r in rows if r[3].startswith("✅")) ; part = sum(1 for r in rows if r[3].startswith("🟡")); tot = sum(1 for r in rows if r[3] != "—")
out = ["# Самотест Genesis (%s, шагов %s)\n" % (time.strftime("%F %T"), st.get("steps")), "| тест | результат | шанс / база | вердикт |", "|---|---|---|---|"]
for r in rows: out.append("| %s | %s | %s | %s |" % r[:4])
out += ["", "**Итого: прошёл %d, слабо %d, не прошёл %d из %d.**" % (score, part, tot - score - part, tot), ""]
out += ["Пояснения:"] + ["- %s: %s" % (r[0].split(":")[0], r[4]) for r in rows if r[4]]
LPm, w_ = rows_t1
best = sorted(((LPm[j, j] - LPm[:, j].mean(), w_[j]) for j in range(n)), reverse=True)
out += ["", "Эмоции, чьи слова лучше всего привязаны к состоянию (прирост нат): " + ", ".join("%s %+.2f" % (w, v) for v, w in best[:5]),
        "Хуже всего: " + ", ".join("%s %+.2f" % (w, v) for v, w in best[-4:])]
out += ["", "Оговорка: это измерение самознания в узком смысле (согласие слов и предсказаний с реальным внутренним состоянием), а не доказательство сознания: такого теста у науки нет."]
txt = "\n".join(out); print(txt); open(os.path.join(HERE, "selftest_report.md"), "w", encoding="utf-8").write(txt + "\n")
