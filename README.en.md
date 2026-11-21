# Genesis

> [Русская версия — основная](README.md)

Project presentation (download `docs/presentation.html` and open it in a browser, text in Russian): [docs/presentation.html](docs/presentation.html)

A small, continuously living agent built around **a transformer written from scratch in NumPy** (no PyTorch, no pretrained model, no external LLM).
It reads, writes code, draws, sleeps, remembers, evolves its own architecture, and keeps an internal state (emotions, body signals, attention)
that is fed **into the network itself**, so that what it says can depend on what is going on inside it.

This is a **research playground**, not a product, and **not a claim of consciousness**. See [Honest status](#honest-status).


## Why this exists

I wanted to understand modern language models from the inside, so I built every part myself and measured every change:
autograd, attention, optimizer, tokenizer, growth, evaluation. The agent layer on top asks a narrower research question:

*What happens if a tiny language model is trained continuously on its own experience, can modify its own architecture under a statistical test, and has a
learned internal state that it can read, report and be measured on?*

## What is in it

| Layer | What it does | Where |
|---|---|---|
| **Network** | Pre-LN transformer (d=128, 4 heads, ctx 128), tied embeddings, hand-written reverse-mode autograd, Muon (for matrices) + Adam, KV-cache generation | `gn_net.py` |
| **Tokenizer** | Word-level BPE that grows during life (up to 4096 merges), plus 256 VQ codes for image patches | `gn_tok.py`, `gn_senses.py` |
| **Growth & evolution** | Function-preserving growth (add a zero-output layer, widen an MLP), then *tournaments*: clone the net, apply a mutation, train both on the same batches, accept only if a paired t-test on held-out text says it is better | `gn_net.py`, `gn_mind.py: a_evolve` |
| **Actions** | A softmax policy (REINFORCE) chooses among 17 activities: read Wikipedia, read books, study / try / write code in a sandbox, take exams with hidden tests, look at pictures, draw, sleep, rest, speak, reflect on itself, protect its memory, evolve | `gn_mind.py` |
| **Learned value & emotions** | A linear TD value model over state + context; the prediction error drives 25 functional emotions with half-lives and soft saturation. The list of emotion names is innate, *what triggers them* is learned. | `gn_mind.py: ValueModel, Emotions` |
| **Taste & curiosity** | LinUCB curiosity; a taste model with a Wundt curve (liking peaks at moderate surprise) | `gn_mind.py: Taste, Curiosity` |
| **State inside the network** | A 64-d state vector (emotions, mood, fatigue, fear, attention focus, memory score, agency skill, last action) is projected by a learned matrix `Ws` and added to every token embedding. `Ws` is zero at birth, so it is function-preserving. Experience is stored together with the state it happened in | `gn_net.py`, `gn_mind.py: state_embed` |
| **Self-model machinery** | Global-workspace-style attention spotlight, attention-schema prediction, metacognitive calibration, theory-of-mind of the human, forward model of its own actions ("agency"), a self-written list of facts about itself, a fear-of-losing-memory drive | `gn_mind.py` |
| **Sleep** | Consolidation, replay of forgotten phrases, a fixed-sample recall probe, dream images and text | `gn_mind.py: sleep, sleep_recall` |
| **World** | Code sandbox (bubblewrap, with a Python audit-hook fallback), a school of programming katas with hidden tests, a gallery, a chat UI and a "brain" visualizer | `gn_world.py`, `chat.html`, `brain.html` |
| **Legacy symbolic layer** | The earliest stdlib-only version (v0.3): idf memory, cloze test, beliefs. Still used for some helpers | `genesis.py`, `genesis_life.py` |

## Quick start

Requires Python 3.10+ and NumPy. Pillow is used for images; OpenCV is optional.

```bash
pip install -r requirements.txt
python3 genesis_web.py            # opens on http://127.0.0.1:8765  (use --port N, --open)
```

The first start is a **birth**: random weights, 3 layers, 0.63 M parameters. It starts learning immediately from a small built-in curriculum and
reads Wikipedia on its own (read-only network access). Over a day or two on a CPU laptop it grows to about 7 layers / 3 M parameters.

Optional, for a proper corpus (~300 MB download of Wikimedia dumps, CC BY-SA; the script is polite and resumable):

```bash
python3 bigcorpus_fetch.py        # writes corpus_big/txt; the agent uses it automatically if present
```

Run as a service on Linux: see `deploy/genesis.service` (+ `genesis-awake.service` to keep a laptop awake).
Useful environment variables: `GEN_THREADS`, `GEN_BATCH`, `GEN_CTX`, `GEN_OPT` (`muon`/`adam`), `GEN_SLEEP`, `GEN_AUTO`, `GEN_HOME` (data folder).

Tests (NumPy only): `python3 -m unittest discover -s tests -v`

## How it is measured

- **Language quality:** bits per byte (bpb) on a held-out chunk of Wikipedia. Per-token loss is *not* comparable across tokenizer changes, bpb is.
- **Every architectural claim has a bench:** `gn_bench.py` trains variants on a fixed corpus under the same CPU-time budget (`bench/`).
- **Evolution:** accept a mutation only with a paired t-test on shared batches and shared evaluation texts, reject early when clearly worse.
- **Self-knowledge:** `selftest.py` runs a battery with chance levels and naive baselines (see below).

### Results so far (single runs on a weak CPU laptop, no GPU; treat as indicative, not as replicated science)

| Change | Result |
|---|---|
| Fast batched attention matmul + GELU | training throughput about 2.9x |
| Muon vs Adam (0.64 M model, 60 s CPU, `bench/`) | 3.33 vs 3.44 bpb; 120 s run: 2.96 vs 3.21 |
| Word-level BPE (4k merges) vs byte-level (12 min bench) | 1.57 vs 1.99 bpb. *The biggest single win in the project, larger than all optimizer changes together.* |
| 4 threads + batch 16 | about +40 % throughput |
| Evolution tournaments | one accepted mutation so far (added layer); several rejected early (e.g. number of heads) |

### Self-awareness test battery (`selftest.py`)

It does **not** test for consciousness (no such test exists). It measures whether what the agent says and predicts about itself agrees with what is
actually happening inside it, each against chance or a naive baseline. First run at ~25 k steps (full table in `docs/selftest_example.md`):

| Test | Result | Chance / baseline |
|---|---|---|
| Intervene on a state dimension, does the matching emotion word become more likely? | 2 of 23, not significant (p=0.26) | 4 % |
| Recognise its own chat replies among the human's | 64 % (optimistic: partly seen in training) | 50 % |
| Predict its own next focus of attention | 80 % | 73 % for "repeat the last one" |
| Calibration: expected reward vs actual reward | correlation -0.21 | 0 |
| Predict how its own actions change its own state | 0 % better than chance | 0 % |
| Remember what the human said (fixed sample) | 63 % | about 0-10 % without replay |

So: memory works, the rest is **not yet better than simple baselines**. The state-to-language link was added after the first training, and the
test is designed to show whether it grows with training. Re-run `selftest.py` over time.

## Honest status

- The model has ~3 M parameters. Its text is word-like with plausible grammar and almost no meaning. It solves 3 of 14 school tasks.
- The internal "emotions" are **functional variables** (they change behaviour and speech and are tracked), not evidence of experience.
  `chat.html` shows a checklist based on the indicator-property approach of Butlin, Long et al., [arXiv:2308.08708](https://arxiv.org/abs/2308.08708),
  with several items marked "no" or "partly" on purpose.
- Large parts are designed by hand (emotion list, action set, reward shaping). What is *learned* is value, taste, policy, language, recall, and the
  mapping from internal state to words. Whether anything interesting *emerges* is exactly what the test battery and ablations are for.
- In prompts and in the built-in curriculum the human interlocutor is called "Теодор" (the author's first name) and the UI is in Russian.

## Safety and privacy

- Generated code runs only in a sandbox (bubblewrap with no network and cgroup limits, or an audit-hook Python fallback), writing only in its own folder.
- Network access is read-only: Wikipedia / Wikimedia Commons. No accounts, no credentials, nothing is posted anywhere.
- All learned state (weights, tokenizer, memory, chats, drawings) lives in the data folder and is excluded from git (`.gitignore`). A fresh clone is a newborn.

## Repository layout

```
gn_net.py        autograd, transformer, Muon/Adam, growth, KV-cache generation
gn_tok.py        growing word-level BPE
gn_senses.py     image/audio VQ perception
gn_mind.py       the agent: policy, value, emotions, taste, evolution, sleep, self-model, state conditioning
gn_world.py      sandbox, school of tasks, files
gn_curriculum.py built-in starter curriculum
genesis_web.py   HTTP server + event stream for the UI;  chat.html, brain.html
genesis.py, genesis_life.py   legacy stdlib layer
gn_bench.py      architecture bench;  bench/ results
selftest.py      self-knowledge test battery
bigcorpus_fetch.py  Wikipedia dump -> text shards
docs/            research notes, example self-test report
tests/           unit tests (function-preserving growth, gradients, cache vs full forward)
```

## Roadmap

- Ablations: with/without state conditioning, with/without evolution, with/without sleep replay, on the same budget.
- A richer environment and feedback signal (so far the main external signal is the human's thumbs up/down).
- Scale the same ideas to a larger model on a GPU to see which parts survive.

## License

MIT. Wikipedia text fetched at run time is CC BY-SA 4.0.
