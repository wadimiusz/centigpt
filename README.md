# nanogpt

A GPT-style transformer written from scratch for learning and small experiments:
causal multi-head attention with RoPE, a SwiGLU MLP, pre-norm blocks with RMSNorm,
and tied input/output embeddings. Training runs on PyTorch Lightning; data is
tokenised once with a byte-level BPE tokenizer and read from disk as a memmap.

**Why this exists.** I wrote this model, its data pipeline and training loop
from scratch to build intuition for how small language models train
(learning-rate sweeps, data packing, scaling) and to have a model I understand
line by line for interpretability experiments. It is meant to be small and
readable, not to compete with nanoGPT or other libraries. Tests and packaging
were done with help from Claude.

## Use it as a package

```bash
uv add git+https://github.com/wadimiusz/nanogpt
```

```python
import torch
from nanogpt.config import Config
from nanogpt.model import GPT

cfg = Config(d_model=64, n_layers=2, n_heads=2, vocab_size=128, max_seq_len=64)
model = GPT(cfg)
logits = model(torch.randint(0, 128, (8, 32)))  # (batch, seq, vocab)
```

While developing against a local checkout, install it in editable mode so that
changes show up immediately:

```bash
uv add --editable ../nanogpt
```

## Train

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
make install          # create .venv and install dependencies, including dev
uv run nanogpt-train  # train with the defaults from Config (TinyStories)
```

Every field of `Config` is a command-line flag, for example
`uv run nanogpt-train --d-model 128 --n-layers 4 --train-tokens-budget 10000000`.
To train on another Hugging Face dataset, set `--dataset-id`,
`--dataset-field-names` (several fields are joined with a `[SEP]` token),
`--tokenizer-path` and `--tokenized-data-path`. The tokenizer is trained and the
data tokenised on first use, then cached on disk.

## Layout

| Path | Contents |
| --- | --- |
| `src/nanogpt/config.py` | `Config`, a frozen dataclass with all hyperparameters |
| `src/nanogpt/model.py` | `CausalMultiheadAttention`, `SwiGLUMLP`, `Block`, `GPT` |
| `src/nanogpt/tokenizer.py` | `TokenizerWrapper`: load a cached BPE tokenizer or train one |
| `src/nanogpt/data/` | Memmap tokenisation, window dataset, Lightning datamodule |
| `src/nanogpt/lightning_module.py` | Training and validation steps, AdamW with warmup and cosine decay |
| `src/nanogpt/probe.py` | Tools for measuring what trained attention heads do with positions |
| `src/nanogpt/train.py` | Training entry point (`nanogpt-train`) |
| `experiments/` | Research experiments built on the model |
| `tests/` | Tests, one file per module |

## Experiments

See [`experiments/README.md`](experiments/README.md) for the question, the
setup and how to run each script.

`experiments/nerode_geometry.py` trains the model on samples from random
probabilistic automata and asks whether prefixes in the same Myhill–Nerode
class (prefixes with the same distribution over all continuations) end up
close together in the residual stream. `experiments/nerode_continuations.py`
compares prefixes through their common continuations, and
`experiments/where_errors.py` measures where the trained model's predictions
diverge from the automaton. `experiments/demo_data.py` exports the data for an
interactive write-up of these results:
[Where does a transformer keep its state?](https://wadimiusz.me/nerode/)

## Checks

`make` with no arguments lists all targets. The main ones:

| Target | What it does |
| --- | --- |
| `make check` | **Everything:** linter, formatter check, type checker, tests. Does not modify files and does not stop at the first failure |
| `make lint` | `ruff check` |
| `make format-check` | Show the diff the formatter would apply |
| `make typecheck` | `mypy` |
| `make test` | `pytest` |
| `make fix` | **Rewrites files:** ruff autofixes and formatting |
| `make hooks` | Install the pre-commit and pre-push git hooks |

All tool configuration lives in `pyproject.toml`.

**ruff** runs on its default rule set (in 0.16 that is a curated set of 413
rules from 38 families, not just "E,F"), with line length 88. To add rules use
`extend-select`, not `select`: the latter replaces the default set entirely.
One rule is disabled, `C408` (rewrite `list()`/`dict()` calls as literals),
since that is a matter of taste rather than correctness.

**mypy** is deliberately not `strict`. The key setting is
`check_untyped_defs = true`: without it mypy does not look inside functions
without annotations, and so does not check the calls they make.

## Git hooks

`make hooks` installs two hooks:

- **pre-commit**: ruff (`check --fix` and `format`), file hygiene, `uv.lock`
  consistency. Only things that fix themselves or cost almost nothing.
- **pre-push**: mypy and pytest. They give a verdict that has to be fixed by
  hand, which should not block a commit.

The ruff and mypy hooks are `repo: local` with `language: system` and run
through `uv run`, i.e. in the **project** environment. In the isolated venv
that pre-commit creates for remote hook repositories (`mirrors-mypy`) torch is
not installed, so mypy would either fail on `import-not-found` or, with
`ignore_missing_imports`, treat all of torch as `Any`. As a side effect, the
hook tools have exactly the versions pinned in `uv.lock`. The mypy hook runs
with `pass_filenames: false` and checks the whole project: a change in
`model.py` can break types in `tests/`, which a per-file check would miss.

## License

MIT
