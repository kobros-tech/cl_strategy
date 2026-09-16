# skill_memory

Class-level, probe-based Skill Memory plugin for [Avalanche](https://avalanche.continualai.org)
continual learning. See `skill_memory/README.md` for the full design/API
doc and `skill_memory/README_PERSISTENT_FINGERPRINTS.md` for the optional
persistent-fingerprint extension.

## Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# editable install of this package + its runtime deps (torch, numpy, avalanche-lib)
pip install -e .

# only needed if you want to run the existing unit tests too
pip install -r requirements-dev.txt
```

If you want a specific CUDA build of torch (rather than the CPU wheel pip
picks by default), install that first per
https://pytorch.org/get-started/locally/, then run `pip install -e .`.

## Lint / format

```bash
ruff check .
ruff format .
```

Config lives in `[tool.ruff]` in `pyproject.toml`. `pre-commit` (via
`.pre-commit-config.yaml`) runs the same `ruff check`/`ruff format`, plus
standard hygiene hooks (merge-conflict markers, trailing whitespace,
end-of-file, YAML syntax, and a 500 KB cap on newly added files):

```bash
pre-commit install     # one-time, sets up the git hook
pre-commit run -a      # run all hooks against the whole repo
```

`skill_memory/tests/demo_splitmnist_eval_log.py` writes its eval-log CSV
under a gitignored `skill_memory/tests/logs/` directory, so it won't trip
the large-file hook or get committed by accident.

## Run the unit tests

```bash
python -m pytest -q
```

## Run the SplitMNIST demo / eval-log script

`skill_memory/tests/demo_splitmnist_eval_log.py` is a runnable demo
(not a pytest unit test) that trains `SkillMemoryPlugin` on Avalanche's
SplitMNIST benchmark and prints a per-sample predicted-y-vs-real-y eval log
after every training step, plus dumps the full log to
`skill_memory_eval_log.csv`.

```bash
python skill_memory/tests/demo_splitmnist_eval_log.py
```

## Layout

```
pyproject.toml            # pip-installable package metadata
requirements.txt          # runtime dependencies
requirements-dev.txt      # + pytest, for running the test suite
skill_memory/             # the importable package (`import skill_memory`)
    __init__.py
    behavior.py
    decision.py
    probing.py
    skill_memory_plugin.py
    persistent_skill_memory_plugin.py
    skill_registry.py
    training.py
    README.md
    README_PERSISTENT_FINGERPRINTS.md
    tests/
        test_*.py                          # existing unit tests (pytest)
        demo_splitmnist_eval_log.py     # SplitMNIST demo / eval log script
```
