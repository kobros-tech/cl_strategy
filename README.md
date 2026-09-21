# skill_memory

Class-level, probe-based Skill Memory plugin for [Avalanche](https://avalanche.continualai.org)
continual learning. See `skill_memory/README.md` for the full design/API doc.

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

## Run the unit tests

```bash
python -m pytest -q
```

## Run the SplitMNIST demo

`skill_memory/demos/demo_splitmnist_ml_er.py` is the repo's one demo, and
what CI runs (see `.github/workflows/demo-splitmnist-ml-er.yml`). Skill
Memory trains the main model as usual; a completely separate ML or CL
evaluator is then trained on frozen per-class evaluation memory and
reports class-level accuracy, loss, and forgetting. The evaluator's
lifetime (`ml`: a fresh one per experience, `cl`: retained across
experiences) is the only difference between the two modes - both are
trained on the full accumulated class memory seen so far, not just the
latest experience. In practice `cl` has given better results than `ml`,
and evaluator accuracy/forgetting depend heavily on giving the evaluator
enough epochs (`--eval-epochs 10` or `20`, not `1`) to actually fit the
accumulated memory:

```bash
python skill_memory/demos/demo_splitmnist_ml_er.py \
    --n-experiences 5 \
    --eval-method cl \
    --eval-memory-per-class 20 \
    --train-epochs 1 \
    --eval-epochs 20 \
    --max-skills 10
```

Pass `--skill-eval-routing oracle` (or `probe`/`both`) to additionally
report Skill Memory's own direct class-oracle/anonymous-routing accuracy,
independent of the auxiliary ML/CL evaluator - this is what the CI
workflow uses. Run `--help` for the full flag list.

This demo is a thin script: all the reusable machinery it drives (frozen
per-class evaluation memory, the independent evaluator's train/evaluate
loop, and direct Skill Memory oracle/probe evaluation) lives in
`skill_memory.evaluation.ml_cl_evaluator`, so it's importable and testable
independent of SplitMNIST or this specific demo - see
`skill_memory/tests/test_ml_cl_evaluator.py`.

## Layout

```
pyproject.toml            # pip-installable package metadata
requirements.txt          # runtime dependencies
requirements-dev.txt      # + pytest, for running the test suite
skill_memory/              # the importable package (`import skill_memory`)
    __init__.py
    cl/                     # continual-learning strategy: the plugin, skill
                            #   registry, and REUSE/SCRATCH decision logic
        skill_memory_plugin.py
        persistent_skill_memory_plugin.py
        decision.py
        skill_registry.py
        training.py
    evaluation/             # anonymous routing, the reverse-engineering
                            #   model used to identify a class at eval time,
                            #   and independent ML/CL evaluation
        fingerprint_routing.py   # public compatibility entry point
        routing.py
        reverse_engineering.py
        behavior.py
        diagnostics.py
        global_fingerprint_refresh.py
        ml_cl_evaluator.py       # independent ML/CL evaluator machinery
    utils/                  # shared, package-independent helpers
        probing.py
        models.py                # SimpleMLP, used by the demo and tests
    demos/                  # runnable scripts (not pytest tests)
        demo_splitmnist_ml_er.py
    README.md
    tests/
        test_*.py           # unit tests (pytest)
```
