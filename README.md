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

`skill_memory/demos/demo_splitmnist_eval_log.py` writes its eval-log CSV
under a gitignored `skill_memory/demos/logs/` directory, so it won't trip
the large-file hook or get committed by accident.

## Run the unit tests

```bash
python -m pytest -q
```

## Run the SplitMNIST demos

`skill_memory/demos/demo_splitmnist_ml_er.py` is the demo CI actually runs
(see `.github/workflows/demo-splitmnist-ml-er.yml`). Skill Memory trains
the main model as usual; a completely separate ML or CL evaluator is then
trained on frozen per-class evaluation memory and reports class-level
accuracy, loss, and forgetting. The evaluator's lifetime (`ml`: a fresh one
per experience, `cl`: retained across experiences) is the only difference
between the two modes - both are trained on the full accumulated class
memory seen so far, not just the latest experience. In practice `cl` has
given better results than `ml`, and evaluator accuracy/forgetting depend
heavily on giving the evaluator enough epochs (`--eval-epochs 10` or `20`,
not `1`) to actually fit the accumulated memory:

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

`skill_memory/demos/demo_splitmnist_eval_log.py` is a separate, simpler
demo that trains `SkillMemoryPlugin` directly and prints a per-sample
predicted-y-vs-real-y eval log after every training step, plus dumps the
full log to `skill_memory_eval_log.csv`.

```bash
python skill_memory/demos/demo_splitmnist_eval_log.py
```

`demo_splitmnist_oracle_retention.py` and
`demo_splitmnist_weight_reverse_engineering.py` are older, more
narrowly-scoped demos kept for manual/local use; they are not wired into
any CI workflow.

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
    evaluation/             # anonymous routing and the reverse-engineering
                            #   model used to identify a class at eval time
        fingerprint_routing.py   # public compatibility entry point
        routing.py
        reverse_engineering.py
        behavior.py
        diagnostics.py
        global_fingerprint_refresh.py
    utils/                  # shared, package-independent helpers
        probing.py
    demos/                  # runnable scripts (not pytest tests)
        demo_splitmnist_eval_log.py
        demo_splitmnist_ml_er.py
        demo_splitmnist_oracle_retention.py
        demo_splitmnist_weight_reverse_engineering.py
    README.md
    tests/
        test_*.py           # unit tests (pytest)
```
