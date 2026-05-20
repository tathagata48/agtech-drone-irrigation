# Contributing Guide

Thank you for considering a contribution — whether it's a bug fix, new sensor
profile, documentation improvement, or new feature.

---

## Development setup

```bash
# 1. Fork & clone
git clone https://github.com/your-org/agtech-drone-irrigation.git
cd agtech-drone-irrigation

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install in editable mode with dev extras
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pre-commit install
```

---

## Workflow

```
main ← develop ← feat/your-feature
```

1. Branch from `develop`: `git checkout -b feat/my-feature develop`
2. Make your changes; add or update tests.
3. Run the test suite locally: `pytest tests/ -v`
4. Run the linter: `ruff check src/ tests/` and formatter: `black src/ tests/`
5. Push and open a Pull Request **against `develop`**.

CI must pass (lint + all tests on Python 3.9–3.12 + smoke test) before merge.

---

## Code style

| Tool   | Config           | Run                  |
|--------|------------------|----------------------|
| Black  | `pyproject.toml` | `black src/ tests/`  |
| Ruff   | `pyproject.toml` | `ruff check src/ tests/` |
| mypy   | `pyproject.toml` | `mypy src/`          |

Line length: **100** characters.

---

## Writing tests

- All new public functions must have at least one happy-path and one
  edge-case test.
- Tests live in `tests/test_<module>.py`.
- Use `conftest.py` fixtures for shared synthetic data — don't generate new
  500×500 images inside individual tests unless necessary.
- Coverage threshold is **80%** (`pyproject.toml: fail_under = 80`).

---

## Adding a new sensor profile

1. Create `configs/<your_sensor>.yaml` extending `default.yaml`.
2. Document the calibration procedure in `docs/calibration_guide.md`.
3. If the sensor uses a non-standard file format (e.g. SEQ, R-JPEG),
   add a loader branch in `src/data/loaders.py` and test it.

---

## Commit message convention

```
type(scope): short description

Types: feat, fix, docs, test, refactor, ci, chore
Example: feat(segmentation): add adaptive HSV calibration from ROI
```

---

## Questions?

Open a [GitHub Discussion](https://github.com/your-org/agtech-drone-irrigation/discussions)
or tag `@maintainers` in your PR.
