# Repository Guidelines

## Project Structure & Module Organization
- `src/`: core C++ implementation (`common/`, `index/`, `cluster/`, `io/`), including optional GPU/CUVS and DiskANN paths.
- `include/knowhere/`: public headers consumed by downstream projects.
- `tests/ut/`: Catch2-based C++ unit tests (binary: `knowhere_tests`).
- `tests/python/`: pytest-based integration tests for the Python package.
- `python/`: SWIG bindings and wheel packaging (`setup.py`, `build_portable_wheel.sh`).
- `benchmark/`, `cmake/`, `scripts/`, `docs/`: performance tools, build modules, helper scripts, and design docs.

## Build, Test, and Development Commands
- `./build.sh --with-ut --with-diskann`: recommended local build entrypoint (Conan + CMake).
- `./build.sh --with-ut --debug`: Debug build for development.
- `cd build && ./Release/tests/ut/knowhere_tests`: run C++ unit tests.
- `pytest tests/python/test_index_with_random.py`: run a representative Python test.
- `pre-commit run --all-files`: run formatting and sanity hooks before pushing.
- `cd python && ./build_portable_wheel.sh -c -v`: build a portable Python wheel.

## Coding Style & Naming Conventions
- C++ standard is C++17; use 4-space indentation and keep lines near 120 chars (see `.clang-format`).
- Format C/C++/CUDA with `clang-format` (configured via pre-commit).
- Prefer `snake_case` for files/functions/variables (for example, `test_pageann.cc`, `index_factory.cc`); use `PascalCase` for types.
- Keep new headers under `include/knowhere/` when they are public API.

## Testing Guidelines
- C++ tests use Catch2 and live in `tests/ut/test_*.cc`.
- Python tests use pytest and live in `tests/python/test_*.py`.
- Enable relevant build flags for feature tests (for example `--with-diskann`, `--with-pageann`).
- For coverage-focused runs, build with coverage flags and use `scripts/run_codecov.sh`.

## Commit & Pull Request Guidelines
- Follow the observed conventional style: `feat: ...`, `fix: ...`, `test: ...`, `docs: ...`, `build: ...`, `chore: ...`.
- Keep commits scoped to one logical change; include tests for behavior changes.
- PRs should include: concise problem/solution summary, impacted modules, test evidence (commands + results), and linked issue/ID when available.
- Run pre-commit and relevant unit/integration tests before requesting review.
