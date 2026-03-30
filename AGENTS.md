# AI Agent Context & Guidelines (`tfrecords-reader`)

This document serves as persistent memory and instructions for any AI agents or coding assistants operating within the `tfrecords-reader` repository.

## 1. Project Overview
`tfr_reader` is a highly performant Python library designed to read TensorFlow TFRecord files. It provides random access capabilities and streaming support from Google Cloud Storage.
- **Core Value:** It operates *without* a TensorFlow dependency by utilizing a custom Cython-based protobuf decoder, providing significant speedups for reading ML datasets.

## 2. Tech Stack & Environment
- **Languages:** Python (>= 3.11), Cython 3+, C++
- **Package Manager:** `uv` (preferred over standard `pip` for speed)
- **Data Manipulation:** `polars` (used for indexing)
- **Formatting/Linting:** `ruff`
- **Type Checking:** `mypy`
- **Testing:** `pytest`

## 3. Core Development Commands (Makefile)
We rely on a `Makefile` to simplify local development steps.

- **Install/Setup Environment:**
  ```bash
  pip install uv
  uv pip install -e .[dev,datasets]
  ```
- **Compile Cython Code:** (CRITICAL: Must be run whenever `*.pyx` or `*.pxd` files are modified).
  ```bash
  make build-ext
  ```
- **Run Tests:**
  ```bash
  make test
  ```
- **Run Type Checks:**
  ```bash
  make mypy
  ```
- **Compile Protobufs:** (Run if `src/tfr_reader/example/tfr_example.proto` is changed).
  ```bash
  make build-proto
  ```

## 4. Repository Structure
- `src/tfr_reader/`: Main python and cython codebase.
  - `example/`: Protobuf definition files (`.proto`) and their generated cython/python output.
  - `cython/`: Direct Cython implementations for fast decoding.
- `tests/`: Pytest test suite covering reader functionality.
- `notebooks/`: Jupyter notebooks used for demonstrations and experiments (e.g., Keras 3 integrations, MLP training with EMA).

## 5. Agent Instructions & Best Practices
1. **Always recompile Cython (`make build-ext`)** before running tests if you modify any `.pyx` or `.cpp` files. Otherwise, Python will execute stale compiled objects (`*.so`), leading to confusing debugging sessions.
2. **Use Ruff:** Code must adhere to `ruff`'s default formatting (Black-like).
3. **Protobuf Handling:** If a bug relates to missing fields in TFRecords, inspect `tfr_example.proto` and the internal cython decoding rather than falling back to standard Python `google.protobuf`.
4. **Interpreter:** For tools like VS Code or testing tasks (`test_reader.py`), ensure the python interpreter is pointed to an environment where the `tfr_reader` module was installed in editable mode (`uv pip install -e .`), so compiled extensions are correctly resolved.
5. **Pull Requests:** When asked to create a pull request (e.g., via `gh` CLI or workflows), **ALWAYS** use the pull request template found in `.github/PULL_REQUEST_TEMPLATE.md`. Ensure that all required fields, checkboxes, and version bump options are explicitly filled out and included in the PR description.
