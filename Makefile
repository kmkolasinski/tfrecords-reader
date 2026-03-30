help: ## Print this message and exit
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z%_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2 | "sort"}' $(MAKEFILE_LIST)

PROTO_DIR=src/tfr_reader/example/


install: ## Install the package for development
	pip install uv
	uv pip install -e .[dev,datasets]


test:  ## Run unit tests
	pytest tests/


mypy: ## Run mypy type checks
	mypy --python-version=3.11 --config-file=pyproject.toml


build-proto: ## Generate Python code from proto files
	protoc --proto_path=${PROTO_DIR} --python_out=${PROTO_DIR} ${PROTO_DIR}/tfr_example.proto


build-ext: ## Build Cython extensions in place
	python setup.py build_ext --inplace


clean:  ## Clean up build artifacts
	rm -rf build/ dist/ 2> /dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2> /dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} + 2> /dev/null || true
	find src/ -type f -name "*.so" -delete 2> /dev/null || true
	find src/ -type f -name "*.cpp" -delete 2> /dev/null || true
	find src/ -type f -name "*.c" -delete 2> /dev/null || true
	find src/ -type f -name "*.html" -delete 2> /dev/null || true

precommit: ## Run precommits without actually commiting
	SKIP=no-commit-to-branch pre-commit run --all-files
