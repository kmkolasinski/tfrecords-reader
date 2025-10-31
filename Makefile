help: ## Print this message and exit
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z%_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2 | "sort"}' $(MAKEFILE_LIST)

PROTO_DIR=src/tfr_reader/example/


install: ## Install the package for development
	pip install uv
	uv pip install -e .[dev]


test:  ## Run unit tests
	pytest tests/


mypy: ## Run mypy type checks
	mypy --python-version=3.11 --config-file=pyproject.toml


build-proto: ## Generate Python code from proto files
	protoc --proto_path=${PROTO_DIR} --python_out=${PROTO_DIR} ${PROTO_DIR}/tfr_example.proto

build-cython: ## Build Cython files with debug info and annotated HTML files
	cythonize -a -i src/tfr_reader/cython/indexer.pyx --force
	cythonize -a -i src/tfr_reader/cython/decoder.pyx --force

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

.ONESHELL:
release: ## Create a new tag for release.
	@echo "WARNING: This operation will create s version tag and push to github"
	@echo "         Update version in pyproject.toml before running this command"
	@echo "         and provide same tag value here."
	@read -p "Version? (provide the next x.y.z semver) : " TAG
	@git add .
	@git commit -m "release: version $${TAG} 🚀" --no-verify
	echo "creating git tag : $${TAG}"
	@git tag $${TAG}
	@git push -u origin HEAD --tags
	@echo "Github Actions will detect the new tag and release the new version."
