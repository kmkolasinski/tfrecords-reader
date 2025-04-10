help: ## Print this message and exit
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z%_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2 | "sort"}' $(MAKEFILE_LIST)

PROTO_DIR=src/tfr_reader/example/

install: ## Install the package for development
	pip install uv
	uv pip install -e .[dev]

test:  ## Run unit tests
	pytest tests/

build-proto: ## Generate Python code from proto files
	protoc --proto_path=${PROTO_DIR} --python_out=${PROTO_DIR} ${PROTO_DIR}/example.proto

build-cython: ## Build Cython files with debug info and annotated HTML files
	cythonize -a -i src/tfr_reader/cython/indexer.pyx --force
	cythonize -a -i src/tfr_reader/cython/decoder.pyx --force

clean:  ## Clean up build artifacts
	rm -r build/  2> /dev/null || true
	rm -r src/build/  2> /dev/null || true
	rm -r src/tfr_reader.egg-info/  2> /dev/null || true
	rm -r dist/ 2> /dev/null || true
	rm  src/tfr_reader/cython/*.so
	rm  src/tfr_reader/cython/*.cpp
	rm  src/tfr_reader/cython/*.html

precommit: ## Run precommits without actually commiting
	SKIP=no-commit-to-branch pre-commit run --all-files
