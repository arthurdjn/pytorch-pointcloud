.DEFAULT_GOAL := help

SRC:=src
TESTS:=tests
EXAMPLES:=examples
DOCS:=docs
CMD:=uv run --no-sync

# Linting, formatting, etc.

.PHONY: format
format: ## Format source code and tests
	$(CMD) ruff format $(SRC) $(TESTS) $(EXAMPLES)

.PHONY: lint
lint: ## Lint source code and tests
	$(CMD) ruff check $(SRC) $(TESTS) $(EXAMPLES)

.PHONY: lint-fix
lint-fix: ## Lint and fix source code and tests
	$(CMD) ruff check --fix $(SRC) $(TESTS) $(EXAMPLES)

.PHONY: type
type: ## Type in source code and tests
	$(CMD) mypy $(SRC) $(TESTS) $(EXAMPLES)

.PHONY: isort
isort: ## Sort imports using ruff
	$(CMD) ruff check --select I $(SRC) $(TESTS) $(EXAMPLES)

.PHONY: isort-fix
isort-fix: ## Sort imports using ruff
	$(CMD) ruff check --select I --fix $(SRC) $(TESTS) $(EXAMPLES)

.PHONY: all
all: format type lint isort ## Run all formatting commands

# Tests

.PHONY: test
test: ## Run tests
	$(CMD) pytest $(TESTS) -s -vv

.PHONY: doctest
doctest: ## Run doctests in source docstrings (>>> examples)
	$(CMD) pytest --doctest-modules $(SRC)

.PHONY: doctest-docs
doctest-docs: ## Run python code blocks in source docstrings and the docs markdown
	$(CMD) pytest --markdown-docs --markdown-docs-syntax=superfences $(SRC) $(DOCS) \
		--ignore=$(DOCS)/scripts --ignore=$(DOCS)/examples --ignore=$(SRC)/torch_pointcloud/lightning

# Documentation

.PHONY: tables
tables: ## Sync the model catalog CSV from the registry (docs/data/models.csv)
	$(CMD) python docs/scripts/build_model_tables.py
	@# The macro renders the CSV at build time, but zensical's cache keys on the
	@# page source (unchanged), so drop the cache to pick up CSV/metric edits.
	rm -rf .cache

.PHONY: examples
examples: ## Render example notebooks to Markdown (docs/examples/*.md)
	$(CMD) python docs/scripts/build_examples.py

.PHONY: api
api: ## Regenerate the per-module API reference stubs (docs/api/)
	$(CMD) python docs/scripts/build_api_reference.py torch_pointcloud --out docs/api

.PHONY: assets
assets: ## Embed author / license metadata in the committed documentation assets
	$(CMD) python docs/scripts/stamp_asset_metadata.py

.PHONY: papers
papers: ## Render the previews behind the paper() macro (docs/assets/papers/)
	$(CMD) python docs/scripts/build_paper_cards.py
	@# The macro reads the metadata at build time, but zensical's cache keys on the
	@# page source (unchanged), so drop the cache to pick up new previews.
	rm -rf .cache

.PHONY: docs
docs: tables examples api ## Generate documentation
	JUPYTER_PLATFORM_DIRS=1 $(CMD) zensical build --strict

.PHONY: serve
serve: tables examples api ## Serve documentation
	JUPYTER_PLATFORM_DIRS=1 $(CMD) zensical serve

# Docker (isolated CUDA environments, see docker/Dockerfile)

CUDA:=12.6.3
PYTHON:=3.12
TORCH:=2.8.0
EXTRAS:=pyg-lib,spconv,ocnn,lightning
CUDNN_TAG=$(if $(filter 11.%,$(CUDA)),cudnn8,cudnn)
DOCKER_IMAGE=torch-pointcloud:py$(PYTHON)-torch$(TORCH)-cu$(subst .,,$(basename $(CUDA)))
GPU_ARGS:=--gpus all
# AppImage terminals (Cursor) export ARGV0, which zsh applies to argv[0] of spawned commands: make then
# believes it was invoked as cursor.AppImage and $(MAKE) recursion launches the IDE. Pin the real binary.
MAKE:=make

DOCKER_MATRIX:=\
	11.8.0:3.10:2.6.0:pyg-lib,spconv,lightning \
	12.4.1:3.11:2.6.0:pyg-lib,spconv,ocnn,lightning \
	12.6.3:3.12:2.8.0:pyg-lib,spconv,ocnn,lightning,torchsparse,mamba,dwconv \
	12.8.1:3.13:2.8.0:pyg-lib,lightning

.PHONY: docker-build
docker-build: ## Build the Docker image for one CUDA / PYTHON / TORCH / EXTRAS combination
	docker build -f docker/Dockerfile \
		--build-arg CUDA_VERSION=$(CUDA) \
		--build-arg CUDNN_TAG=$(CUDNN_TAG) \
		--build-arg PYTHON_VERSION=$(PYTHON) \
		--build-arg TORCH_VERSION=$(TORCH) \
		--build-arg EXTRAS=$(EXTRAS) \
		-t $(DOCKER_IMAGE) .

.PHONY: docker-test
docker-test: ## Run the test suite inside the Docker image (GPU_ARGS= to run without GPU)
	docker run --rm $(GPU_ARGS) -v $(CURDIR):/workspace -w /workspace $(DOCKER_IMAGE) pytest $(TESTS) -s -vv

.PHONY: docker-shell
docker-shell: ## Open a shell inside the Docker image (GPU_ARGS= to run without GPU)
	docker run --rm -it $(GPU_ARGS) -v $(CURDIR):/workspace -w /workspace $(DOCKER_IMAGE) bash

.PHONY: docker-matrix
docker-matrix: ## Build and run the test suite in every supported CUDA / python / torch combination
	@set -e; for combo in $(DOCKER_MATRIX); do \
		cuda=$${combo%%:*}; rest=$${combo#*:}; \
		python=$${rest%%:*}; rest=$${rest#*:}; \
		torch=$${rest%%:*}; extras=$${rest#*:}; \
		echo ">>> CUDA=$$cuda PYTHON=$$python TORCH=$$torch EXTRAS=$$extras"; \
		$(MAKE) docker-build CUDA=$$cuda PYTHON=$$python TORCH=$$torch EXTRAS=$$extras; \
		$(MAKE) docker-test CUDA=$$cuda PYTHON=$$python TORCH=$$torch EXTRAS=$$extras; \
	done

# Misc

.PHONY: clean
clean: ## Clear local caches and build artifacts
	rm -rf `find . -name __pycache__`
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -f .coverage
	rm -f .coverage.*
	rm -rf *.egg-info
	rm -rf build
	rm -rf dist
	rm -rf *.log
	rm -rf site
	rm -rf .cache

.PHONY: help
help: ## Show available commands
	@awk '/^[a-zA-Z0-9_-]+:.*?## .*$$/ { \
		helpCommand = substr($$0, 1, index($$0, ":")-1); \
		helpMessage = substr($$0, index($$0, "## ") + 3); \
		printf "\033[36m%-15s\033[0m %s\n", helpCommand, helpMessage; \
	}' $(MAKEFILE_LIST)
