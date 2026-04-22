.DEFAULT_GOAL := help

SRC:=src
TESTS:=tests
EXAMPLES:=examples
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

# Documentation

.PHONY: docs
docs: ## Generate documentation
	JUPYTER_PLATFORM_DIRS=1 $(CMD) mkdocs build

.PHONY: serve
serve: ## Serve documentation
	JUPYTER_PLATFORM_DIRS=1 $(CMD) mkdocs serve

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
