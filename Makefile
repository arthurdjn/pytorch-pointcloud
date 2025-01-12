.DEFAULT_GOAL := help

SRC:=src
TESTS:=tests
CMD:=

# Linting, formatting, etc.

.PHONY: format
format: ## Format source code and tests
	$(CMD) ruff format $(SRC) $(TESTS)

.PHONY: lint
lint: ## Lint source code and tests
	$(CMD) ruff check $(SRC) $(TESTS)

.PHONY: lint-fix
lint-fix: ## Lint and fix source code and tests
	$(CMD) ruff check --fix $(SRC) $(TESTS)

.PHONY: type
type: ## Type in source code and tests
	$(CMD) mypy $(SRC) $(TESTS)

.PHONY: isort
isort: ## Sort imports using ruff
	$(CMD) ruff check --select I

.PHONY: isort-fix
isort-fix: ## Sort imports using ruff
	$(CMD) ruff check --select I --fix

.PHONY: all
all: format type lint isort ## Run all formatting commands

# Tests

.PHONY: test
test: ## Run tests
	$(CMD) pytest $(TESTS) -s -vv

# Utils

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

.PHONY: help
help: ## Show available commands
	@awk '/^[a-zA-Z0-9_-]+:.*?## .*$$/ { \
		helpCommand = substr($$0, 1, index($$0, ":")-1); \
		helpMessage = substr($$0, index($$0, "## ") + 3); \
		printf "\033[36m%-15s\033[0m %s\n", helpCommand, helpMessage; \
	}' $(MAKEFILE_LIST)
