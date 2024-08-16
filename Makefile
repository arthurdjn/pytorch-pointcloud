# clean:
# 	rm -rf build/
# 	rm -rf dist/
# 	rm -rf *.egg-info/

u:
	pip uninstall torch_pointcloud

i:
	# python setup.py install
	pip install -v -e .

.PHONY: clean
clean: ## Clear local caches and build artifacts
	rm -rf `find . -name __pycache__`
	rm -f `find . -type f -name '*.py[co]'`
	rm -f `find . -type f -name '*~'`
	rm -f `find . -type f -name '.*~'`
	rm -rf .cache
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -f .coverage
	rm -f .coverage.*
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info

.PHONY: help
help: ## Show available commands
	@echo "Available targets:"
	@awk '/^[a-zA-Z0-9_-]+:.*?## .*$$/ { \
		helpCommand = substr($$0, 1, index($$0, ":")-1); \
		helpMessage = substr($$0, index($$0, "## ") + 3); \
		printf "\033[36m%-20s\033[0m %s\n", helpCommand, helpMessage; \
	}' $(MAKEFILE_LIST)
