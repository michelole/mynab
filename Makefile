.PHONY: help install install-dev run format lint type-check test check clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

install-dev: ## Install dependencies with development extras
	uv sync --group dev

run: ## Run the Streamlit dashboard
	uv run streamlit run mynab/app.py

format: ## Format code with ruff
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Run linting with ruff
	uv run ruff check .

type-check: ## Run type checking with ty
	uv run ty check --error-on-warning

test: ## Run tests
	uv run pytest

check: format lint type-check test ## Run all code quality checks

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .ruff_cache
