.PHONY: help install install-dev run format lint type-check test clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	uv sync

install-dev: ## Install dependencies with development extras
	uv sync --extra dev

run: ## Run the Streamlit dashboard
	uv run streamlit run app.py

format: ## Format code with black and isort
	uv run black .
	uv run isort .

lint: ## Run linting with flake8
	uv run flake8 .

type-check: ## Run type checking with mypy
	uv run mypy .

test: ## Run tests
	uv run pytest

check: format lint type-check test ## Run all code quality checks

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .mypy_cache 