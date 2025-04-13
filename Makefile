.PHONY: setup test lint format run-api run-frontend run-cli clean

# Setup the development environment
setup:
	uv venv
	uv pip install -e .

# Run tests
test:
	pytest

# Lint the code
lint:
	flake8 src tests

# Format the code
format:
	black src tests

# Run the API
run-api:
	python -m src.main api

# Run the frontend
run-frontend:
	python -m src.main frontend

# Run the CLI
run-cli:
	python -m src.main cli

# Clean up temporary files
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
