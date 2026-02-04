.PHONY: install test coverage coverage-check coverage-baseline-update lint type-check format format-check clean clean-cache help

# Install dependencies
install:
	@echo "Checking Python version..."
	@python3 --version || (echo "ERROR: Python 3 is not installed" && exit 1)
	@python3 -c "import sys; major, minor = sys.version_info[:2]; sys.exit(0 if (major == 3 and minor >= 12) or major > 3 else 1)" || \
		(echo "ERROR: Python 3.12+ is required." && \
		 echo "Please install Python 3.12+ manually:" && \
		 echo "  brew install python@3.12" && \
		 echo "Or visit: https://www.python.org/downloads/" && exit 1)
	@echo "Python version check passed."
	@echo "Upgrading pip..."
	@python3 -m pip install --upgrade pip --no-cache-dir
	@echo "Installing dependencies from requirements.txt..."
	@pip3 install --no-cache-dir -r requirements.txt

# Run all tests
test:
	@echo "Running tests..."
	@if command -v pytest > /dev/null 2>&1 || python3 -m pytest --version > /dev/null 2>&1; then \
		CURRENT_DIR=$$(pwd); \
		if [ -z "$$PYTHONPATH" ]; then \
			PYTHONPATH_VAR="$$CURRENT_DIR"; \
		else \
			PYTHONPATH_VAR="$$PYTHONPATH:$$CURRENT_DIR"; \
		fi; \
		PYTHONPATH="$$PYTHONPATH_VAR" python3 -m pytest tests/ -v --tb=short; \
	else \
		echo "pytest not found, running tests directly..."; \
		find tests -name "test_*.py" -type f | while read test_file; do \
			echo "Running $$test_file..."; \
			python3 "$$test_file" || exit 1; \
		done; \
	fi

# Run tests with coverage report
coverage:
	@echo "Running tests with coverage..."
	@PYTHONPATH_VAR="$${PYTHONPATH:-}:$$(pwd)"; \
	export PYTHONPATH="$$PYTHONPATH_VAR"; \
	if command -v pytest > /dev/null 2>&1 || python3 -m pytest --version > /dev/null 2>&1; then \
		if PYTHONPATH="$$PYTHONPATH_VAR" python3 -m pytest --collect-only --quiet > /dev/null 2>&1 && python3 -c "import pytest_cov" 2>/dev/null; then \
			PYTHONPATH="$$PYTHONPATH_VAR" python3 -m pytest tests/ --cov=src --cov-report=term-missing --no-cov-on-fail -v --tb=short; \
		elif python3 -c "import coverage" 2>/dev/null; then \
			PYTHONPATH="$$PYTHONPATH_VAR" python3 -m coverage run -m pytest tests/ -v --tb=short; \
			echo ""; \
			echo "==========================================="; \
			echo "COVERAGE REPORT"; \
			echo "==========================================="; \
			python3 -m coverage report --include="src/*" --show-missing; \
		else \
			echo "Warning: pytest-cov or coverage not installed. Installing pytest-cov..."; \
			pip3 install pytest-cov --quiet; \
			PYTHONPATH="$$PYTHONPATH_VAR" python3 -m pytest tests/ --cov=src --cov-report=term-missing --no-cov-on-fail -v --tb=short; \
		fi; \
	else \
		echo "Error: pytest is required for coverage. Please install pytest first."; \
		exit 1; \
	fi

# Check coverage and fail if below baseline (for CI)
# Compares current coverage with baseline stored in .coverage.baseline
coverage-check:
	@echo "Running coverage check against baseline..."
	@if command -v pytest > /dev/null 2>&1 || python3 -m pytest --version > /dev/null 2>&1; then \
		if ! python3 -c "import pytest_cov" 2>/dev/null; then \
			pip3 install pytest-cov --quiet; \
		fi; \
		CURRENT_DIR=$$(pwd); \
		if [ -z "$$PYTHONPATH" ]; then \
			PYTHONPATH_VAR="$$CURRENT_DIR"; \
		else \
			PYTHONPATH_VAR="$$PYTHONPATH:$$CURRENT_DIR"; \
		fi; \
		export PYTHONPATH="$$PYTHONPATH_VAR"; \
		if [ ! -f .coverage.baseline ]; then \
			echo "Warning: No baseline file found. Creating baseline from current coverage..."; \
			PYTHONPATH="$$PYTHONPATH_VAR" python3 -m coverage run -m pytest tests/ -q > /dev/null 2>&1; \
			python3 -m coverage report --include="src/*" --format=total 2>&1 | tail -1 > .coverage.baseline; \
			BASELINE=$$(cat .coverage.baseline); \
			echo "Baseline set to: $$BASELINE%"; \
		fi; \
		BASELINE=$$(cat .coverage.baseline | tr -d ' '); \
		echo "Baseline coverage: $$BASELINE%"; \
		if ! PYTHONPATH="$$PYTHONPATH_VAR" python3 -m pytest tests/ --cov=src --cov-report=term-missing --no-cov-on-fail -v --tb=short; then \
			echo ""; \
			echo "ERROR: Tests failed. Cannot check coverage."; \
			exit 1; \
		fi; \
		CURRENT=$$(python3 -m coverage report --include="src/*" --format=total 2>&1 | tail -1 | tr -d ' '); \
		echo ""; \
		echo "==========================================="; \
		echo "COVERAGE COMPARISON"; \
		echo "==========================================="; \
		echo "Current coverage: $$CURRENT%"; \
		echo "Baseline coverage: $$BASELINE%"; \
		if [ -z "$$CURRENT" ]; then \
			echo "Error: Could not determine current coverage"; \
			exit 1; \
		fi; \
		if [ -z "$$BASELINE" ]; then \
			echo "Error: Could not read baseline coverage"; \
			exit 1; \
		fi; \
		RESULT=$$(python3 -c "import sys; current=float('$$CURRENT'); baseline=float('$$BASELINE'); sys.exit(0 if current >= baseline else 1)" 2>/dev/null; echo $$?); \
		if [ "$$RESULT" = "1" ]; then \
			echo ""; \
			echo "ERROR: Coverage regression detected!"; \
			echo "Current: $$CURRENT% < Baseline: $$BASELINE%"; \
			DIFF=$$(python3 -c "current=float('$$CURRENT'); baseline=float('$$BASELINE'); print('{:.2f}'.format(baseline - current))" 2>/dev/null || echo "unknown"); \
			echo "Coverage dropped by $$DIFF%"; \
			exit 1; \
		else \
			echo ""; \
			echo "Coverage check passed ($$CURRENT% >= $$BASELINE%)"; \
			IMPROVED=$$(python3 -c "import sys; current=float('$$CURRENT'); baseline=float('$$BASELINE'); sys.exit(0 if current > baseline else 1)" 2>/dev/null; echo $$?); \
			if [ "$$IMPROVED" = "0" ]; then \
				DIFF=$$(python3 -c "current=float('$$CURRENT'); baseline=float('$$BASELINE'); print('{:.2f}'.format(current - baseline))" 2>/dev/null || echo "unknown"); \
				echo "Coverage improved by $$DIFF%"; \
			fi; \
		fi; \
	else \
		echo "Error: pytest is required for coverage. Please install pytest first."; \
		exit 1; \
	fi

# Update coverage baseline with current coverage
coverage-baseline-update:
	@echo "Updating coverage baseline..."
	@if command -v pytest > /dev/null 2>&1 || python3 -m pytest --version > /dev/null 2>&1; then \
		if ! python3 -c "import coverage" 2>/dev/null; then \
			pip3 install coverage --quiet; \
		fi; \
		CURRENT_DIR=$$(pwd); \
		if [ -z "$$PYTHONPATH" ]; then \
			PYTHONPATH_VAR="$$CURRENT_DIR"; \
		else \
			PYTHONPATH_VAR="$$PYTHONPATH:$$CURRENT_DIR"; \
		fi; \
		PYTHONPATH="$$PYTHONPATH_VAR" python3 -m coverage run -m pytest tests/ -q > /dev/null 2>&1; \
			python3 -m coverage report --include="src/*" --format=total 2>&1 | tail -1 > .coverage.baseline; \
		BASELINE=$$(cat .coverage.baseline); \
		echo "Baseline updated to: $$BASELINE%"; \
	else \
		echo "Error: pytest is required for coverage. Please install pytest first."; \
		exit 1; \
	fi

# Lint code
lint:
	@echo "Running pylint..."
	@if ! python3 -c "import pylint" 2>/dev/null; then \
		echo "pylint not found, installing..."; \
		pip3 install pylint --quiet; \
	fi
	@CURRENT_DIR=$$(pwd); \
	if [ -z "$$PYTHONPATH" ]; then \
		PYTHONPATH_VAR="$$CURRENT_DIR"; \
	else \
		PYTHONPATH_VAR="$$PYTHONPATH:$$CURRENT_DIR"; \
	fi; \
	PYTHONPATH="$$PYTHONPATH_VAR" python3 -m pylint --rcfile=.pylintrc src/ tests/ --recursive=y --errors-only || exit 1

# Type check code
type-check:
	@echo "Running mypy type checker..."
	@if ! python3 -c "import mypy" 2>/dev/null; then \
		echo "mypy not found, installing..."; \
		pip3 install mypy --quiet; \
	fi
	@python3 -m mypy src/ --config-file=mypy.ini || exit 1

# Format code
format:
	@echo "Formatting code with black..."
	@if ! python3 -c "import black" 2>/dev/null; then \
		echo "black not found, installing..."; \
		pip3 install black --quiet; \
	fi
	@python3 -m black src/ tests/

# Check code formatting (without modifying files)
format-check:
	@echo "Checking code formatting with black..."
	@if ! python3 -c "import black" 2>/dev/null; then \
		echo "black not found, installing..."; \
		pip3 install black --quiet; \
	fi
	@python3 -m black --check src/ tests/

# Clean Python cache and build artifacts
clean-cache:
	@echo "Cleaning Python cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type f -name "*.coverage" -delete 2>/dev/null || true
	@find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".hypothesis" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "catboost_info" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .coverage 2>/dev/null || true
	@rm -rf .pytest_cache 2>/dev/null || true
	@echo "Python cache cleaned successfully!"

# Clean all generated files and cache
clean:
	@echo "Cleaning all generated files..."
	@$(MAKE) clean-cache
	@rm -rf build/ dist/ 2>/dev/null || true
	@rm -rf results/ 2>/dev/null || true
	@rm -rf *.egg 2>/dev/null || true
	@echo "All generated files cleaned successfully!"

# Show help
help:
	@echo "Available commands:"
	@echo "  make install          - Install dependencies from requirements.txt"
	@echo "  make test             - Run all tests"
	@echo "  make coverage         - Run tests with coverage report (terminal only)"
	@echo "  make coverage-check   - Run coverage check against baseline (for CI)"
	@echo "  make coverage-baseline-update - Update coverage baseline with current coverage"
	@echo "  make lint             - Lint code with pylint"
	@echo "  make type-check       - Type check code with mypy"
	@echo "  make format           - Format code with black"
	@echo "  make format-check     - Check code formatting with black (CI)"
	@echo "  make clean-cache      - Clean Python cache files (.pyc, __pycache__, etc.)"
	@echo "  make clean            - Clean all generated files and cache"
	@echo "  make help             - Show this help message"
