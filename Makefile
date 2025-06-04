# Build, package, test, and clean
PROJECT=choclo
CHECK_STYLE=src/$(PROJECT) doc test
GITHUB_ACTIONS=.github/workflows

.PHONY: build install test test-coverage test-numba format check check-format check_style check-actions clean

help:
	@echo "Commands:"
	@echo ""
	@echo "  install   install in editable mode"
	@echo "  test      run the test suite (including doctests) and report coverage"
	@echo "  format    automatically format the code"
	@echo "  check     run code style and quality checks"
	@echo "  build     build source and wheel distributions"
	@echo "  clean     clean up build and generated files"
	@echo ""

build:
	python -m build .

install:
	python -m pip install --no-deps --editable .

test: test-numba test-coverage

test-coverage:
	NUMBA_DISABLE_JIT=1 pytest --cov-report=term-missing --cov --doctest-modules --verbose test src/$(PROJECT)

test-numba:
	NUMBA_DISABLE_JIT=0 pytest --cov-report=term-missing --cov --doctest-modules --verbose test src/$(PROJECT)

format:
	ruff check --select I --fix $(CHECK_STYLE) # fix isort errors
	ruff format $(CHECK_STYLE)
	burocrata --extension=py $(CHECK_STYLE)

check: check-format check-style check-actions

check-format:
	ruff format --check $(CHECK_STYLE)
	burocrata --check --extension=py $(CHECK_STYLE)

check-style:
	ruff check $(CHECK_STYLE)

check-actions:
	zizmor $(GITHUB_ACTIONS)

clean:
	find . -name "*.pyc" -exec rm -v "{}" \;
	find . -name "*.orig" -exec rm -v "{}" \;
	find . -name ".coverage.*" -exec rm -v "{}" \;
	find . -name "_version.py" -exec rm -v "{}" \;
	find . -name "*.egg-info" -type d -exec rm -vr "{}" \; -prune
	find . -name "__pycache__" -type d -exec rm -vr "{}" \; -prune
	rm -rvf build dist MANIFEST .coverage .cache .pytest_cache
