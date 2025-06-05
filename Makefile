# Build, package, test, and clean
PROJECT=choclo
TESTDIR=tmp-test-dir-with-unique-name
PYTEST_ARGS=--cov-report=term-missing --cov=$(PROJECT) --doctest-modules -v --pyargs
NUMBATEST_ARGS=--doctest-modules -v --pyargs
CHECK_STYLE=$(PROJECT) doc
GITHUB_ACTIONS=.github/workflows

.PHONY: build install test test_coverage test_numba format check check-format check_style check-actions clean

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
	python -m pip install --no-deps -e .

test: test_coverage test_numba

test_coverage:
	# Run a tmp folder to make sure the tests are run on the installed version
	mkdir -p $(TESTDIR)
	cd $(TESTDIR); NUMBA_DISABLE_JIT=1 pytest $(PYTEST_ARGS) $(PROJECT)
	cp $(TESTDIR)/.coverage* .
	rm -rvf $(TESTDIR)

test_numba:
	# Run a tmp folder to make sure the tests are run on the installed version
	mkdir -p $(TESTDIR)
	cd $(TESTDIR); NUMBA_DISABLE_JIT=0 pytest $(NUMBATEST_ARGS) $(PROJECT)
	rm -rvf $(TESTDIR)

format:
	ruff check --select I --fix $(CHECK_STYLE) # fix isort errors
	ruff format $(CHECK_STYLE)
	burocrata --extension=py $(CHECK_STYLE)

check: check-format check-style check-actions check-docstrings

check-format:
	ruff format --check $(CHECK_STYLE)
	burocrata --check --extension=py $(CHECK_STYLE)

check-style:
	ruff check $(CHECK_STYLE)

check-actions:
	zizmor $(GITHUB_ACTIONS)

check-docstrings:
	numpydoc lint $(wildcard ${PROJECT}/**/*.py)

clean:
	find . -name "*.pyc" -exec rm -v {} \;
	find . -name "*.orig" -exec rm -v {} \;
	find . -name ".coverage.*" -exec rm -v {} \;
	rm -rvf build dist MANIFEST *.egg-info __pycache__ .coverage .cache .pytest_cache $(PROJECT)/_version.py
	rm -rvf $(TESTDIR) dask-worker-space
