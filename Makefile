.PHONY: clean clean-pyc clean-build clean-test help train serve test reports

help:
	@echo "clean - remove all build, test, coverage, and Python artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-build - remove build artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "train - train the model"
	@echo "serve - run the API service"
	@echo "test - run tests"
	@echo "reports - generate reports"

clean: clean-pyc clean-build clean-test
	@echo "Cleaned all artifacts"

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '.DS_Store' -exec rm -f {} +

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info/

clean-test:
	rm -fr .tox/
	rm -fr .pytest_cache/
	rm -f .coverage
	rm -fr htmlcov/

train:
	python scripts/train.py

serve:
	uvicorn scripts.predict:app --host 0.0.0.0 --port 8000

test:
	pytest

reports:
	python scripts/metrics_report.py

gifs:
	python scripts/gifs_gen.py