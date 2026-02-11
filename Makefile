.PHONY: help devenv init_codespace docs lint package test lab

help:
	@echo "The following make targets are available:"
	@echo "	 devenv			sync deps and install pre-commit hooks"
	@echo "	 init_codespace		initialize codespace with claude code and deps"
	@echo "	 docs			generate sphinx documentation"
	@echo "	 lint			run pre-commit on all files"
	@echo "	 package		package for pypi"
	@echo "	 test			run all tests"
	@echo "	 lab			start jupyter lab server"

devenv:
	uv sync --quiet --dev --all-extras $(if $(upgrade),--upgrade,--frozen)
	uv pip freeze
	uv run --no-sync pre-commit install

init_codespace:
	curl -fsSL https://claude.ai/install.sh | bash
	git pull || true
	uv sync --quiet --dev --all-extras --frozen

docs:
	rm -rf docs/api
	rm -rf docs/build
	uv run sphinx-apidoc --no-toc -f -t=docs/_templates -o docs/api triad/
	uv run sphinx-build -b html docs/ docs/build/

lint:
	uv run pre-commit run --all-files

package:
	rm -rf dist/*
	python3 setup.py sdist
	python3 setup.py bdist_wheel

test:
	uv run pytest tests/

lab:
	mkdir -p tmp
	uv run jupyter lab --port=8888 --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*'

release_branch:
	uv pip install -e .
	$(eval VERSION := $(shell uv pip show triad | grep "^Version:" | cut -d' ' -f2))
	@if echo "$(VERSION)" | grep -q "dev"; then \
		git tag v$(VERSION); \
		git push origin v$(VERSION); \
	else \
		echo "Error: Can only release dev versions (current: $(VERSION))"; \
		exit 1; \
	fi
