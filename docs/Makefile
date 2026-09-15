.PHONY: help install serve build clean

PYTHON ?= python
MKDOCS ?= mkdocs

help:
	@echo "Available commands:"
	@echo "  make install      - Install documentation dependencies"
	@echo "  make serve        - Serve documentation locally"
	@echo "  make build        - Build the static site with strict validation"

install:
	$(PYTHON) -m pip install -r requirements.txt

serve:
	$(MKDOCS) serve -f mkdocs.yml

build:
	$(MKDOCS) build --strict -f mkdocs.yml

clean:
	rm -rf generated/
