.PHONY: help install serve build deploy-dev deploy-version clean

PYTHON ?= python
MKDOCS ?= mkdocs
MIKE ?= mike

help:
	@echo "Available commands:"
	@echo "  make install      - Install documentation dependencies"
	@echo "  make serve        - Serve documentation locally"
	@echo "  make build        - Build the static site with strict validation"
	@echo "  make  - Check links in public documentation"
	@echo "  make deploy-dev   - Build dev/latest into the local gh-pages branch"

install:
	$(PYTHON) -m pip install -r requirements.txt

serve:
	$(MKDOCS) serve -f mkdocs.yml

build:
	$(MKDOCS) build --strict -f mkdocs.yml

deploy-dev:
	$(MIKE) deploy --config-file mkdocs.yml --ignore-remote-status --update-aliases dev latest
	$(MIKE) set-default --ignore-remote-status latest

deploy-version:
	test -n "$(VERSION)"
	$(MIKE) deploy --config-file mkdocs.yml --update-aliases "$(VERSION)" latest
	$(MIKE) set-default latest

clean:
	rm -rf generated/
