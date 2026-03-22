.PHONY: install install-dev \
        install-dlslime install-nanoctrl install-nanodeploy \
        install-dlslime-dev install-nanoctrl-dev install-nanodeploy-dev

# ── Production installs (build from source, no editable) ─────────────────────

install:
	pip install ".[all]"

install-dlslime:
	pip install ".[dlslime]"

install-nanoctrl:
	pip install ".[nanoctrl]"

install-nanodeploy:
	pip install ".[nanodeploy]"

# ── Development installs (editable, changes take effect immediately) ──────────

install-dev:
	pip install -e ./DLSlime
	pip install -e ./NanoCtrl
	pip install -e ./NanoDeploy

install-dlslime-dev:
	pip install -e ./DLSlime

install-nanoctrl-dev:
	pip install -e ./NanoCtrl

install-nanodeploy-dev:
	pip install -e ./NanoDeploy
