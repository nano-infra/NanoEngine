# Documentation website

The public DLEngine documentation site uses MkDocs Material and follows the same versioned GitHub Pages pattern as DLSlime.

From this directory:

```bash
python -m pip install -r requirements.txt
make serve
```

The local site is available at `http://127.0.0.1:8000/`. Run the same strict build used by CI with:

```bash
make build
```

Public pages live under `docs/site/`. Internal design notes elsewhere under `docs/` are intentionally excluded from the published navigation and site artifact.

Pushes to `Pure_dp` publish the `dev` and `latest` aliases. Tags named `v*` publish a versioned snapshot through `mike`. The repository's GitHub Pages source must be configured to serve the `gh-pages` branch from its root.
