# Documentation website

The DLEngine documentation site uses MkDocs Material. CI builds it as a private GitHub Actions artifact; it is not published to GitHub Pages.

From this directory:

```bash
python -m pip install -r requirements.txt
make serve
```

The local site is available at `http://127.0.0.1:8000/`. Run the same strict build used by CI with:

```bash
make build
```

Site pages live under `docs/site/`. Internal design notes elsewhere under `docs/` are intentionally excluded from the generated artifact.

Every documentation workflow run uploads `dlengine-docs-<commit>` for 14 days. Only users with repository access can download Actions artifacts. GitHub Pages publishing is intentionally disabled because standard Pages would make a site from this private personal repository publicly accessible.
