# DLEngine documentation

This directory has three deliberately separate layers. Use the first layer for
running DLEngine, the second for implementation decisions, and the third for
long-form measurements and explanations.

## Find the right document

### User and operator guides — `docs/site/`

These pages are built into the MkDocs site and describe behavior that users can
rely on:

- [Installation](site/installation.md) — development image, local install,
  router, and DLSlime prerequisites.
- [Production serving](site/online-serving.md) — Ray, control plane, P/D
  engines, router, requests, and operational checks.
- [Transport-neutral P/D](site/pd-fabric-deployment.md) — deployment when the
  KV transport is provided by a fabric rather than a fixed topology.
- [Offline inference](site/offline-inference.md) — checkpoint validation and
  batch execution without a public gateway.
- [Feature matrix](site/features.md) and [supported models](site/supported-models.md)
  — current support and known constraints.
- [Performance profiling](site/performance-profiling.md) — reproducible traces
  and checks for MTP, PP prefill, and P/D migration.

The site navigation is defined in [`mkdocs.yml`](mkdocs.yml). If a page is
intended for users, add it there and link it from the site home or a relevant
guide.

### Engineering notes and plans — this directory

These documents capture design context, migration plans, or implementation
proposals. They are intentionally excluded from the public site unless a page
is promoted into `docs/site/`:

- Backend boundaries: [interface refactor](backend-interface-refactor.md),
  [family layout refactor](backend-family-layout-refactor.md), and the
  [DSA backend family](dsa-backend-family.md).
- Attention and cache design: [HiSparse design (English)](hisparse-design.md),
  [HiSparse design (中文)](hisparse-design.zh.md), [NSA/DSA notes](nsa_sparse_attention.md),
  [cache system](caching-system.md), and [chunked prefill](chunk-prefill.md).
- Roadmaps and implementation plans: [DSpark support](dspark-support-plan.zh.md)
  and [decode metadata CUDA kernel](decode-metadata-cuda-kernel-plan.zh.md).
- Project operations: [GitHub workflow](github-workflow.zh.md) and
  [releasing](releasing.md).

The similarly named [`hisparse_design.md`](hisparse_design.md) is an earlier
MVP proposal kept for historical context. New HiSparse decisions belong in
`hisparse-design.md` (and its Chinese counterpart) so there is one canonical
design pair.

### Research articles — `docs/site/blogs/`

Articles explain measurements or implementation details at greater length.
Start with the [blog index](site/blogs/index.md). An article becomes part of the
site only when it is listed in both that index and `mkdocs.yml`.

## Build the documentation site

The site uses MkDocs Material. From this directory:

```bash
python -m pip install -r requirements.txt
make serve       # http://127.0.0.1:8000/
make build       # strict build used by CI
```

CI uploads `dlengine-docs-<commit>` as a private artifact for 14 days. GitHub
Pages remains disabled because this repository is private.

## Keeping the system tidy

When adding documentation, choose the layer by audience first. Keep runnable
instructions in `docs/site/`, decisions and proposed work in the root `docs/`,
and measured deep dives in `docs/site/blogs/`. Give a design note a status and a
canonical filename, link related documents from this map, and update the
MkDocs navigation whenever a site page is added or renamed.
