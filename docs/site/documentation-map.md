# Documentation map

DLEngine documentation is organized by the task you are trying to complete.

## Run DLEngine

1. [Installation](installation.md) — prepare the development image or a local
   editable install.
2. [Supported models](supported-models.md) — confirm the checkpoint and its
   hardware, parallelism, and feature constraints.
3. Choose an execution path:
   - [Production serving](online-serving.md) for an OpenAI or Anthropic API.
   - [Offline inference](offline-inference.md) for validation and batch jobs.
   - [Transport-neutral P/D](pd-fabric-deployment.md) when prefill and decode
     use a transport fabric.

## Check behavior and tune a deployment

- [Feature matrix](features.md) describes what is available, experimental, or
  still planned.
- [GLM recurrent MTP](glm-recurrent-mtp.md) documents the five-token draft,
  six-row verification, cache reservation, and supported topologies.
- [Performance profiling](performance-profiling.md) gives a repeatable trace
  workflow and interpretation checklist.

## Read the engineering context

Design proposals and migration plans live in the repository `docs/` root so
they do not get mistaken for production guarantees. The catalog and status of
those notes is maintained in the [repository documentation README](https://github.com/JimyMa/NanoDeploy/blob/Pure_dp/docs/README.md).
Long-form measurements and architecture explanations are collected in the
[blog index](blogs/index.md).

## Where to make a change

Put user-facing, executable instructions in `docs/site/`; add the page to
`docs/mkdocs.yml` navigation. Put a decision, proposal, or migration plan in
the `docs/` root and link it from `docs/README.md`. Put a benchmark narrative
or explanatory deep dive in `docs/site/blogs/` and add it to the site blog index.
