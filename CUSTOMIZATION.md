# wiz4rd customization boundary

`wiz4rd` is the dedicated Omi flow-router implementation derived from
[`nlcli-wizard`](https://github.com/n4s5ti/nlcli-wizard). It retains the
upstream `nlcli_wizard` Python import architecture and does not act as a shared
plugin host or registry.

## Ownership

This repository owns:

- the packaged snapshot in `nlcli_wizard/catalog/flows.json`;
- deterministic flow training data and the independently written holdout;
- the Omi flow-router training recipe and quantized-model identity;
- the hash-bound artifact record in `implementation.json`;
- the `wiz4rd` CLI, cache, state, and evaluation namespaces;
- validation that emitted flow targets belong to this catalog.

The consuming Omi host continues to own flow lifecycle, authorization, process
launching, and final catalog validation. A model prediction is preview-only and
only a candidate.

## Identities and namespaces

- Distribution and CLI identity: `wiz4rd`.
- Python import identity: `nlcli_wizard`, retained for upstream compatibility.
- Model cache: `~/.cache/wiz4rd/models`.
- Local state: `~/.local/state/wiz4rd`.
- Ollama model name: `wiz4rd:latest`.
- Model weights and raw generations remain local/ignored.
- The only configured Git remote is `upstream`; there is no push target for the
  customized repository until a separate remote is deliberately created.

## Editable global installation

Install from the checkout:

```bash
uv tool install --editable . --force
```

Editable mode is required because the bundled GGUF remains in checkout-relative
`models/`, which is excluded from wheel data. It makes the global `wiz4rd`
command resolve that model while retaining the `nlcli_wizard` import package.

From any directory, inspect the flow registry or produce a preview:

```bash
wiz4rd list-tools
wiz4rd translate --cli-tool flow start the greeting demo
```

The `flow` registry entry returns a candidate only; it does not execute a flow.

## Local artifact lifecycle

Generate and validate the implementation-owned dataset:

```bash
.venv/bin/python -m nlcli_wizard.dataset_flows
.venv/bin/python -m training.train_flow_router --validate-only
```

Create the namespaced Ollama artifact after the GGUF is available:

```bash
ollama create wiz4rd -f Modelfile.flow-router
```

Future plugin discovery may wrap this implementation, but must not move its model,
state, catalog, or lifecycle ownership into another use.