# 0m1-w1z4rd customization boundary

`0m1-w1z4rd` is the isolated Omi flow-router implementation derived from
`nlcli-wizard`. It intentionally does not act as a shared plugin host or registry.

## Ownership

This repository owns:

- the packaged snapshot in `nlcli_wizard/catalog/flows.json`;
- deterministic flow training data and the independently written holdout;
- the Omi flow-router training recipe and quantized-model identity;
- the hash-bound artifact record in `implementation.json`;
- the `0m1-w1z4rd` CLI, cache, state, and evaluation namespaces;
- validation that emitted flow targets belong to this catalog.

The consuming Omi host continues to own flow lifecycle, authorization, process
launching, and final catalog validation. A model prediction is only a candidate.

## Isolation

- Distribution and CLI identity: `0m1-w1z4rd`.
- Python import identity: `nlcli_wizard`, retained for upstream compatibility.
- Model cache: `~/.cache/0m1-w1z4rd/models`.
- Local state: `~/.local/state/0m1-w1z4rd`.
- Ollama model name: `0m1-w1z4rd:latest`.
- Model weights and raw generations remain local/ignored.
- The only configured Git remote is `upstream`; there is no push target for the
  customized repository until a separate remote is deliberately created.

## Local artifact lifecycle

Generate and validate the implementation-owned dataset:

```bash
.venv/bin/python -m nlcli_wizard.dataset_flows
.venv/bin/python -m training.train_flow_router --validate-only
```

Create the namespaced Ollama artifact after the GGUF is available:

```bash
ollama create 0m1-w1z4rd -f Modelfile.flow-router
```

Future plugin discovery may wrap this implementation, but must not move its model,
state, catalog, or lifecycle ownership into another use.