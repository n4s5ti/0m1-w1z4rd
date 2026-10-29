# wiz4rd

`wiz4rd` is the Omi flow-router implementation derived from
[`nlcli-wizard`](https://github.com/n4s5ti/nlcli-wizard). It provides the
`wiz4rd` global CLI while retaining the upstream `nlcli_wizard` Python import
package.

For `flow`, the model returns preview-only, closed-catalog candidates. The
consuming Omi host retains lifecycle, authorization, validation, and execution
authority.

## Install globally

Install from this checkout in editable mode:

```bash
uv tool install --editable . --force
```

Editable installation is required. The bundled GGUF lives in this checkout's
`models/` directory and is deliberately excluded from wheel data. An editable
installation keeps the global `wiz4rd` executable connected to that
checkout-relative model.

After installation, use the executable from any directory:

```bash
wiz4rd --help
```

Do not install a wheel when using the bundled local model. Moving or deleting
the checkout also removes the model location used by the editable installation.

## Translate a request

Use the global CLI from any directory:

```bash
wiz4rd docker "build an image tagged demo"
```

`wiz4rd <tool> <request...>` resolves only an executable available on `PATH`.
It never executes the candidate it produces. `venvy`, `docker`, and `flow` use
their dedicated models; `atuin` and all other resolved tools use the personal
model.

Static inspection commands are also available:

```bash
wiz4rd list-tools
wiz4rd model-info
wiz4rd cache-dir
```

For an explicit registry selection, use `translate`:

```bash
wiz4rd translate --cli-tool flow "start the greeting demo"
```

Before model use, `wiz4rd` supplies only bounded `--help` output as untrusted
reference data; it never treats that output as instructions.

## Python integration

The public distribution and executable are named `wiz4rd`. Python integrations
continue to import `nlcli_wizard`, preserving the upstream import architecture:

```python
import nlcli_wizard
```

This import identity is intentional; do not substitute a `wiz4rd` Python
package name.

## Local identities

| Purpose | Value |
| --- | --- |
| Distribution and CLI | `wiz4rd` |
| Python import package | `nlcli_wizard` |
| Model cache | `~/.cache/wiz4rd/models` |
| Local state | `~/.local/state/wiz4rd` |
| Ollama model | `wiz4rd:latest` |
| Version | `0.1.0+omi.1` |

Model weights and raw generations remain local and ignored. See
[`CUSTOMIZATION.md`](CUSTOMIZATION.md) for ownership boundaries and the local
artifact lifecycle.

## License

[MIT License](LICENSE)
