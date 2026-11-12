# Sovara for the OpenAI Agents SDK

Record OpenAI Agents tool calls in Sovara traces.

| Package | Install | Docs |
| --- | --- | --- |
| [`python/`](python) | `uv add sovara-openai-agents` or `pip install sovara-openai-agents` | [README](python/README.md) |
| [`typescript/`](typescript) | `npm install @sovara/openai-agents @sovara/sdk` | [README](typescript/README.md) |

Runnable examples are in [`examples/`](examples).

## Development

The packages use the Sovara SDK from a [sovara](https://github.com/SovaraLabs/sovara) checkout
next to this repository (`../sovara`).
CI checks out sovara's `main` branch (set the `SOVARA_REF` repository variable to use
another one) with the `SOVARA_REPO_TOKEN` secret, a token that can read `SovaraLabs/sovara`.

Python (`python/`):

```sh
cd python
uv sync
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
```

TypeScript (`typescript/`): build the SDK first, then run the checks:

```sh
(cd ../sovara/packages/typescript-sdk && npm ci && npm run build)
cd typescript
npm ci
npm run check
```

## License

MIT.
