# Security

## Reporting

Do not open a public issue containing API keys, SSH credentials, private data,
or model-provider tokens. Rotate any credential that has appeared in a terminal
or chat transcript.

## Runtime guidance

- Supply secrets through environment variables or a secret manager.
- Keep the demo bound to `127.0.0.1` unless authentication and TLS are added.
- Treat uploaded candidate text and provider responses as untrusted input.
- Preserve request-size limits, rate limiting, timeouts, circuit breaking, and
  Student fallback before exposing the API.
- Never commit raw user logs, checkpoints with private training data, or cached
  provider payloads containing sensitive content.

The current application is a local demonstration and does not implement user
authentication or multi-tenant isolation.
