# Security

Do not report secrets by committing them to this repository or pasting them into
public issues. If you find a leaked credential, revoke it first, then report the
file path and commit or release artifact where it appeared.

The companion code is educational sample code. It does not read `.env` files by
default; workflows that use external APIs should keep `.env` limited to secrets
and keep prompts, manifests, outputs, and run metadata as ordinary ignored data
artifacts.
