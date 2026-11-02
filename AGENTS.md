# Public repository contributions

This repository is public. Keep code, documentation, examples, tests and pull
request descriptions reusable and independent of a contributor's private setup.

- Never commit real meeting recordings or transcripts, voice embeddings,
  application databases, raw benchmark results, diagnostic dumps, credentials,
  tokens or private model assets. Use clearly fictional fixtures in tests.
- Keep private benchmark inputs, outputs and machine-specific setup outside the
  checkout. Benchmark programs and generic reproducible instructions are welcome.
- Do not include personal home paths, account names, internal hostnames, network
  addresses, server inventories or descriptions of hardware owned by a user.
  Use placeholders and generic platform names such as CPU, NVIDIA CUDA and
  Apple Silicon. Public upstream project names and model identifiers are fine.
- Permission to inspect or process a recording or machine does not authorize
  publishing its content or identifying details. Publish real measurements or
  identifying information only when the user explicitly requests publication,
  after removing unrelated private details.
- Before committing, inspect every staged file and the staged diff, including
  new documentation and generated files. Check for sensitive content locally
  without printing secrets into logs or the conversation. Exclude private
  artifacts; an ignore rule alone does not sanitize an already tracked file.
