# Public repository safety policy

This repository is public and intentionally contains **no** production images, customer/vendor identifiers, source annotations, internal dataset audit files, local ComfyUI JSON, model weights, tokens or credentials.

## Never publish

- `data/raw/`, `data/work/`, `data/pilot/`, `outputs/`, `reports/` or their contents.
- Local copies of ComfyUI graphs in `workflows/existing/`.
- Files under `docs/reference/`, `docs/internal/`, `docs/private/` or commercial class mappings.
- Real `.env` files, passwords, API keys, Dropbox links, tokens or authentication cookies.
- Dataset photographs, machine-generated annotation sets, reports, manifests or model weights without independent authorization.

## Before any push

Stage intended files (`git add ...`) and run `python scripts/check_public_repo.py`, then inspect `git diff --cached --name-only` and `git diff --cached` manually. The script rejects known file classes, patterns and secret formats; it **cannot prove** that a commit contains no confidential business information.

If private files were previously committed or pushed, `.gitignore` does not remove them from Git history. Stop publication, inspect history and rewrite it if needed; rotate any exposed credentials. Avoid force-pushing if collaborators depend on that history.

Report any suspected secret leakage privately to the owner; **do not** put actual credentials or proprietary sample data into public GitHub issues.
