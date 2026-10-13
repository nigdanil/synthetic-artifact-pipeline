# Windows setup — public SAP scaffold

This installs an **isolated Python environment**. It does not install a new ComfyUI or modify the currently working ComfyUI environment.

## Requirements

- Windows 10/11 x64, PowerShell, Git and Python 3.12 x64 (Python 3.11–3.13 may work with compatible dependencies).
- Optional: existing local ComfyUI for a future integration step.
- GitHub CLI (`gh`) for repository publishing, see https://cli.github.com/.

## Install

Clone the public repository after creating it, then run:

```powershell
git clone https://github.com/nigdanil/synthetic-artifact-pipeline.git
cd synthetic-artifact-pipeline
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev,image]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\check_environment.py
```

If cloning is not yet available, extract the public ZIP to a separate folder and run the same commands from the extracted repository root, without the two `git clone`/`cd` lines.

Python 3.12 is a recommendation for **this project's `.venv`**; it is not a request to change ComfyUI's Python. `[image]` includes NumPy, Pillow and OpenCV headless; `[dev]` includes pytest and Ruff. No new diffusion model, GPU packages or Blender are needed yet.

## Local-only inputs

Place dataset files in `data/raw/`, respecting legal and third-party restrictions. Keep photographs and vendor annotations *only on your own machine*. Never stage or upload this directory. Place an existing ComfyUI workflow JSON under `workflows/existing/` locally. Avoid storing credentials in the graph or repo; use placeholders in shared templates.

## Before public push

```powershell
git status --short
git add .
.\.venv\Scripts\python.exe scripts\check_public_repo.py
git diff --cached --name-only
```

The audit must report success; check every listed file yourself. Only then commit and push. If the tool flags a private file, **unstage** it with `git restore --staged <path>` and remove it from the publication set.

## Publishing the initial repository through GitHub CLI

On the **sanitized public-source folder only**:

```powershell
gh auth login
git init -b main
git add .
.\.venv\Scripts\python.exe scripts\check_public_repo.py
git commit -m "chore: publish safe Python scaffold"
gh repo create nigdanil/synthetic-artifact-pipeline --public --source=. --remote=origin --push
```

**Warning:** Do not use `git push --all` or `git push --mirror` from an old local repository if older commits contained internal datasets or secrets. Use a newly initialized Git history made exclusively from this vetted public-source directory.

## What is not ready yet

Dataset indexer, parser, ROI selection, masks, effects, custom nodes, exporter, QA and batch mode are future tasks. The smoke tests only validate packaging and the scaffold.
