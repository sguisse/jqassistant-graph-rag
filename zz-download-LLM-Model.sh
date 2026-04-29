#!/usr/bin/env bash

# 1) Ensure certifi is installed and point env vars to it
.venv/bin/python -m pip install --upgrade certifi
export SSL_CERT_FILE="$(.venv/bin/python -m certifi)"
export REQUESTS_CA_BUNDLE="$SSL_CERT_FILE"

# (Optional macOS system Python fix)
# If you use the official python.org installer, run:
# /Applications/Python\ 3.x/Install\ Certificates.command

# 2) Install git-lfs (Homebrew)
brew install git-lfs
git lfs install

# 3) Clone and fetch LFS objects securely (normal TLS, DO NOT disable verification)
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 models/all-MiniLM-L6-v2
cd models/all-MiniLM-L6-v2
git lfs fetch --all
git lfs pull
