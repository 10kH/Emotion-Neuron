#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f emoprism.json.gz ]]; then
  echo "ERROR: emoprism.json.gz not found. Pull it from the repo first." >&2
  exit 1
fi
gunzip -k emoprism.json.gz
sha256sum -c <<< "886d80a549c81e4ca77a17c4f8605450bfe4ba557de35efb644f4c6711dddb9a  emoprism.json"
echo "emoprism.json extracted and verified."
