#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Metal GPU backend
USE_METAL=1 cargo run --release --features metal
