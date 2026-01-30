#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# CPU backend (default features)
cargo run --release
