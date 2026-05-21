#!/usr/bin/env bash
# Deploy the Temporal Gaussian Hierarchy plugin into an EasyVolcap clone.
#
# Usage:
#   ./tgh-plugin/install.sh /path/to/EasyVolcap
#
# Copies the 7 contributed/patched files into their target locations.
# Re-run after editing the canonical copies under tgh-plugin/.
set -e

EVC="${1:-../EasyVolcap}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$EVC/easyvolcap" ]; then
    echo "error: '$EVC' does not look like an EasyVolcap checkout" >&2
    echo "usage: $0 /path/to/EasyVolcap" >&2
    exit 1
fi

echo "Deploying TGH plugin into: $EVC"
while IFS= read -r f; do
    rel="${f#$HERE/}"
    mkdir -p "$EVC/$(dirname "$rel")"
    cp "$f" "$EVC/$rel"
    echo "  $rel"
done < <(find "$HERE" -type f \( -name '*.py' -o -name '*.yaml' \))

echo "done. Re-register the EasyVolcap CLI if needed:"
echo "  pip install -e \"$EVC\" --no-build-isolation --no-deps"
