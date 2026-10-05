#!/bin/sh
# Install sserve: link the script onto your PATH and create a starter config.
# Safe to re-run: an existing config is never overwritten.
set -e

repo=$(cd "$(dirname "$0")" && pwd)
bin=${SSERVE_BIN:-$HOME/.local/bin}
cfg=${SSERVE_CONFIG:-$HOME/.config/sserve/config.toml}

for tool in python3 ssh; do
    command -v "$tool" >/dev/null || { echo "sserve needs $tool; install it first" >&2; exit 1; }
done

mkdir -p "$bin" "$(dirname "$cfg")"
ln -sf "$repo/sserve" "$bin/sserve"
echo "linked   $bin/sserve -> $repo/sserve"

if [ -e "$cfg" ]; then
    echo "kept     $cfg"
else
    cp "$repo/config.example.toml" "$cfg"
    echo "created  $cfg"
fi

python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' ||
    echo "note: the config file needs Python >= 3.11 (try: pixi global install python)"
case ":$PATH:" in
    *":$bin:"*) ;;
    *) echo "note: $bin is not on your PATH; add it, e.g. export PATH=\"$bin:\$PATH\"" ;;
esac
echo "next:    ssh -o BatchMode=yes orix true && sserve ls"
