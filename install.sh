#!/bin/sh
# Install (or update) sserve: link the script onto your PATH and create a starter config.
#
#   From a checkout:   ./install.sh
#   Without cloning:   curl -fsSL https://raw.githubusercontent.com/alessandrocornacchia/sserve/main/install.sh | sh
#
# Without a checkout, the repo is fetched into $SSERVE_HOME (default ~/.local/share/sserve):
# with git if available (re-running pulls), otherwise as a tarball. Safe to re-run:
# an existing config is never overwritten.
set -e

url=${SSERVE_REPO:-https://github.com/alessandrocornacchia/sserve}
ref=${SSERVE_REF:-main}
bin=${SSERVE_BIN:-$HOME/.local/bin}
cfg=${SSERVE_CONFIG:-$HOME/.config/sserve/config.toml}

for tool in python3 ssh; do
    command -v "$tool" >/dev/null || { echo "sserve needs $tool; install it first" >&2; exit 1; }
done

here=$(cd "$(dirname "$0")" 2>/dev/null && pwd || true)
if [ -n "$here" ] && [ -f "$here/sserve" ] && [ -f "$here/install.sh" ]; then
    repo=$here
else  # piped from curl: no checkout to link to, fetch one
    repo=${SSERVE_HOME:-$HOME/.local/share/sserve}
    if [ -d "$repo/.git" ] && command -v git >/dev/null; then
        git -C "$repo" pull --ff-only --quiet
        echo "updated  $repo"
    elif command -v git >/dev/null; then
        rm -rf "$repo" && mkdir -p "$(dirname "$repo")"
        git clone --quiet --depth 1 --branch "$ref" "$url" "$repo"
        echo "cloned   $url -> $repo"
    else
        command -v curl >/dev/null || { echo "sserve needs git or curl to download itself" >&2; exit 1; }
        rm -rf "$repo" && mkdir -p "$repo"
        curl -fsSL "$url/archive/$ref.tar.gz" | tar -xzf - -C "$repo" --strip-components=1
        echo "fetched  $url ($ref) -> $repo"
    fi
fi

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
echo "next:    edit $cfg, then: sserve config show && sserve ls"
