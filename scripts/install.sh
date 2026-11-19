#!/bin/sh
set -eu

REPO="Mieluoxxx/mizar"
INSTALL_DIR="${MIZAR_INSTALL_DIR:-${HOME}/.local/bin}"
VERSION="${MIZAR_VERSION:-}"

usage() {
  cat <<'EOF'
Install prebuilt mizar binary from GitHub Releases.

Usage: install.sh [OPTIONS]

Options:
  -v, --version <ver>   Install a specific release tag (e.g. v0.1.0 or 0.1.0)
  -d, --dir <path>      Installation directory (default: ~/.local/bin)
  -h, --help            Print this help message

Environment:
  MIZAR_VERSION         Same as --version
  MIZAR_INSTALL_DIR     Same as --dir
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -v|--version)
      if [ "$#" -lt 2 ]; then
        echo "error: missing value for $1" >&2
        exit 1
      fi
      VERSION="$2"
      shift 2
      ;;
    -d|--dir)
      if [ "$#" -lt 2 ]; then
        echo "error: missing value for $1" >&2
        exit 1
      fi
      INSTALL_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option '$1'" >&2
      usage >&2
      exit 1
      ;;
  esac
done

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "error: required command '$1' was not found" >&2
    exit 1
  fi
}

need_cmd curl
need_cmd tar

detect_target() {
  os=$(uname -s)
  arch=$(uname -m)

  case "$os" in
    Darwin)
      if [ "$arch" = "x86_64" ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ]; then
        arch="arm64"
      fi
      case "$arch" in
        arm64|aarch64) echo "aarch64-apple-darwin" ;;
        x86_64) echo "x86_64-apple-darwin" ;;
        *)
          echo "error: unsupported macOS architecture '$arch'" >&2
          exit 1
          ;;
      esac
      ;;
    Linux)
      case "$arch" in
        x86_64|amd64) echo "x86_64-unknown-linux-gnu" ;;
        aarch64|arm64) echo "aarch64-unknown-linux-gnu" ;;
        *)
          echo "error: unsupported Linux architecture '$arch'" >&2
          exit 1
          ;;
      esac
      ;;
    *)
      echo "error: unsupported OS '$os' (prebuilt binaries are provided for macOS and Linux)" >&2
      exit 1
      ;;
  esac
}

resolve_tag() {
  if [ -n "$VERSION" ]; then
    case "$VERSION" in
      v*) echo "$VERSION" ;;
      *) echo "v$VERSION" ;;
    esac
    return 0
  fi

  effective_url=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/${REPO}/releases/latest" || true)
  tag="${effective_url##*/}"
  case "$tag" in
    v*)
      echo "$tag"
      return 0
      ;;
  esac

  api_tag=$( (curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" || true) \
    | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
    | head -n 1)
  if [ -z "$api_tag" ]; then
    echo "error: no latest GitHub Release found for ${REPO} (pass --version vX.Y.Z to install a specific tag)" >&2
    exit 1
  fi
  echo "$api_tag"
}

verify_sha256() {
  file_path="$1"
  checksums_file="$2"
  archive_name="$3"

  expected=$(awk -v name="$archive_name" '$2 == name || $2 == ("*" name) { print $1; exit }' "$checksums_file")
  if [ -z "$expected" ]; then
    echo "error: checksum entry for '$archive_name' not found in checksums.txt" >&2
    exit 1
  fi

  if command -v sha256sum >/dev/null 2>&1; then
    actual=$(sha256sum "$file_path" | awk '{ print $1 }')
  elif command -v shasum >/dev/null 2>&1; then
    actual=$(shasum -a 256 "$file_path" | awk '{ print $1 }')
  else
    echo "error: neither 'sha256sum' nor 'shasum' is available to verify download" >&2
    exit 1
  fi

  if [ "$expected" != "$actual" ]; then
    echo "error: SHA256 mismatch for $archive_name (expected $expected, got $actual)" >&2
    exit 1
  fi
}

TARGET=$(detect_target)
TAG=$(resolve_tag)
ARCHIVE="mizar-${TAG}-${TARGET}.tar.gz"
BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT INT TERM

echo "Installing mizar ${TAG} (${TARGET})..."
curl -fsSL "${BASE_URL}/${ARCHIVE}" -o "${TMP_DIR}/${ARCHIVE}"
curl -fsSL "${BASE_URL}/checksums.txt" -o "${TMP_DIR}/checksums.txt"
verify_sha256 "${TMP_DIR}/${ARCHIVE}" "${TMP_DIR}/checksums.txt" "${ARCHIVE}"

tar -xzf "${TMP_DIR}/${ARCHIVE}" -C "${TMP_DIR}"
BIN_SRC="${TMP_DIR}/mizar-${TAG}-${TARGET}/mizar"
if [ ! -f "$BIN_SRC" ]; then
  echo "error: extracted archive did not contain mizar binary" >&2
  exit 1
fi

mkdir -p "$INSTALL_DIR"
if command -v install >/dev/null 2>&1; then
  install -m 755 "$BIN_SRC" "${INSTALL_DIR}/mizar"
else
  cp "$BIN_SRC" "${INSTALL_DIR}/mizar"
  chmod 755 "${INSTALL_DIR}/mizar"
fi

echo "Installed $("${INSTALL_DIR}/mizar" --version) to ${INSTALL_DIR}/mizar"
case ":${PATH}:" in
  *":${INSTALL_DIR}:"*) ;;
  *)
    echo "Add ${INSTALL_DIR} to your PATH:"
    echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
    ;;
esac
