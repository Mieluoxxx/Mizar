#!/bin/sh
set -eu

REPO="Mieluoxxx/Mizar"
INSTALL_DIR="${MIZAR_INSTALL_DIR:-${HOME}/.local/bin}"
MIZAR_BIN_DIR="${HOME}/.mizar/bin"
VERSION="${MIZAR_VERSION:-}"
SKIP_TOOLS="${MIZAR_SKIP_TOOLS:-0}"

usage() {
  cat <<'EOF'
Install prebuilt mizar binary from GitHub Releases.

Usage: install.sh [OPTIONS]

Options:
  -v, --version <ver>   Install a specific release tag (e.g. v0.1.0 or 0.1.0)
  -d, --dir <path>      Installation directory (default: ~/.local/bin)
      --skip-tools      Skip installing managed rg/fd helper binaries to ~/.mizar/bin
  -h, --help            Print this help message

Environment:
  MIZAR_VERSION         Same as --version
  MIZAR_INSTALL_DIR     Same as --dir
  MIZAR_SKIP_TOOLS      Set to 1 to skip managed rg/fd installation
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
    --skip-tools)
      SKIP_TOOLS=1
      shift
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

# 通过 /releases/latest 302 重定向解析最新 tag，避免触发 api.github.com 匿名限流。
resolve_latest_tag_redirect() {
  gh_repo="$1"
  effective_url=$(curl -fsSLI --connect-timeout 10 --max-time 20 -o /dev/null -w '%{url_effective}' "https://github.com/${gh_repo}/releases/latest" 2>/dev/null || true)
  case "$effective_url" in
    */releases/tag/*)
      echo "${effective_url##*/releases/tag/}"
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

tool_target_candidates() {
  case "$TARGET" in
    aarch64-apple-darwin) echo "aarch64-apple-darwin" ;;
    x86_64-apple-darwin) echo "x86_64-apple-darwin" ;;
    x86_64-unknown-linux-gnu) echo "x86_64-unknown-linux-musl x86_64-unknown-linux-gnu" ;;
    aarch64-unknown-linux-gnu) echo "aarch64-unknown-linux-musl aarch64-unknown-linux-gnu" ;;
    *) echo "" ;;
  esac
}

install_github_tool() {
  bin_name="$1"
  gh_repo="$2"
  asset_prefix="$3"
  tag_has_v="$4"

  tag=$(resolve_latest_tag_redirect "$gh_repo") || return 1
  ver="${tag#v}"
  if [ "$tag_has_v" = "1" ]; then
    ver_part="v${ver}"
  else
    ver_part="${ver}"
  fi

  candidates=$(tool_target_candidates)
  [ -n "$candidates" ] || return 1

  tool_tmp="${TMP_DIR}/tool-${bin_name}"
  mkdir -p "$tool_tmp"

  for triple in $candidates; do
    archive="${asset_prefix}-${ver_part}-${triple}.tar.gz"
    url="https://github.com/${gh_repo}/releases/download/${tag}/${archive}"
    archive_path="${tool_tmp}/${archive}"
    if curl -fsSL --connect-timeout 10 --max-time 120 "$url" -o "$archive_path" 2>/dev/null; then
      if tar -xzf "$archive_path" -C "$tool_tmp" 2>/dev/null; then
        found_bin=$(find "$tool_tmp" -type f -name "$bin_name" | head -n 1 || true)
        if [ -n "$found_bin" ]; then
          mkdir -p "$MIZAR_BIN_DIR"
          if command -v install >/dev/null 2>&1; then
            install -m 755 "$found_bin" "${MIZAR_BIN_DIR}/${bin_name}"
          else
            cp "$found_bin" "${MIZAR_BIN_DIR}/${bin_name}"
            chmod 755 "${MIZAR_BIN_DIR}/${bin_name}"
          fi
          echo "Installed managed ${bin_name} (${tag}) to ${MIZAR_BIN_DIR}/${bin_name}"
          return 0
        fi
      fi
    fi
  done

  return 1
}

install_helper_tools() {
  case "$SKIP_TOOLS" in
    1|true|TRUE|yes|YES)
      echo "Skipping managed rg/fd installation (MIZAR_SKIP_TOOLS/--skip-tools)."
      return 0
      ;;
  esac

  if command -v rg >/dev/null 2>&1 || [ -x "${MIZAR_BIN_DIR}/rg" ]; then
    :
  else
    echo "Installing managed rg to ${MIZAR_BIN_DIR}..."
    if ! install_github_tool "rg" "BurntSushi/ripgrep" "ripgrep" "0"; then
      echo "warning: failed to download ripgrep (rg); mizar will use its built-in Rust grep fallback" >&2
    fi
  fi

  if command -v fd >/dev/null 2>&1 || command -v fdfind >/dev/null 2>&1 || [ -x "${MIZAR_BIN_DIR}/fd" ]; then
    :
  else
    echo "Installing managed fd to ${MIZAR_BIN_DIR}..."
    if ! install_github_tool "fd" "sharkdp/fd" "fd" "1"; then
      echo "warning: failed to download fd; mizar will use its built-in Rust file walker fallback" >&2
    fi
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
install_helper_tools

case ":${PATH}:" in
  *":${INSTALL_DIR}:"*) ;;
  *)
    echo "Add ${INSTALL_DIR} to your PATH:"
    echo "  export PATH=\"${INSTALL_DIR}:\$PATH\""
    ;;
esac
