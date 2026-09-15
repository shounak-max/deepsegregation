#!/usr/bin/env bash
set -Eeuo pipefail

# Verify or stage the original PSNet5 dataset on the remote Linux host.
# Authentication is intentionally delegated to ssh; no password is stored here.

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/deepsegregation}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/PSNet/PSNet5}"
ARCHIVE_PATH="${ARCHIVE_PATH:-$PROJECT_ROOT/data/PSNet/psnet5-download}"
DATA_URL="${DATA_URL:-}"
KEEP_ARCHIVE="${KEEP_ARCHIVE:-0}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

check_layout() {
    local missing=0
    local area
    for area in Area_1 Area_2 Area_3 Area_4; do
        if [[ ! -d "$DATA_ROOT/$area" ]]; then
            echo "MISSING area: $DATA_ROOT/$area"
            missing=1
        else
            echo "FOUND area: $DATA_ROOT/$area"
        fi
    done

    local annotation_count
    annotation_count="$(find "$DATA_ROOT" -type f -path '*/Room_*/Annotations/*.txt' | wc -l)"
    echo "Annotation files: $annotation_count"
    [[ "$missing" -eq 0 ]] || return 1
    [[ "$annotation_count" -gt 0 ]] || die "no PSNet5 annotation files found"
}

download_and_extract() {
    [[ -n "$DATA_URL" ]] || die "DATA_URL is required when PSNet5 is not already staged"
    require_command curl
    require_command tar
    mkdir -p "$(dirname "$ARCHIVE_PATH")" "$PROJECT_ROOT/data/PSNet"

    echo "Downloading PSNet5 archive to $ARCHIVE_PATH"
    curl --fail --location --continue-at - --retry 3 --output "$ARCHIVE_PATH" "$DATA_URL"
    [[ -s "$ARCHIVE_PATH" ]] || die "downloaded archive is empty"

    local extract_root="$PROJECT_ROOT/data/PSNet"
    case "$ARCHIVE_PATH" in
        *.tar|*.tar.gz|*.tgz)
            tar -xf "$ARCHIVE_PATH" -C "$extract_root"
            ;;
        *.zip)
            require_command unzip
            unzip -q -n "$ARCHIVE_PATH" -d "$extract_root"
            ;;
        *)
            die "unsupported archive type; use .tar, .tar.gz, .tgz, or .zip"
            ;;
    esac

    if [[ ! -d "$DATA_ROOT" ]]; then
        local candidate
        candidate="$(find "$extract_root" -mindepth 1 -maxdepth 3 -type d -name PSNet5 -print -quit)"
        [[ -n "$candidate" ]] || die "archive extracted but no PSNet5 directory was found"
        DATA_ROOT="$candidate"
    fi

    if [[ "$KEEP_ARCHIVE" != "1" ]]; then
        rm -f -- "$ARCHIVE_PATH"
    fi
}

main() {
    require_command find
    require_command df
    mkdir -p "$PROJECT_ROOT/data/PSNet"

    echo "Project root: $PROJECT_ROOT"
    echo "Dataset root: $DATA_ROOT"
    df -h "$PROJECT_ROOT"

    if ! check_layout; then
        if [[ -n "$DATA_URL" ]]; then
            echo "PSNet5 is incomplete; downloading archive."
            download_and_extract
            check_layout
        else
            die "PSNet5 is not staged. Set DATA_URL to an official archive URL and rerun."
        fi
    fi

    if [[ -f "$PROJECT_ROOT/scripts/check_psnet5.py" ]]; then
        python "$PROJECT_ROOT/scripts/check_psnet5.py" \
            --data-root "$DATA_ROOT" \
            --manifest "$PROJECT_ROOT/psnet5_manifest.json"
    else
        echo "WARNING: check_psnet5.py is not available; layout check completed locally."
    fi

    echo "PSNet5 staging verification passed."
}

main "$@"
