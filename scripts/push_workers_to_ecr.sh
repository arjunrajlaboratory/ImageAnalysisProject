#!/usr/bin/env bash
#
# push_workers_to_ecr.sh
#
# Build worker Docker images and push them to AWS ECR, one or more workers
# at a time. Designed to be run from a developer laptop after sourcing AWS
# credentials (e.g. `source ../AWSDeploy/aws_credentials_prod.sh`).
#
# Each push produces two tags per worker:
#   <acct>.dkr.ecr.<region>.amazonaws.com/<prefix>/<category>/<worker>:<short-sha>
#   <acct>.dkr.ecr.<region>.amazonaws.com/<prefix>/<category>/<worker>:latest
#
# Run with --help for full usage. Run with --list to see available workers.

set -uo pipefail

if [ -z "${BASH_VERSION:-}" ] || [ "${BASH_VERSINFO[0]:-0}" -lt 4 ]; then
    echo "ERROR: This script needs bash 4 or newer (uses 'mapfile' and associative arrays)." >&2
    echo "       Detected: ${BASH_VERSION:-unknown}" >&2
    echo "       On macOS: 'brew install bash' and re-run with /opt/homebrew/bin/bash (or /usr/local/bin/bash)." >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

DEFAULT_REGION="${AWS_REGION:-us-east-1}"
DEFAULT_PREFIX="nimbus"
DEFAULT_PLATFORM="linux/amd64"
BUILDER_NAME="nimbus-ecr-builder"
CACHE_DIR="${REPO_ROOT}/.cache/buildx"

# Worker images build `FROM nimbusimage/<base>:latest`. Those base images are
# not on any public registry and are normally built locally for the host's
# architecture, so they can't be used for an amd64 cross-build. We build the
# needed base image(s) for the target platform, push them to ECR under
# <prefix>/base/<base>, and redirect each worker's `FROM` to the ECR copy with
# `docker buildx --build-context` (no worker Dockerfile edits required).
#
# Map: base image name (as it appears after `nimbusimage/` in FROM) -> Dockerfile.
declare -A BASE_DOCKERFILE=(
    [worker-base]="workers/base_docker_images/Dockerfile.worker_base"
    [image-processing-base]="workers/base_docker_images/Dockerfile.image_processing_worker_base"
    [test-worker-base]="workers/base_docker_images/Dockerfile.test_worker_base"
)
declare -A BUILT_BASES=()   # base name -> pushed image ref (built once per run)

# Workers that --all skips: dev/test/sample artifacts that aren't deployed to
# ECR. They still build fine on amd64, so they remain pushable by explicit name
# if ever needed. (`*_M1` variants are excluded earlier, in discover_workers.)
declare -a NON_DEPLOY_WORKERS=(
    random_squares
    sample_interface
    test_multiple_annotation
    test_file_creation_worker
)

REGION=""
PREFIX=""
PLATFORM=""
NO_CACHE=""
DRY_RUN=0
ALL=0
LIST_ONLY=0
ASSUME_YES=0
SKIP_BASE=0
declare -a REQUESTED=()

log()  { printf '[push-ecr] %s\n' "$*"; }
err()  { printf '[push-ecr] ERROR: %s\n' "$*" >&2; }
warn() { printf '[push-ecr] WARN:  %s\n' "$*" >&2; }

confirm() {
    [ "$ASSUME_YES" = "1" ] && return 0
    local prompt="$1 [y/N] "
    local answer=""
    read -r -p "$prompt" answer || return 1
    case "$answer" in
        y|Y|yes|YES) return 0 ;;
        *)           return 1 ;;
    esac
}

print_usage() {
    cat <<'USAGE'
Usage:
  push_workers_to_ecr.sh [OPTIONS] WORKER [WORKER ...]
  push_workers_to_ecr.sh --all [OPTIONS]
  push_workers_to_ecr.sh --list

Build and push worker Docker images to AWS ECR.

Options:
  --all              Build and push every discovered worker (asks first)
  --list             Print every discovered worker name and exit
  --region REGION    AWS region (default: $AWS_REGION or us-east-1)
  --prefix PREFIX    ECR repo prefix (default: nimbus)
  --platform PLAT    Build platform (default: linux/amd64)
  --no-cache         Disable the local buildx cache for this run
  --skip-base        Don't (re)build base images; assume they're already in ECR
  --dry-run          Print actions but run nothing
  -y, --yes          Skip confirmation prompts (creating ECR repos, --all)
  -h, --help         Show this help

Typical workflow:
  source ../AWSDeploy/aws_credentials_prod.sh
  ./scripts/push_workers_to_ecr.sh --list
  ./scripts/push_workers_to_ecr.sh blob_metrics_worker connect_timelapse

Worker names are directory names (see --list), e.g. blob_metrics_worker.

The script reads the AWS account ID from `aws sts get-caller-identity` and
fails fast with a clear message if credentials aren't loaded. It only uses
production Dockerfiles (the Dockerfile_M1 variants are skipped).

USAGE
}

# Output lines: "<worker_name>\t<category>\t<dockerfile_path>"
#
# We deliberately look only at the standard layouts:
#   workers/annotations/<name>/Dockerfile
#   workers/properties/{blobs,points,lines,connections}/<name>/Dockerfile
#
# Non-standard layouts (currently only piscis, which has predict/ and train/
# subdirs and its own docker-compose.yaml) are skipped; build those with
# build_machine_learning_workers.sh.
#
# `*_M1` worker directories are also skipped: those Dockerfiles hardcode the
# aarch64 Miniconda installer for Apple Silicon development and cannot build
# for linux/amd64 (the production target), so they are never pushed to ECR.
discover_workers() {
    local f name
    while IFS= read -r f; do
        name="$(basename "$(dirname "$f")")"
        case "$name" in *_M1) continue ;; esac   # arm64-only dev variant
        printf '%s\tannotations\t%s\n' "$name" "$f"
    done < <(find workers/annotations -mindepth 2 -maxdepth 2 -name Dockerfile -type f 2>/dev/null | sort)
    while IFS= read -r f; do
        name="$(basename "$(dirname "$f")")"
        case "$name" in *_M1) continue ;; esac   # arm64-only dev variant
        printf '%s\tproperties\t%s\n' "$name" "$f"
    done < <(find workers/properties -mindepth 3 -maxdepth 3 -name Dockerfile -type f 2>/dev/null | sort)
}

while [ $# -gt 0 ]; do
    case "$1" in
        --all)         ALL=1; shift ;;
        --list)        LIST_ONLY=1; shift ;;
        --region)      REGION="$2"; shift 2 ;;
        --region=*)    REGION="${1#*=}"; shift ;;
        --prefix)      PREFIX="$2"; shift 2 ;;
        --prefix=*)    PREFIX="${1#*=}"; shift ;;
        --platform)    PLATFORM="$2"; shift 2 ;;
        --platform=*)  PLATFORM="${1#*=}"; shift ;;
        --no-cache)    NO_CACHE="--no-cache"; shift ;;
        --skip-base)   SKIP_BASE=1; shift ;;
        --dry-run)     DRY_RUN=1; shift ;;
        -y|--yes)      ASSUME_YES=1; shift ;;
        -h|--help)     print_usage; exit 0 ;;
        --)            shift; while [ $# -gt 0 ]; do REQUESTED+=("$1"); shift; done ;;
        -*)            err "Unknown option: $1"; print_usage; exit 2 ;;
        *)             REQUESTED+=("$1"); shift ;;
    esac
done

REGION="${REGION:-$DEFAULT_REGION}"
PREFIX="${PREFIX:-$DEFAULT_PREFIX}"
PLATFORM="${PLATFORM:-$DEFAULT_PLATFORM}"

cd "$REPO_ROOT"

declare -a WORKER_LIST
mapfile -t WORKER_LIST < <(discover_workers)

if [ "${#WORKER_LIST[@]}" -eq 0 ]; then
    err "No workers found under workers/. Are you in the repo root?"
    exit 1
fi

declare -A CATEGORY_OF
declare -A DOCKERFILE_OF
declare -a ALL_NAMES=()
for line in "${WORKER_LIST[@]}"; do
    IFS=$'\t' read -r _name _cat _dfile <<<"$line"
    CATEGORY_OF["$_name"]="$_cat"
    DOCKERFILE_OF["$_name"]="$_dfile"
    ALL_NAMES+=("$_name")
done

if [ "$LIST_ONLY" = "1" ]; then
    printf '%-50s %-12s %s\n' "WORKER" "CATEGORY" "DOCKERFILE"
    for n in "${ALL_NAMES[@]}"; do
        printf '%-50s %-12s %s\n' "$n" "${CATEGORY_OF[$n]}" "${DOCKERFILE_OF[$n]}"
    done
    exit 0
fi

declare -a TARGETS=()
if [ "$ALL" = "1" ]; then
    if [ "${#REQUESTED[@]}" -gt 0 ]; then
        err "--all is incompatible with positional worker names."
        exit 2
    fi
    declare -a _skipped=()
    for n in "${ALL_NAMES[@]}"; do
        _is_excluded=0
        for d in "${NON_DEPLOY_WORKERS[@]}"; do
            [ "$n" = "$d" ] && { _is_excluded=1; break; }
        done
        if [ "$_is_excluded" = "1" ]; then
            _skipped+=("$n")
        else
            TARGETS+=("$n")
        fi
    done
    if [ "${#_skipped[@]}" -gt 0 ]; then
        log "Skipping ${#_skipped[@]} non-deployed worker(s) (push by name if needed): ${_skipped[*]}"
    fi
    log "About to build and push ${#TARGETS[@]} workers."
    if ! confirm "Proceed?"; then
        log "Aborted."
        exit 1
    fi
else
    if [ "${#REQUESTED[@]}" -eq 0 ]; then
        err "No workers specified. Use --list to see options or --all."
        print_usage
        exit 2
    fi
    declare -a _missing=()
    for n in "${REQUESTED[@]}"; do
        if [ -z "${DOCKERFILE_OF[$n]:-}" ]; then
            _missing+=("$n")
        fi
    done
    if [ "${#_missing[@]}" -gt 0 ]; then
        err "Unknown worker(s): ${_missing[*]}"
        err "Run with --list to see available worker names."
        exit 2
    fi
    TARGETS=("${REQUESTED[@]}")
fi

for tool in docker aws git; do
    command -v "$tool" >/dev/null 2>&1 || { err "Required tool '$tool' not found on PATH."; exit 1; }
done

log "Verifying AWS credentials..."
if ! ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)"; then
    err "aws sts get-caller-identity failed."
    err "Source your AWS credentials first (e.g. 'source aws_credentials_prod')."
    exit 1
fi
log "AWS account: $ACCOUNT_ID  region: $REGION"

REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

if ! SHA="$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null)"; then
    err "Not a git repository or no commits yet."
    exit 1
fi
# `git status --porcelain` reports tracked modifications, staged changes, AND
# untracked (non-.gitignored) files. Untracked files matter here because the
# build context is usually the repo root, so a stray file can be COPYed into
# an image while the tag still claims to match HEAD — breaking the `-dirty`
# provenance guarantee. (.gitignored build artifacts like .cache/ are excluded.)
if [ -n "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]; then
    warn "Working tree is dirty (tracked changes and/or untracked files). Tagging SHA images as '${SHA}-dirty'."
    SHA="${SHA}-dirty"
fi
log "Git SHA tag: $SHA"

ensure_builder() {
    if ! docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
        log "Creating buildx builder '$BUILDER_NAME'..."
        if [ "$DRY_RUN" = "1" ]; then
            printf '+ docker buildx create --name %s --driver docker-container --bootstrap\n' "$BUILDER_NAME"
        else
            docker buildx create --name "$BUILDER_NAME" --driver docker-container --bootstrap >/dev/null
        fi
    fi
}

ecr_login() {
    log "Logging in to ECR registry ${REGISTRY}"
    if [ "$DRY_RUN" = "1" ]; then
        printf '+ aws ecr get-login-password --region %s | docker login --username AWS --password-stdin %s\n' \
            "$REGION" "$REGISTRY"
        return 0
    fi
    aws ecr get-login-password --region "$REGION" \
        | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null
}

ensure_repo() {
    local repo="$1"
    if aws ecr describe-repositories --region "$REGION" --repository-names "$repo" >/dev/null 2>&1; then
        return 0
    fi
    log "ECR repo '$repo' does not exist."
    if [ "$DRY_RUN" = "1" ]; then
        printf '+ aws ecr create-repository --region %s --repository-name %s\n' "$REGION" "$repo"
        return 0
    fi
    if ! confirm "Create ECR repo '${repo}' now?"; then
        err "Skipping '$repo' (no repo)."
        return 1
    fi
    if aws ecr create-repository \
        --region "$REGION" \
        --repository-name "$repo" \
        --image-tag-mutability MUTABLE \
        --image-scanning-configuration scanOnPush=true \
        >/dev/null 2>&1; then
        log "Created repo '$repo'."
    else
        err "Failed to create ECR repo '$repo' (check name constraints / permissions)."
        return 1
    fi
}

# Determine the build context a Dockerfile expects by checking where its
# COPY/ADD sources actually resolve. Worker Dockerfiles in this repo are
# inconsistent: most compose-built workers COPY repo-root-relative paths
# (e.g. `./workers/<cat>/<name>/entrypoint.py`) and need the repo root as
# context, while a few (cellpose, stardist, random_point, ...) COPY
# worker-dir-relative paths (e.g. `./environment.yml`) and need the worker
# directory as context. We pick the context under which every source exists.
detect_context() {
    local dfile="$1"
    local wd; wd="$(dirname "$dfile")"
    local -a srcs=()
    local line
    while IFS= read -r line; do
        case "$line" in
            [Cc][Oo][Pp][Yy][[:space:]]*|[Aa][Dd][Dd][[:space:]]*) : ;;
            *) continue ;;
        esac
        case "$line" in *--from=*) continue ;; esac   # multi-stage, not from context
        local -a toks=() args=()
        read -r -a toks <<<"$line"
        local i
        for ((i = 1; i < ${#toks[@]}; i++)); do
            case "${toks[$i]}" in
                --*) continue ;;       # drop flags like --chown=/--chmod=
                *)   args+=("${toks[$i]}") ;;
            esac
        done
        local n=${#args[@]}
        [ "$n" -lt 2 ] && continue     # need at least one source + dest
        local j
        for ((j = 0; j < n - 1; j++)); do
            srcs+=("${args[$j]}")
        done
    done < "$dfile"

    if [ "${#srcs[@]}" -eq 0 ]; then
        printf '%s\n' "$REPO_ROOT"
        return 0
    fi

    local all_root=1 all_wd=1 s p
    for s in "${srcs[@]}"; do
        p="${s#./}"
        case "$p" in *[\*\?\[]*) p="$(dirname "$p")" ;; esac  # test dir for globs
        [ -e "${REPO_ROOT}/${p}" ] || all_root=0
        [ -e "${wd}/${p}" ]        || all_wd=0
    done

    if [ "$all_root" = "1" ]; then
        printf '%s\n' "$REPO_ROOT"
    elif [ "$all_wd" = "1" ]; then
        printf '%s\n' "$REPO_ROOT/$wd"
    else
        warn "Ambiguous build context for ${dfile}; defaulting to repo root."
        printf '%s\n' "$REPO_ROOT"
    fi
}

# Echo the full `nimbusimage/<base>:<tag>` reference a worker builds FROM, or
# nothing if the worker doesn't build from a known nimbusimage base.
base_ref_for_worker() {
    local dfile="$1"
    grep -iE '^[[:space:]]*FROM[[:space:]]+nimbusimage/' "$dfile" \
        | head -1 | awk '{print $2}'
}

# Build the given base image (by name, e.g. "worker-base") for the target
# platform and push it to ECR. Cached per run via BUILT_BASES.
ensure_base_image() {
    local base="$1"
    [ -n "${BUILT_BASES[$base]:-}" ] && return 0
    local dfile="${BASE_DOCKERFILE[$base]:-}"
    if [ -z "$dfile" ]; then
        err "No Dockerfile known for base image 'nimbusimage/${base}'. Cannot redirect FROM."
        return 1
    fi
    local repo="${PREFIX}/base/${base}"
    local img="${REGISTRY}/${repo}:latest"

    if [ "$SKIP_BASE" = "1" ]; then
        log "Base image: assuming ${img} already exists (--skip-base)."
        BUILT_BASES["$base"]="$img"
        return 0
    fi

    log "Ensuring amd64 base image '${base}' -> ${img}"
    ensure_repo "$repo" || return 1

    local base_cache_dir="${CACHE_DIR}/base-${base}"
    local -a cache_args=()
    if [ -z "$NO_CACHE" ]; then
        mkdir -p "$base_cache_dir"
        cache_args+=(--cache-from "type=local,src=${base_cache_dir}")
        cache_args+=(--cache-to   "type=local,dest=${base_cache_dir},mode=max")
    fi
    local -a no_cache_arg=()
    [ -n "$NO_CACHE" ] && no_cache_arg=("$NO_CACHE")

    if [ "$DRY_RUN" = "1" ]; then
        printf '+ docker buildx build --builder %s --platform %s %s %s -f %s -t %s --push %s\n' \
            "$BUILDER_NAME" "$PLATFORM" "${no_cache_arg[*]-}" "${cache_args[*]-}" \
            "$dfile" "$img" "$REPO_ROOT"
        BUILT_BASES["$base"]="$img"
        return 0
    fi

    if docker buildx build \
        --builder "$BUILDER_NAME" \
        --platform "$PLATFORM" \
        "${no_cache_arg[@]}" \
        "${cache_args[@]}" \
        -f "$dfile" \
        -t "$img" \
        --push \
        "$REPO_ROOT"; then
        BUILT_BASES["$base"]="$img"
        log "Base image '${base}' pushed: ${img}"
        return 0
    fi
    err "Failed to build/push base image '${base}'."
    return 1
}

build_and_push_worker() {
    local name="$1"
    local cat="${CATEGORY_OF[$name]}"
    local dfile="${DOCKERFILE_OF[$name]}"
    # ECR repo names and Docker image tags must be lowercase; some worker
    # directory names are not (e.g. blob_point_count_3D_..., the *_M1 variants).
    local repo="${PREFIX}/${cat}/${name,,}"
    local img_sha="${REGISTRY}/${repo}:${SHA}"
    local img_latest="${REGISTRY}/${repo}:latest"
    local worker_cache_dir="${CACHE_DIR}/${name}"

    # Build context this worker's Dockerfile actually expects.
    local ctx; ctx="$(detect_context "$dfile")"

    # If the worker builds FROM a nimbusimage base, build+push that base for
    # the target platform and redirect FROM to the ECR copy.
    local from_ref base_name
    from_ref="$(base_ref_for_worker "$dfile")"
    local -a base_ctx_args=()
    if [ -n "$from_ref" ]; then
        base_name="${from_ref#nimbusimage/}"; base_name="${base_name%%:*}"
        if [ -n "${BASE_DOCKERFILE[$base_name]:-}" ]; then
            ensure_base_image "$base_name" || return 1
            base_ctx_args+=(--build-context "${from_ref}=docker-image://${BUILT_BASES[$base_name]}")
        fi
    fi

    log "------------------------------------------------------------------"
    log "Worker:     ${name}"
    log "  Category: ${cat}"
    log "  Source:   ${dfile}"
    log "  Context:  ${ctx}"
    [ -n "$from_ref" ] && log "  Base:     ${from_ref} -> ${BUILT_BASES[$base_name]:-<unredirected>}"
    log "  Tags:     ${img_sha}"
    log "            ${img_latest}"
    log "------------------------------------------------------------------"

    ensure_repo "$repo" || return 1

    mkdir -p "$worker_cache_dir"

    local -a cache_args=()
    if [ -z "$NO_CACHE" ]; then
        cache_args+=(--cache-from "type=local,src=${worker_cache_dir}")
        cache_args+=(--cache-to   "type=local,dest=${worker_cache_dir},mode=max")
    fi

    local -a no_cache_arg=()
    [ -n "$NO_CACHE" ] && no_cache_arg=("$NO_CACHE")

    local attempt rc stderr_file
    for attempt in 1 2; do
        if [ "$DRY_RUN" = "1" ]; then
            printf '+ docker buildx build --builder %s --platform %s %s %s %s -f %s -t %s -t %s --push %s\n' \
                "$BUILDER_NAME" "$PLATFORM" \
                "${base_ctx_args[*]-}" \
                "${no_cache_arg[*]-}" "${cache_args[*]-}" \
                "$dfile" "$img_sha" "$img_latest" "$ctx"
            return 0
        fi
        stderr_file="$(mktemp)"
        docker buildx build \
            --builder "$BUILDER_NAME" \
            --platform "$PLATFORM" \
            "${base_ctx_args[@]}" \
            "${no_cache_arg[@]}" \
            "${cache_args[@]}" \
            -f "$dfile" \
            -t "$img_sha" \
            -t "$img_latest" \
            --push \
            "$ctx" 2> >(tee "$stderr_file" >&2)
        rc=$?
        if [ "$rc" -eq 0 ]; then
            rm -f "$stderr_file"
            return 0
        fi
        if [ "$attempt" -eq 1 ] && grep -qiE 'unauthorized|denied|no basic auth credentials|authentication required' "$stderr_file"; then
            warn "Push failed with an auth error. Re-logging in to ECR and retrying once."
            rm -f "$stderr_file"
            ecr_login || { err "ECR re-login failed."; return 1; }
            continue
        fi
        rm -f "$stderr_file"
        err "Build/push failed for '$name' (exit $rc)."
        return 1
    done
}

ensure_builder
ecr_login

declare -a SUCCESS=()
declare -a FAILURE=()
START_TS=$SECONDS

for name in "${TARGETS[@]}"; do
    worker_start=$SECONDS
    if build_and_push_worker "$name"; then
        SUCCESS+=("$(printf '%s (%s, %ds)' "$name" "${CATEGORY_OF[$name]}" $((SECONDS-worker_start)))")
    else
        FAILURE+=("$name")
    fi
done

TOTAL=$((SECONDS - START_TS))

log ""
log "=================================================================="
log "Summary"
log "=================================================================="
log "Registry: ${REGISTRY}"
log "Prefix:   ${PREFIX}"
log "SHA tag:  ${SHA}"
log "Targets:  ${#TARGETS[@]}"
log "Pushed:   ${#SUCCESS[@]}"
log "Failed:   ${#FAILURE[@]}"
log "Elapsed:  ${TOTAL}s"

if [ "${#SUCCESS[@]}" -gt 0 ]; then
    log ""
    log "Pushed:"
    for s in "${SUCCESS[@]}"; do
        log "  - $s"
    done
fi
if [ "${#FAILURE[@]}" -gt 0 ]; then
    log ""
    err "Failed:"
    for f in "${FAILURE[@]}"; do
        err "  - $f"
    done
    exit 1
fi
