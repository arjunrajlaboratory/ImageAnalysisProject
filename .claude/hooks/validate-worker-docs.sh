#!/bin/bash
# Pre-PR hook: validates that modified workers have documentation and REGISTRY.md is updated.
# Called as a PreToolUse hook on Bash commands. Reads tool input JSON from stdin.
# Exit 0 = allow, Exit 2 = block with message.

set -euo pipefail

# Read the tool input JSON from stdin
INPUT=$(cat)

# Extract the command being run
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Only activate on `gh pr create` commands
if [[ ! "$COMMAND" =~ ^gh[[:space:]]+pr[[:space:]]+create ]]; then
    exit 0
fi

# Find changed files relative to master
CHANGED_FILES=$(git diff --name-only origin/master...HEAD 2>/dev/null || git diff --name-only master...HEAD 2>/dev/null || echo "")

if [ -z "$CHANGED_FILES" ]; then
    exit 0
fi

ERRORS=""

# Check annotation workers: any changed entrypoint.py should have a *.md doc in the same directory
while IFS= read -r file; do
    if [[ "$file" =~ ^workers/annotations/([^/]+)/entrypoint\.py$ ]]; then
        worker_dir=$(dirname "$file")
        worker_name="${BASH_REMATCH[1]}"
        # Check if any .md file (other than EXPERIMENT_RESULTS.md) exists in the worker dir
        has_doc=false
        for md_file in "$worker_dir"/*.md; do
            if [ -f "$md_file" ] && [[ ! "$md_file" =~ EXPERIMENT_RESULTS\.md$ ]]; then
                has_doc=true
                break
            fi
        done
        # Also check local_tests subdirectory exclusion
        if [ "$has_doc" = false ]; then
            ERRORS="$ERRORS\n  - Annotation worker '$worker_name' missing documentation (expected ${worker_dir}/$(echo "$worker_name" | tr '[:lower:]' '[:upper:]').md)"
        fi
    fi
done <<< "$CHANGED_FILES"

# Check property workers: any changed entrypoint.py should have a *.md doc
while IFS= read -r file; do
    if [[ "$file" =~ ^workers/properties/[^/]+/([^/]+)/entrypoint\.py$ ]]; then
        worker_dir=$(dirname "$file")
        worker_name="${BASH_REMATCH[1]}"
        has_doc=false
        for md_file in "$worker_dir"/*.md; do
            if [ -f "$md_file" ] && [[ ! "$(basename "$md_file")" == "README.md" ]]; then
                has_doc=true
                break
            fi
        done
        if [ "$has_doc" = false ]; then
            ERRORS="$ERRORS\n  - Property worker '$worker_name' missing documentation (expected a WORKERNAME.md file in ${worker_dir}/)"
        fi
    fi
done <<< "$CHANGED_FILES"

# Check if any registry-affecting changes were made without updating REGISTRY.md.
# REGISTRY.md is generated from: entrypoint.py (interface + description) and
# Dockerfile LABEL lines (interfaceName, interfaceCategory, description,
# annotationShape). Pure Dockerfile changes that don't touch those labels —
# e.g. ENV/RUN edits, base image bumps — don't change the registry, so we
# shouldn't require the user to "update" it.

# Compute the diff base once and reuse for line-level inspection.
DIFF_BASE=$(git merge-base origin/master HEAD 2>/dev/null || git merge-base master HEAD 2>/dev/null || echo "")

HAS_REGISTRY_RELEVANT_CHANGES=false

# Any added/deleted worker directory (detected via added/deleted entrypoint.py)
# affects the registry. `--diff-filter=AD` gives us add/delete-only changes.
ADDED_DELETED=$(git diff --name-only --diff-filter=AD ${DIFF_BASE:+"$DIFF_BASE"...HEAD} 2>/dev/null || echo "")
while IFS= read -r file; do
    [ -z "$file" ] && continue
    if [[ "$file" =~ ^workers/(annotations|properties/[^/]+)/[^/]+/entrypoint\.py$ ]]; then
        HAS_REGISTRY_RELEVANT_CHANGES=true
        break
    fi
done <<< "$ADDED_DELETED"

# Inspect each modified worker file. entrypoint.py edits always count.
# Dockerfile* edits only count if they touch a registry-relevant LABEL.
if [ "$HAS_REGISTRY_RELEVANT_CHANGES" = false ]; then
    while IFS= read -r file; do
        [ -z "$file" ] && continue
        if [[ "$file" =~ ^workers/.+/entrypoint\.py$ ]]; then
            HAS_REGISTRY_RELEVANT_CHANGES=true
            break
        fi
        if [[ "$file" =~ ^workers/.+/Dockerfile([^/]*)?$ ]] && [ -n "$DIFF_BASE" ]; then
            if git diff "$DIFF_BASE...HEAD" -- "$file" 2>/dev/null \
                 | grep -qE '^[+-][[:space:]]*(LABEL|interfaceName|interfaceCategory|annotationShape|description)[[:space:]]*='; then
                HAS_REGISTRY_RELEVANT_CHANGES=true
                break
            fi
        fi
    done <<< "$CHANGED_FILES"
fi

HAS_REGISTRY_UPDATE=false
while IFS= read -r file; do
    if [[ "$file" == "REGISTRY.md" ]]; then
        HAS_REGISTRY_UPDATE=true
        break
    fi
done <<< "$CHANGED_FILES"

if [ "$HAS_REGISTRY_RELEVANT_CHANGES" = true ] && [ "$HAS_REGISTRY_UPDATE" = false ]; then
    ERRORS="$ERRORS\n  - REGISTRY.md was not updated but worker files affecting the registry (entrypoint.py or Dockerfile LABELs) were modified. Run \`python3 generate_worker_docs.py --registry-only\` and commit the result."
fi

if [ -n "$ERRORS" ]; then
    echo "Worker documentation validation failed:"
    echo -e "$ERRORS"
    echo ""
    echo "All workers must have a WORKERNAME.md documentation file, and REGISTRY.md must be updated when worker files change."
    exit 2
fi

exit 0
