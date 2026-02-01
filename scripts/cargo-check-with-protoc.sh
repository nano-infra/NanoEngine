#!/bin/bash
# Wrapper for cargo check that uses local protoc if available.
# Used by pre-commit for NanoDeploy (requires protoc for etcd-client).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROTOC="${SCRIPT_DIR}/.bin/protoc"

if [[ -x "$LOCAL_PROTOC" ]] && [[ -z "$PROTOC" ]]; then
    export PROTOC="$LOCAL_PROTOC"
fi
exec cargo "$@"
