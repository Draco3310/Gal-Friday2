#!/bin/bash
# Credential storage helper

set -euo pipefail

store_credential() {
    local service="$1"
    local account="$2"
    local secret="$3"

    case "$(uname -s)" in
        Linux*)
            if command -v secret-tool &>/dev/null; then
                echo "$secret" | secret-tool store --label="$service" service "$service" account "$account"
            else
                echo "No keyring available"
                return 1
            fi
            ;;
        Darwin*)
            security add-generic-password -a "$account" -s "$service" -w "$secret" -U
            ;;
        MINGW*|CYGWIN*|MSYS*)
            cmdkey /generic:"$service" /user:"$account" /pass:"$secret"
            ;;
    esac
}

retrieve_credential() {
    local service="$1"
    local account="$2"

    case "$(uname -s)" in
        Linux*)
            if command -v secret-tool &>/dev/null; then
                secret-tool lookup service "$service" account "$account"
            else
                return 1
            fi
            ;;
        Darwin*)
            security find-generic-password -a "$account" -s "$service" -w
            ;;
        MINGW*|CYGWIN*|MSYS*)
            # Windows doesn't have a simple CLI retrieval
            echo "Use Credential Manager GUI to view credentials"
            return 1
            ;;
    esac
}

# Main
case "${1:-}" in
    store)
        store_credential "$2" "$3" "$4"
        ;;
    retrieve)
        retrieve_credential "$2" "$3"
        ;;
    *)
        echo "Usage: $0 {store|retrieve} service account [secret]"
        exit 1
        ;;
esac
