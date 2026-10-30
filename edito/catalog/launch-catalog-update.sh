#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

set -euo pipefail

readonly API_URL="https://datalab.dive.edito.eu/api"
readonly CATALOG_ID="core-processes"
readonly PACKAGE_NAME="generic"
readonly PACKAGE_VERSION="1.1.1"
readonly SOURCE_TYPE="My code is based on a Python or Shell script"
readonly TARGET_TYPE="My process should execute on EDITO Cloud"
readonly ENTRYPOINT_PATH="edito/catalog/update-catalog.sh"
readonly ENVIRONMENT_PATH="edito/catalog/environment.yaml"
readonly MAIN_BRANCH="main"
readonly MAIN_PROCESS_NAME="oceanbench-catalog"
readonly ONYXIA_PROJECT="${ONYXIA_PROJECT:-project-oceanbench}"
readonly ONYXIA_REGION="WAW3-1"
readonly TOKEN_URL="https://auth.dive.edito.eu/auth/realms/datalab/protocol/openid-connect/token"
readonly TOKEN_CLIENT_ID="edito"
readonly PROCESS_NAME_MAXIMUM_LENGTH=40
readonly LAUNCH_ATTEMPTS=6
readonly LAUNCH_RETRY_INTERVAL_SECONDS=10

readonly BRANCH="${BRANCH:-$(git branch --show-current)}"
readonly GIT_REPOSITORY="${GIT_REPOSITORY:-https://github.com/mercator-ocean/oceanbench.git}"
readonly GIT_TOKEN="${GIT_TOKEN:-}"
readonly EDITO_ACCESS_TOKEN="${EDITO_ACCESS_TOKEN:-}"
readonly EDITO_OFFLINE_TOKEN="${EDITO_OFFLINE_TOKEN:-}"
readonly CPU_REQUEST="${CPU_REQUEST:-1000m}"
readonly MEMORY_REQUEST="${MEMORY_REQUEST:-4Gi}"
readonly CPU_LIMIT="${CPU_LIMIT:-4000m}"
readonly MEMORY_LIMIT="${MEMORY_LIMIT:-8Gi}"

normalized_branch() {
    echo "${BRANCH}" \
        | tr '[:upper:]' '[:lower:]' \
        | tr -c 'a-z0-9\n' '-'
}

catalog_environment() {
    case "${BRANCH}" in
        "${MAIN_BRANCH}") echo "" ;;
        *) normalized_branch ;;
    esac
}

process_name() {
    case "${BRANCH}" in
        "${MAIN_BRANCH}") echo "${MAIN_PROCESS_NAME}" ;;
        *) echo "${MAIN_PROCESS_NAME}-$(normalized_branch)" | cut -c1-"${PROCESS_NAME_MAXIMUM_LENGTH}" | sed 's/-*$//' ;;
    esac
}

readonly PROCESS_NAME="$(process_name)"

access_token() {
    case "${EDITO_ACCESS_TOKEN}:${EDITO_OFFLINE_TOKEN}" in
        ":") echo "Set EDITO_ACCESS_TOKEN or EDITO_OFFLINE_TOKEN to launch ${PROCESS_NAME}" >&2; return 1 ;;
        ":"*)
            curl -fsS -X POST "${TOKEN_URL}" \
                -H "Content-Type: application/x-www-form-urlencoded" \
                -d "client_id=${TOKEN_CLIENT_ID}" \
                -d "grant_type=refresh_token" \
                --data-urlencode "refresh_token=${EDITO_OFFLINE_TOKEN}" \
                | jq -r '.access_token'
            ;;
        *) echo "${EDITO_ACCESS_TOKEN}" ;;
    esac
}

launch_request_body() {
    local token="$1"
    jq -n \
        --arg catalogId "${CATALOG_ID}" \
        --arg packageName "${PACKAGE_NAME}" \
        --arg packageVersion "${PACKAGE_VERSION}" \
        --arg name "${PROCESS_NAME}" \
        --arg friendlyName "OceanBench catalog ${BRANCH}" \
        --arg source "${SOURCE_TYPE}" \
        --arg repository "${GIT_REPOSITORY}" \
        --arg branch "${BRANCH}" \
        --arg gitToken "${GIT_TOKEN}" \
        --arg entrypointPath "${ENTRYPOINT_PATH}" \
        --arg environmentPath "${ENVIRONMENT_PATH}" \
        --arg target "${TARGET_TYPE}" \
        --arg cpuRequest "${CPU_REQUEST}" \
        --arg memoryRequest "${MEMORY_REQUEST}" \
        --arg cpuLimit "${CPU_LIMIT}" \
        --arg memoryLimit "${MEMORY_LIMIT}" \
        --arg accessToken "${token}" \
        --arg catalogEnvironment "$(catalog_environment)" \
        '{
            catalogId: $catalogId,
            packageName: $packageName,
            packageVersion: $packageVersion,
            name: $name,
            friendlyName: $friendlyName,
            share: true,
            options: {
                catalogType: "Process",
                configuration: {
                    source: $source,
                    git: {name: "", email: "", cache: "0", token: $gitToken, repository: $repository, branch: $branch},
                    containerImage: {imageIsBuilt: false, tag: "", path: ""},
                    script: {entrypointPath: $entrypointPath, environmentPath: $environmentPath}
                },
                target: {
                    type: $target,
                    gpu: {enabled: false},
                    edito: {
                        requests: {cpu: $cpuRequest, memory: $memoryRequest},
                        limits: {cpu: $cpuLimit, memory: $memoryLimit}
                    }
                },
                s3: {enabled: false},
                copernicusMarine: {enabled: false},
                processInputs: [
                    {name: "EDITO_ACCESS_TOKEN", value: $accessToken},
                    {name: "OCEANBENCH_CATALOG_ENVIRONMENT", value: $catalogEnvironment}
                ]
            },
            dryRun: false
        }'
}

my_lab_request() {
    local token="$1"
    shift
    curl --fail-with-body -sS \
        -H "Authorization: Bearer ${token}" \
        -H "ONYXIA-PROJECT: ${ONYXIA_PROJECT}" \
        -H "ONYXIA-REGION: ${ONYXIA_REGION}" \
        -H "Content-Type: application/json" \
        "$@"
}

delete_previous_process() {
    local token="$1"
    my_lab_request "${token}" -X DELETE "${API_URL}/my-lab/app?path=${PROCESS_NAME}" > /dev/null || true
}

launch_process() {
    local token="$1"
    local attempt
    for attempt in $(seq 1 "${LAUNCH_ATTEMPTS}"); do
        if my_lab_request "${token}" -X PUT "${API_URL}/my-lab/app" -d "$(launch_request_body "${token}")"; then
            echo
            return 0
        fi
        echo "Launch attempt ${attempt} failed, previous ${PROCESS_NAME} may still be uninstalling" >&2
        sleep "${LAUNCH_RETRY_INTERVAL_SECONDS}"
    done
    return 1
}

main() {
    local token
    token="$(access_token)"
    echo "Launching ${PROCESS_NAME} on branch ${BRANCH} with catalog environment '$(catalog_environment)'"
    delete_previous_process "${token}"
    launch_process "${token}"
}

main
