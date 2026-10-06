#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

set -euo pipefail

readonly API_URL="https://datalab.dive.edito.eu/api"
readonly CATALOG_ID="data-visualization"
readonly PACKAGE_NAME="static-pages"
readonly PACKAGE_VERSION="1.0.9"
readonly BUILD_SCRIPT_PATH="website/build-website.sh"
readonly MAIN_BRANCH="main"
readonly MAIN_SERVICE_NAME="oceanbench"
readonly ONYXIA_PROJECT="${ONYXIA_PROJECT:-project-oceanbench}"
readonly ONYXIA_REGION="WAW3-1"
readonly TOKEN_URL="https://auth.dive.edito.eu/auth/realms/datalab/protocol/openid-connect/token"
readonly TOKEN_CLIENT_ID="edito"
readonly UPDATE_TIMEOUT_SECONDS=30
readonly CURL_TIMEOUT_EXIT_CODE=28
readonly HEALTH_CHECK_ATTEMPTS=60
readonly HEALTH_CHECK_INTERVAL_SECONDS=10
readonly SERVICE_NAME_MAXIMUM_LENGTH=40

readonly BRANCH="${BRANCH:-$(git branch --show-current)}"
readonly GIT_REPOSITORY="${GIT_REPOSITORY:-https://github.com/mercator-ocean/oceanbench.git}"
readonly GIT_TOKEN="${GIT_TOKEN:-}"
readonly EDITO_ACCESS_TOKEN="${EDITO_ACCESS_TOKEN:-}"
readonly EDITO_OFFLINE_TOKEN="${EDITO_OFFLINE_TOKEN:-}"
readonly EXPECTED_CONTENT="${EXPECTED_CONTENT:-OceanBench}"
readonly CPU_REQUEST="${CPU_REQUEST:-100m}"
readonly MEMORY_REQUEST="${MEMORY_REQUEST:-1Gi}"
readonly CPU_LIMIT="${CPU_LIMIT:-1000m}"
readonly MEMORY_LIMIT="${MEMORY_LIMIT:-4Gi}"

branch_service_name() {
    echo "oceanbench-${BRANCH}" \
        | tr '[:upper:]' '[:lower:]' \
        | tr -c 'a-z0-9\n' '-' \
        | cut -c1-"${SERVICE_NAME_MAXIMUM_LENGTH}" \
        | sed 's/-*$//'
}

default_service_name() {
    case "${BRANCH}" in
        "${MAIN_BRANCH}") echo "${MAIN_SERVICE_NAME}" ;;
        *) branch_service_name ;;
    esac
}

readonly SERVICE_NAME="${SERVICE_NAME:-$(default_service_name)}"
readonly SERVICE_HOSTNAME="${SERVICE_NAME}.lab.dive.edito.eu"
readonly SERVICE_URL="https://${SERVICE_HOSTNAME}"

access_token_from_offline_token() {
    curl -fsS -X POST "${TOKEN_URL}" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "client_id=${TOKEN_CLIENT_ID}" \
        -d "grant_type=refresh_token" \
        --data-urlencode "refresh_token=${EDITO_OFFLINE_TOKEN}" \
        -d "scope=openid" \
        | jq -r '.access_token'
}

access_token() {
    case "${EDITO_ACCESS_TOKEN}:${EDITO_OFFLINE_TOKEN}" in
        ":") echo "Service ${SERVICE_NAME} does not exist: set EDITO_ACCESS_TOKEN or EDITO_OFFLINE_TOKEN to create it" >&2; return 1 ;;
        ":"*) access_token_from_offline_token ;;
        *) echo "${EDITO_ACCESS_TOKEN}" ;;
    esac
}

launch_request_body() {
    jq -n \
        --arg catalogId "${CATALOG_ID}" \
        --arg packageName "${PACKAGE_NAME}" \
        --arg packageVersion "${PACKAGE_VERSION}" \
        --arg name "${SERVICE_NAME}" \
        --arg friendlyName "OceanBench ${BRANCH}" \
        --arg repository "${GIT_REPOSITORY}" \
        --arg branch "${BRANCH}" \
        --arg token "${GIT_TOKEN}" \
        --arg hostname "${SERVICE_HOSTNAME}" \
        --arg buildScriptPath "${BUILD_SCRIPT_PATH}" \
        --arg cpuRequest "${CPU_REQUEST}" \
        --arg memoryRequest "${MEMORY_REQUEST}" \
        --arg cpuLimit "${CPU_LIMIT}" \
        --arg memoryLimit "${MEMORY_LIMIT}" \
        '{
            catalogId: $catalogId,
            packageName: $packageName,
            packageVersion: $packageVersion,
            name: $name,
            friendlyName: $friendlyName,
            share: true,
            options: {
                resources: {requests: {cpu: $cpuRequest, memory: $memoryRequest}, limits: {cpu: $cpuLimit, memory: $memoryLimit}},
                git: {enabled: true, cache: "0", token: $token, repository: $repository, branch: $branch},
                website: {source: ""},
                ingress: {enabled: true, hostname: $hostname},
                environment_variables: [],
                build: {scriptPath: $buildScriptPath, args: ""},
                catalogType: "Service"
            },
            dryRun: false
        }'
}

launch_service() {
    local token="$1"
    curl --fail-with-body -sS -X PUT "${API_URL}/my-lab/app" \
        -H "Authorization: Bearer ${token}" \
        -H "ONYXIA-PROJECT: ${ONYXIA_PROJECT}" \
        -H "ONYXIA-REGION: ${ONYXIA_REGION}" \
        -H "Content-Type: application/json" \
        -d "$(launch_request_body)"
    echo
}

create_service() {
    local token
    token="$(access_token)"
    echo "Creating ${SERVICE_NAME} on branch ${BRANCH}"
    launch_service "${token}"
}

update_service() {
    local exit_code=0
    curl -fsS --max-time "${UPDATE_TIMEOUT_SECONDS}" "${SERVICE_URL}/update" > /dev/null || exit_code=$?
    case "${exit_code}" in
        0 | "${CURL_TIMEOUT_EXIT_CODE}") return 0 ;;
        *) return 1 ;;
    esac
}

create_or_update_service() {
    if update_service; then
        echo "Updated existing ${SERVICE_NAME}"
        return 0
    fi
    create_service
}

serves_expected_content() {
    local page
    page="$(curl -fsS --max-time 10 "${SERVICE_URL}/" 2> /dev/null)" && grep -qF -- "${EXPECTED_CONTENT}" <<< "${page}"
}

wait_until_healthy() {
    local attempt
    for attempt in $(seq 1 "${HEALTH_CHECK_ATTEMPTS}"); do
        if serves_expected_content; then
            echo "Healthy: ${SERVICE_URL} serves branch ${BRANCH}"
            return 0
        fi
        echo "Attempt ${attempt}/${HEALTH_CHECK_ATTEMPTS}: ${SERVICE_URL} not ready"
        sleep "${HEALTH_CHECK_INTERVAL_SECONDS}"
    done
    echo "Unhealthy: ${SERVICE_URL} never served the expected content" >&2
    return 1
}

create_or_update_service
wait_until_healthy
