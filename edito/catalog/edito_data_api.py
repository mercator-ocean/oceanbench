# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
from collections.abc import Iterator
from dataclasses import dataclass
from http import HTTPStatus

import requests

EDITO_DATA_API_URL = "https://api.dive.edito.eu/data"
PROJECT_CATALOGS_URL = f"{EDITO_DATA_API_URL}/catalogs/projects"
COLLECTIONS_URL = f"{EDITO_DATA_API_URL}/collections"
TOKEN_URL = "https://auth.dive.edito.eu/auth/realms/datalab/protocol/openid-connect/token"
TOKEN_CLIENT_IDENTIFIER = "edito"
ACCESS_TOKEN_VARIABLE = "EDITO_ACCESS_TOKEN"
OFFLINE_TOKEN_VARIABLE = "EDITO_OFFLINE_TOKEN"
ITEMS_PAGE_SIZE = 500
REQUEST_TIMEOUT_SECONDS = 120
FORCE_PARAMETERS = {"_force": "true"}


class EditoDataApiError(Exception):
    pass


@dataclass(frozen=True)
class EditoSession:
    access_token: str


def session_from_environment() -> EditoSession:
    access_token = os.environ.get(ACCESS_TOKEN_VARIABLE)
    if access_token:
        return EditoSession(access_token)
    offline_token = os.environ.get(OFFLINE_TOKEN_VARIABLE)
    if not offline_token:
        raise EditoDataApiError(f"Set {ACCESS_TOKEN_VARIABLE} or {OFFLINE_TOKEN_VARIABLE} to publish the catalog")
    return EditoSession(_access_token_from_offline_token(offline_token))


def _access_token_from_offline_token(offline_token: str) -> str:
    response = requests.post(
        TOKEN_URL,
        data={"client_id": TOKEN_CLIENT_IDENTIFIER, "grant_type": "refresh_token", "refresh_token": offline_token},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    _raise_for_failure(response, "Exchange the offline token")
    return response.json()["access_token"]


def collection_url(collection_identifier: str) -> str:
    return f"{COLLECTIONS_URL}/{collection_identifier}"


def upsert_collection_in_project_catalog(session: EditoSession, catalog_identifier: str, collection: dict) -> None:
    _upsert(session, collection_url(collection["id"]), f"{PROJECT_CATALOGS_URL}/{catalog_identifier}", collection)


def upsert_item(session: EditoSession, collection_identifier: str, item: dict) -> None:
    items_url = f"{collection_url(collection_identifier)}/items"
    creation_response = _upsert(session, f"{items_url}/{item['id']}", items_url, item)
    if creation_response is not None and creation_response.json().get("inError", 0) > 0:
        raise EditoDataApiError(f"Create item {item['id']} failed: {creation_response.json().get('errors')}")


def ensure_project_catalog(session: EditoSession, catalog: dict) -> None:
    catalog_url = f"{PROJECT_CATALOGS_URL}/{catalog['id']}"
    if _request(session, "GET", catalog_url).status_code != HTTPStatus.NOT_FOUND:
        return
    _raise_for_failure(_request(session, "POST", PROJECT_CATALOGS_URL, json=catalog), f"Create {catalog_url}")


def upsert_project_catalog(session: EditoSession, catalog: dict, visibility: tuple[str, ...]) -> None:
    catalog_url = f"{PROJECT_CATALOGS_URL}/{catalog['id']}"
    _upsert(session, catalog_url, PROJECT_CATALOGS_URL, catalog)
    _raise_for_failure(
        _request(session, "PUT", catalog_url, json={"visibility": list(visibility)}, params=FORCE_PARAMETERS),
        f"Set visibility of {catalog_url}",
    )


def item_identifiers(session: EditoSession, collection_identifier: str) -> tuple[str, ...]:
    return tuple(
        feature["properties"]["productIdentifier"]
        for page in _item_pages(session, f"{collection_url(collection_identifier)}/items", {"limit": ITEMS_PAGE_SIZE})
        for feature in page.get("features", [])
    )


def delete_item(session: EditoSession, collection_identifier: str, identifier: str) -> None:
    item_url = f"{collection_url(collection_identifier)}/items/{identifier}"
    _raise_for_failure(_request(session, "DELETE", item_url), f"Delete {item_url}")


def _item_pages(session: EditoSession, url: str, parameters: dict | None) -> Iterator[dict]:
    response = _request(session, "GET", url, params=parameters)
    if response.status_code == HTTPStatus.NOT_FOUND:
        return
    _raise_for_failure(response, f"List {url}")
    page = response.json()
    yield page
    next_url = next((link["href"] for link in page.get("links", []) if link.get("rel") == "next"), None)
    if next_url and page.get("features"):
        yield from _item_pages(session, next_url, None)


def _upsert(session: EditoSession, resource_url: str, creation_url: str, resource: dict) -> requests.Response | None:
    update_response = _request(session, "PUT", resource_url, json=resource, params=FORCE_PARAMETERS)
    if update_response.ok:
        return None
    creation_response = _request(session, "POST", creation_url, json=resource)
    _raise_for_failure(
        creation_response,
        f"Update {resource_url} ({update_response.status_code}: {update_response.text[:500]}) then create it",
    )
    return creation_response


def _request(session: EditoSession, method: str, url: str, **keyword_arguments) -> requests.Response:
    return requests.request(
        method,
        url,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {session.access_token}",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
        **keyword_arguments,
    )


def _raise_for_failure(response: requests.Response, action: str) -> None:
    if not response.ok:
        raise EditoDataApiError(f"{action} failed with {response.status_code}: {response.text[:1000]}")
