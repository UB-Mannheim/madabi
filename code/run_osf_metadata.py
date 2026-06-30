"""
Fetch OSF metadata in ONE pass (resource metadata + file inventory + contributors),
and write compact JSON outputs without OSF relationship-graph clutter.

Output (default)
- data/from_papers/osf_metadata/osf_<id>.json
- data/from_papers/osf_metadata/manifest.csv

Input (default)
- data/from_papers/extracted_repositories.csv  (expects columns including: repository,url)

"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


OSF_ID_RE = re.compile(r"^/([0-9a-z]{5})(?:/|$)", re.IGNORECASE)
OSF_FILES_SUBID_RE = re.compile(r"/files/([0-9a-z]{5})(?:/|$)", re.IGNORECASE)
OSF_DOI_RE = re.compile(r"10\.17605/OSF\.IO/([0-9a-z]{5})", re.IGNORECASE)
OSF_USER_RE = re.compile(r"/v2/users/([^/?]+)/?(?:\\?.*)?$", re.IGNORECASE)


DATA_EXTENSIONS = {
    "csv",
    "xlsx",
    "dta",
    "sav",
    "tsv",
    "tab",
    "xls",
    "ods",
    "por",
    "json",
    "jsonl",
    "ndjson",
    "parquet",
    "feather",
    "avro",
    "sqlite",
    "db",
    "h5",
    "hdf5",
    "nc",
    "sas7bdat",
    "xpt",
}

ARCHIVE_EXTENSIONS = {
    "zip",
    "7z",
    "rar",
    "tar",
    "gz",
    "tgz",
    "bz2",
    "xz",
}


def extract_osf_id(url: str) -> Optional[str]:
    u = (url or "").strip()
    if not u:
        return None

    m = OSF_DOI_RE.search(u)
    if m:
        return m.group(1).lower()

    try:
        from urllib.parse import urlparse

        p = urlparse(u)
        if p.netloc.lower().endswith("osf.io"):
            mf = OSF_FILES_SUBID_RE.search(p.path or "")
            if mf:
                return mf.group(1).lower()
            m2 = OSF_ID_RE.match(p.path or "")
            if m2:
                return m2.group(1).lower()
    except Exception:
        return None

    return None


def osf_get_json(api_url: str, view_only: str | None, timeout: int) -> requests.Response:
    params: dict[str, str] = {}
    if view_only:
        params["view_only"] = view_only
    # Use separate connect/read timeouts so slow OSF responses can use a higher read timeout
    # while keeping connect failures fast.
    connect_timeout = min(30, max(5, int(timeout)))
    read_timeout = max(connect_timeout, int(timeout))
    return requests.get(api_url, params=params, timeout=(connect_timeout, read_timeout))


@dataclass(frozen=True)
class OSFResult:
    status: str  # ok | not_found | forbidden | error
    kind: str  # node | registration | preprint | file | guid_* | unknown
    api_url: str
    error: str = ""
    data: Any = None


def fetch_osf(id5: str, view_only: str | None, timeout: int) -> OSFResult:
    base = "https://api.osf.io/v2"
    guid_url = f"{base}/guids/{id5}/"
    try:
        guid_resp = osf_get_json(guid_url, view_only=view_only, timeout=timeout)
    except requests.RequestException as e:
        return OSFResult(status="error", kind="guid", api_url=guid_url, error=f"request_error: {e.__class__.__name__}: {e}")

    if guid_resp.status_code == 200:
        try:
            guid_json = guid_resp.json()
        except Exception as e:
            return OSFResult(status="error", kind="guid", api_url=guid_url, error=f"json_decode_error: {e}")

        data = (guid_json or {}).get("data") or {}
        resolved_type = (data.get("type") or "unknown").rstrip("s")
        resolved_id = data.get("id")

        follow: list[tuple[str, str]] = []
        if resolved_type == "node":
            follow = [("node", f"{base}/nodes/{id5}/")]
        elif resolved_type == "registration":
            follow = [("registration", f"{base}/registrations/{id5}/")]
        elif resolved_type == "preprint":
            follow = [("preprint", f"{base}/preprints/{id5}/")]
        elif resolved_type == "file" and resolved_id:
            follow = [("file", f"{base}/files/{resolved_id}/")]

        if not follow:
            return OSFResult(status="ok", kind=f"guid_{resolved_type}", api_url=guid_url, data=guid_json)

        kind2, url2 = follow[0]
        try:
            resp2 = osf_get_json(url2, view_only=view_only, timeout=timeout)
        except requests.RequestException:
            return OSFResult(status="ok", kind=f"guid_{resolved_type}", api_url=guid_url, data=guid_json)

        if resp2.status_code == 200:
            try:
                full_json = resp2.json()
            except Exception:
                return OSFResult(status="ok", kind=f"guid_{resolved_type}", api_url=guid_url, data=guid_json)
            return OSFResult(status="ok", kind=kind2, api_url=url2, data={"guid": guid_json, "resolved": full_json})

        if resp2.status_code in (401, 403):
            return OSFResult(status="forbidden", kind=kind2, api_url=url2, error=f"http_{resp2.status_code}")
        if resp2.status_code == 410:
            return OSFResult(status="error", kind=kind2, api_url=url2, error="http_410")

        return OSFResult(status="ok", kind=f"guid_{resolved_type}", api_url=guid_url, data=guid_json)

    if guid_resp.status_code in (401, 403):
        return OSFResult(status="forbidden", kind="guid", api_url=guid_url, error=f"http_{guid_resp.status_code}")
    if guid_resp.status_code == 410:
        return OSFResult(status="error", kind="guid", api_url=guid_url, error="http_410")
    if guid_resp.status_code != 404:
        return OSFResult(status="error", kind="guid", api_url=guid_url, error=f"http_{guid_resp.status_code}")

    return OSFResult(status="not_found", kind="unknown", api_url=guid_url, error="http_404")


def _get_nested(d: dict[str, Any], path: list[str]) -> Any:
    cur: Any = d
    for p in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _ext(name: str) -> str:
    n = (name or "").strip().lower()
    if not n or "." not in n:
        return ""
    return n.rsplit(".", 1)[-1]


def list_osf_collection(
    href: str, *, view_only: str | None, timeout: int, delay: float, max_pages: int = 500
) -> tuple[list[dict[str, Any]], str | None]:
    items: list[dict[str, Any]] = []
    url = href
    pages = 0
    while url and pages < max_pages:
        pages += 1
        # OSF can be intermittently slow; retry transient request errors a few times.
        last_exc: Exception | None = None
        resp: requests.Response | None = None
        for attempt in range(3):
            try:
                resp = osf_get_json(url, view_only=view_only, timeout=timeout)
                last_exc = None
                break
            except requests.RequestException as e:
                last_exc = e
                # brief backoff (0.5s, 1s) between retries
                if attempt < 2:
                    time.sleep(0.5 * (2**attempt))
        if last_exc is not None or resp is None:
            return items, f"request_error: {last_exc.__class__.__name__}: {last_exc}"
        if resp.status_code != 200:
            return items, f"http_{resp.status_code}"
        try:
            j = resp.json()
        except Exception as e:
            return items, f"json_decode_error: {e}"
        items.extend((j.get("data") or []) if isinstance(j, dict) else [])
        url = _get_nested(j, ["links", "next"])
        if delay > 0:
            time.sleep(delay)
    if pages >= max_pages:
        return items, f"max_pages_exceeded:{max_pages}"
    return items, None


def get_files_root_hrefs(resolved_json: dict[str, Any]) -> list[str]:
    href = _get_nested(resolved_json, ["data", "relationships", "files", "links", "related", "href"])
    return [href] if isinstance(href, str) and href else []


def get_resources_href(resolved_json: dict[str, Any]) -> str | None:
    href = _get_nested(resolved_json, ["data", "relationships", "resources", "links", "related", "href"])
    return href if isinstance(href, str) and href else None


def traverse_files_tree(
    resolved_json: dict[str, Any],
    *,
    view_only: str | None,
    timeout: int,
    delay: float,
    max_items: int,
) -> tuple[dict[str, Any], str | None]:
    roots = get_files_root_hrefs(resolved_json)
    if not roots:
        return {"has_files": False, "reason": "no_files_relationship"}, None

    provider_items, err = list_osf_collection(roots[0], view_only=view_only, timeout=timeout, delay=delay)
    if err:
        return {"has_files": False, "reason": "providers_list_failed"}, err

    provider_hrefs: list[str] = []
    for p in provider_items:
        href = _get_nested(p, ["relationships", "files", "links", "related", "href"])
        if isinstance(href, str) and href:
            provider_hrefs.append(href)

    if not provider_hrefs:
        return {"has_files": False, "reason": "no_provider_file_hrefs"}, None

    total_files = 0
    total_folders = 0
    total_bytes = 0
    extensions_count: dict[str, int] = {}
    data_files = 0
    data_ext_count: dict[str, int] = {}
    archive_files = 0
    archive_ext_count: dict[str, int] = {}
    sample_archive_files: list[dict[str, Any]] = []
    sample_data_files: list[dict[str, Any]] = []

    queue: list[str] = list(dict.fromkeys(provider_hrefs))
    visited: set[str] = set()
    items_seen = 0

    while queue:
        href = queue.pop(0)
        if href in visited:
            continue
        visited.add(href)

        listing, lerr = list_osf_collection(href, view_only=view_only, timeout=timeout, delay=delay)
        if lerr:
            return (
                {
                    "has_files": total_files > 0,
                    "partial": True,
                    "total_files": total_files,
                    "total_folders": total_folders,
                    "total_bytes": total_bytes,
                    "extensions_count": extensions_count,
                    "data_files": data_files,
                    "data_ext_count": data_ext_count,
                    "sample_data_files": sample_data_files,
                },
                f"list_failed:{lerr}",
            )

        for it in listing:
            items_seen += 1
            if items_seen > max_items:
                return (
                    {
                        "has_files": total_files > 0,
                        "partial": True,
                        "reason": f"max_items_exceeded:{max_items}",
                        "total_files": total_files,
                        "total_folders": total_folders,
                        "total_bytes": total_bytes,
                        "extensions_count": extensions_count,
                        "data_files": data_files,
                        "data_ext_count": data_ext_count,
                        "sample_data_files": sample_data_files,
                    },
                    None,
                )

            attrs = it.get("attributes") or {}
            kind = (attrs.get("kind") or "").lower()
            name = attrs.get("name") or ""

            if kind == "folder":
                total_folders += 1
                child_href = _get_nested(it, ["relationships", "files", "links", "related", "href"])
                if isinstance(child_href, str) and child_href:
                    queue.append(child_href)
                continue

            total_files += 1
            size = attrs.get("size")
            if isinstance(size, int):
                total_bytes += size
            ext = _ext(name)
            if ext:
                extensions_count[ext] = extensions_count.get(ext, 0) + 1
                if ext in DATA_EXTENSIONS:
                    data_files += 1
                    data_ext_count[ext] = data_ext_count.get(ext, 0) + 1
                    if len(sample_data_files) < 25:
                        sample_data_files.append({"name": name, "ext": ext, "size": size if isinstance(size, int) else None, "id": it.get("id")})
                if ext in ARCHIVE_EXTENSIONS:
                    archive_files += 1
                    archive_ext_count[ext] = archive_ext_count.get(ext, 0) + 1
                    if len(sample_archive_files) < 25:
                        sample_archive_files.append({"name": name, "ext": ext, "size": size if isinstance(size, int) else None, "id": it.get("id")})

    return (
        {
            "has_files": total_files > 0,
            "partial": False,
            "total_files": total_files,
            "total_folders": total_folders,
            "total_bytes": total_bytes,
            "extensions_count": extensions_count,
            "data_files": data_files,
            "has_data_files": data_files > 0,
            "data_ext_count": data_ext_count,
            "sample_data_files": sample_data_files,
            "archive_files": archive_files,
            "has_archive_files": archive_files > 0,
            "archive_ext_count": archive_ext_count,
            "sample_archive_files": sample_archive_files,
            "has_potential_data_files": (data_files > 0) or (archive_files > 0),
        },
        None,
    )


def _merge_inventory_summaries(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    a = a or {}
    b = b or {}

    out["partial"] = bool(a.get("partial")) or bool(b.get("partial"))

    for k in ("total_files", "total_folders", "total_bytes", "data_files", "archive_files"):
        av = a.get(k) if isinstance(a.get(k), int) else 0
        bv = b.get(k) if isinstance(b.get(k), int) else 0
        out[k] = av + bv

    def merge_counts(x: Any, y: Any) -> dict[str, int]:
        outc: dict[str, int] = {}
        if isinstance(x, dict):
            for kk, vv in x.items():
                if isinstance(vv, int):
                    outc[kk] = outc.get(kk, 0) + vv
        if isinstance(y, dict):
            for kk, vv in y.items():
                if isinstance(vv, int):
                    outc[kk] = outc.get(kk, 0) + vv
        return outc

    out["extensions_count"] = merge_counts(a.get("extensions_count"), b.get("extensions_count"))
    out["data_ext_count"] = merge_counts(a.get("data_ext_count"), b.get("data_ext_count"))
    out["archive_ext_count"] = merge_counts(a.get("archive_ext_count"), b.get("archive_ext_count"))

    def merge_samples(x: Any, y: Any, limit: int = 25) -> list[dict[str, Any]]:
        outl: list[dict[str, Any]] = []
        for src in (x, y):
            if not isinstance(src, list):
                continue
            for item in src:
                if not isinstance(item, dict):
                    continue
                outl.append(item)
                if len(outl) >= limit:
                    return outl
        return outl

    out["sample_data_files"] = merge_samples(a.get("sample_data_files"), b.get("sample_data_files"))
    out["sample_archive_files"] = merge_samples(a.get("sample_archive_files"), b.get("sample_archive_files"))

    out["has_files"] = bool(out.get("total_files", 0) > 0)
    out["has_data_files"] = bool(out.get("data_files", 0) > 0)
    out["has_archive_files"] = bool(out.get("archive_files", 0) > 0)
    out["has_potential_data_files"] = bool(out.get("has_data_files") or out.get("has_archive_files"))

    # Preserve traversal metadata when present (e.g., OSF registrations' Resources → Data pids).
    if "resource_traversal" in a or "resource_traversal" in b:
        out["resource_traversal"] = a.get("resource_traversal") or b.get("resource_traversal")
    if "reason" in a or "reason" in b:
        out["reason"] = a.get("reason") or b.get("reason")
    return out


def traverse_files_tree_with_resources(
    resolved_json: dict[str, Any],
    *,
    view_only: str | None,
    timeout: int,
    delay: float,
    max_items: int,
) -> tuple[dict[str, Any], str | None]:
    base_summary, base_err = traverse_files_tree(
        resolved_json, view_only=view_only, timeout=timeout, delay=delay, max_items=max_items
    )
    if isinstance(base_summary, dict) and base_summary.get("has_files"):
        return base_summary, base_err

    resources_href = get_resources_href(resolved_json)
    if not resources_href:
        return base_summary, base_err

    resources, rerr = list_osf_collection(resources_href, view_only=view_only, timeout=timeout, delay=delay)
    if rerr:
        return base_summary, (base_err or "") + (f"; resources_list_failed:{rerr}" if rerr else "")

    merged = dict(base_summary) if isinstance(base_summary, dict) else {"has_files": False}
    merged.setdefault("resource_traversal", {"checked": [], "errors": []})

    for res in resources:
        attrs = (res or {}).get("attributes") or {}
        rtype = (attrs.get("resource_type") or "").strip().lower()
        pid = (attrs.get("pid") or "").strip()
        if rtype != "data" or not pid:
            continue

        rid = extract_osf_id(pid)
        if not rid:
            merged["resource_traversal"]["errors"].append({"pid": pid, "error": "cannot_extract_osf_id"})
            continue

        merged["resource_traversal"]["checked"].append({"resource_type": rtype, "pid": pid, "osf_id": rid})

        rresult = fetch_osf(rid, view_only=view_only, timeout=timeout)
        if rresult.status != "ok" or not isinstance(rresult.data, dict):
            merged["resource_traversal"]["errors"].append({"pid": pid, "osf_id": rid, "status": rresult.status, "error": rresult.error})
            continue

        rresolved = rresult.data.get("resolved") if "resolved" in rresult.data else rresult.data
        if not isinstance(rresolved, dict):
            merged["resource_traversal"]["errors"].append({"pid": pid, "osf_id": rid, "error": "no_resolved_json"})
            continue

        rsum, _ = traverse_files_tree(rresolved, view_only=view_only, timeout=timeout, delay=delay, max_items=max_items)
        if isinstance(rsum, dict):
            merged = _merge_inventory_summaries(merged, rsum)

    return merged, base_err


def extract_user_id_from_href(href: str | None) -> str | None:
    if not href:
        return None
    m = OSF_USER_RE.search(href)
    return m.group(1) if m else None


def get_contributors_href(resolved_json: dict[str, Any]) -> str | None:
    rel = (resolved_json.get("data") or {}).get("relationships") or {}
    return ((rel.get("contributors") or {}).get("links") or {}).get("related", {}).get("href")


def fetch_osf_contributors(
    contributors_href: str,
    *,
    view_only: str | None,
    timeout: int,
    delay: float,
) -> dict[str, Any]:
    try:
        resp = osf_get_json(contributors_href, view_only=view_only, timeout=timeout)
    except requests.RequestException as e:
        return {"contributors_href": contributors_href, "error": f"request_error: {e.__class__.__name__}: {e}"}
    if resp.status_code != 200:
        return {"contributors_href": contributors_href, "error": f"http_{resp.status_code}"}
    try:
        contrib_json = resp.json()
    except Exception as e:
        return {"contributors_href": contributors_href, "error": f"json_decode_error: {e}"}

    out: dict[str, Any] = {"contributors_href": contributors_href, "contributors": [], "users": {}}

    for c in contrib_json.get("data", []) or []:
        attrs = c.get("attributes") or {}
        rel = c.get("relationships") or {}
        user_href = ((rel.get("users") or {}).get("links") or {}).get("related", {}).get("href")
        # Prefer relationship data id (stable even with view_only query params)
        user_id = _get_nested(rel, ["users", "data", "id"]) or extract_user_id_from_href(user_href)
        out["contributors"].append(
            {
                "permission": attrs.get("permission"),
                "bibliographic": attrs.get("bibliographic"),
                "user_id": user_id,
            }
        )

    user_ids = sorted({c.get("user_id") for c in out["contributors"] if c.get("user_id")})
    for user_id in user_ids:
        try:
            uresp = osf_get_json(f"https://api.osf.io/v2/users/{user_id}/", view_only=view_only, timeout=timeout)
        except requests.RequestException as e:
            out["users"][user_id] = {"error": f"request_error: {e.__class__.__name__}: {e}"}
            continue
        if uresp.status_code != 200:
            out["users"][user_id] = {"error": f"http_{uresp.status_code}"}
            continue
        try:
            uj = uresp.json()
        except Exception as e:
            out["users"][user_id] = {"error": f"json_decode_error: {e}"}
            continue
        data = (uj.get("data") or {}) if isinstance(uj, dict) else {}
        a = (data.get("attributes") or {}) if isinstance(data, dict) else {}
        ext = a.get("external_identity") if isinstance(a, dict) else None
        orcid = ""
        if isinstance(ext, dict):
            orcid_block = ext.get("ORCID")
            if isinstance(orcid_block, dict):
                orcid = str(orcid_block.get("id") or "").strip()
            elif isinstance(orcid_block, str):
                orcid = orcid_block.strip()

        employment = a.get("employment") if isinstance(a, dict) else None
        education = a.get("education") if isinstance(a, dict) else None
        social = a.get("social") if isinstance(a, dict) else None

        # Keep external_identity as-is (can include ORCID, Twitter, etc.) but avoid None vs dict confusion.
        external_identity = ext if isinstance(ext, dict) else None
        out["users"][user_id] = {
            "full_name": a.get("full_name"),
            "family_name": a.get("family_name"),
            "given_name": a.get("given_name"),
            "active": a.get("active"),
            "orcid": orcid,
            "employment": employment,
            "education": education,
            "social": social,
            "external_identity": external_identity,
        }
        if delay > 0:
            time.sleep(delay)

    return out


def _compact_resource(resolved_json: dict[str, Any] | None, api_url: str | None) -> dict[str, Any]:
    if not isinstance(resolved_json, dict):
        return {"api_url": api_url or ""}

    data = (resolved_json.get("data") or {}) if isinstance(resolved_json.get("data"), dict) else {}
    attrs = (data.get("attributes") or {}) if isinstance(data.get("attributes"), dict) else {}
    links = (data.get("links") or {}) if isinstance(data.get("links"), dict) else {}
    rel = (data.get("relationships") or {}) if isinstance(data.get("relationships"), dict) else {}

    title = attrs.get("title") or attrs.get("name") or ""
    description = attrs.get("description") or ""

    dates: dict[str, Any] = {}
    for k in ("date_created", "date_modified", "date_published", "date_registered", "registration_supplement"):
        v = attrs.get(k)
        if v:
            dates[k] = v

    identifiers_href = _get_nested(rel, ["identifiers", "links", "related", "href"])
    license_href = _get_nested(rel, ["license", "links", "related", "href"])

    node_license = attrs.get("node_license") if isinstance(attrs, dict) else None

    return {
        "id": data.get("id") or "",
        "type": data.get("type") or "",
        "title": title,
        "description": description,
        "category": attrs.get("category"),
        "public": attrs.get("public"),
        "tags": attrs.get("tags") if isinstance(attrs.get("tags"), list) else None,
        "dates": dates,
        "urls": {"api": api_url or "", "html": links.get("html") or ""},
        "identifiers_href": identifiers_href or "",
        "license_href": license_href or "",
        "node_license": node_license if isinstance(node_license, dict) else None,
    }


def fetch_license(license_href: str, *, view_only: str | None, timeout: int, delay: float) -> tuple[dict[str, Any] | None, str | None]:
    """
    Fetch OSF license details from the license endpoint.
    Returns (license_dict, error_string).
    """
    if not license_href:
        return None, None
    try:
        resp = osf_get_json(license_href, view_only=view_only, timeout=timeout)
    except requests.RequestException as e:
        return None, f"request_error:{e.__class__.__name__}:{e}"
    if resp.status_code != 200:
        return None, f"http_{resp.status_code}"
    try:
        j = resp.json()
    except Exception as e:
        return None, f"json_decode_error:{e}"

    data = (j.get("data") or {}) if isinstance(j, dict) else {}
    attrs = (data.get("attributes") or {}) if isinstance(data, dict) else {}
    out = {
        "id": data.get("id"),
        "name": attrs.get("name"),
        "text": attrs.get("text"),
        "url": attrs.get("url"),
    }
    if delay > 0:
        time.sleep(delay)
    return out, None


def fetch_identifiers(
    identifiers_href: str, *, view_only: str | None, timeout: int, delay: float
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not identifiers_href:
        return None, None
    try:
        resp = osf_get_json(identifiers_href, view_only=view_only, timeout=timeout)
    except requests.RequestException as e:
        return None, f"request_error:{e.__class__.__name__}:{e}"
    if resp.status_code != 200:
        return None, f"http_{resp.status_code}"
    try:
        j = resp.json()
    except Exception as e:
        return None, f"json_decode_error:{e}"

    out: list[dict[str, Any]] = []
    for item in (j.get("data") or []) if isinstance(j, dict) else []:
        if not isinstance(item, dict):
            continue
        attrs = item.get("attributes") or {}
        if not isinstance(attrs, dict):
            continue
        cat = attrs.get("category")
        val = attrs.get("value")
        if cat or val:
            out.append({"category": cat, "value": val})

    if delay > 0:
        time.sleep(delay)
    return out, None


def _compact_file_inventory(inv: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(inv, dict):
        return None
    keys = [
        "has_files",
        "total_files",
        "total_folders",
        "total_bytes",
        "partial",
        "has_data_files",
        "data_files",
        "data_ext_count",
        "sample_data_files",
        "has_archive_files",
        "archive_files",
        "archive_ext_count",
        "sample_archive_files",
        "has_potential_data_files",
        # Present when we had to follow OSF "Resources" (e.g., registrations' Data resources)
        "resource_traversal",
        "reason",
    ]
    out: dict[str, Any] = {}
    for k in keys:
        if k in inv:
            out[k] = inv.get(k)
    return out


def _compact_contributors(contrib_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(contrib_payload, dict):
        return None
    if contrib_payload.get("error"):
        return {"error": contrib_payload.get("error"), "contributors_href": contrib_payload.get("contributors_href") or ""}

    contribs = contrib_payload.get("contributors")
    users = contrib_payload.get("users")
    if not isinstance(contribs, list):
        contribs = []
    if not isinstance(users, dict):
        users = {}

    people: list[dict[str, Any]] = []
    for c in contribs:
        if not isinstance(c, dict):
            continue
        user_id = c.get("user_id")
        u = users.get(user_id) if user_id else None
        person = {
            "user_id": user_id,
            "permission": c.get("permission"),
            "bibliographic": c.get("bibliographic"),
        }
        if isinstance(u, dict):
            for k in (
                "full_name",
                "given_name",
                "family_name",
                "active",
                "orcid",
                "employment",
                "education",
                "social",
                "external_identity",
            ):
                if k in u:
                    person[k] = u.get(k)
        people.append(person)

    return {
        "contributors_href": contrib_payload.get("contributors_href") or "",
        "n_contributors": len(people),
        "people": people,
    }


def load_osf_urls(csv_path: Path) -> list[tuple[str, str, str | None]]:
    seen: set[tuple[str, str | None]] = set()
    out: list[tuple[str, str, str | None]] = []
    view_only_re = re.compile(r"[?&]view_only=([0-9a-f]{32})", re.IGNORECASE)

    # Use utf-8-sig so CSVs written by PowerShell/Excel with a BOM still parse "repository" correctly.
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("repository", "").strip() != "OSF":
                continue
            raw = (row.get("url") or "").strip()
            osf_id = extract_osf_id(raw)
            if not osf_id:
                continue
            m = view_only_re.search(raw)
            token = m.group(1) if m else None
            key = (osf_id, token)
            if key in seen:
                continue
            seen.add(key)
            out.append((raw, osf_id, token))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch compact OSF metadata (resource metadata + files + contributors).")
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("data/from_papers/extracted_repositories_2026-06.csv"),
        help="CSV from analyze_from_papers (repository,url,...).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/from_papers/osf_metadata"),
        help="Output directory for compact per-id JSON + manifest.csv.",
    )
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds (default 60).")
    ap.add_argument("--delay", type=float, default=0.2, help="Delay seconds between requests (default 0.2).")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N ids (0 = all).")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if output JSON already exists.")
    ap.add_argument(
        "--fetch-identifiers",
        action="store_true",
        help="Fetch and store identifiers (e.g., DOI) via resource.identifiers_href.",
    )
    ap.add_argument(
        "--only-potential-data-files",
        action="store_true",
        help="Only write outputs where file_inventory.has_potential_data_files is true.",
    )
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"Input CSV not found: {args.input.resolve()}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.csv"
    write_header = not manifest_path.exists() or manifest_path.stat().st_size == 0

    rows = load_osf_urls(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]

    fieldnames = [
        "osf_id",
        "original_url",
        "html_url",
        "api_url",
        "status",
        "error",
        "inventory_error",
        "contributors_error",
        "identifiers_error",
        "contributors_url",
        "identifiers_url",
        "kind",
        "title",
        "public",
        "doi",
        "has_potential_data_files",
        "total_files",
        "total_bytes",
        "n_contributors",
        "output_json",
    ]

    with manifest_path.open("a", newline="", encoding="utf-8") as mf:
        w = csv.DictWriter(mf, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        for original_url, osf_id, token in tqdm(rows, desc="OSF metadata"):
            out_path = args.out_dir / f"osf_{osf_id}.json"
            out_rel = out_path.as_posix()

            if args.skip_existing and out_path.exists():
                continue

            result = fetch_osf(osf_id, view_only=token, timeout=args.timeout)
            kind = result.kind
            api_url = result.api_url

            resolved = None
            if result.status == "ok" and isinstance(result.data, dict):
                resolved = result.data.get("resolved") if "resolved" in result.data else result.data

            resource = _compact_resource(resolved if isinstance(resolved, dict) else None, api_url=api_url)

            inventory_summary: dict[str, Any] | None = None
            inventory_error: str | None = None
            if isinstance(resolved, dict):
                inventory_summary, inventory_error = traverse_files_tree_with_resources(
                    resolved,
                    view_only=token,
                    timeout=args.timeout,
                    delay=args.delay,
                    max_items=25000,
                )

            file_inventory = _compact_file_inventory(inventory_summary)
            if inventory_error and isinstance(file_inventory, dict):
                file_inventory["inventory_error"] = inventory_error

            if args.only_potential_data_files:
                flag = (file_inventory or {}).get("has_potential_data_files") if isinstance(file_inventory, dict) else False
                if not bool(flag):
                    continue

            contributors_href = get_contributors_href(resolved) if isinstance(resolved, dict) else None
            raw_contrib = None
            if contributors_href:
                raw_contrib = fetch_osf_contributors(
                    contributors_href,
                    view_only=token,
                    timeout=args.timeout,
                    delay=args.delay,
                )
            contributors = _compact_contributors(raw_contrib)

            identifiers = None
            identifiers_error = None
            doi = ""
            if args.fetch_identifiers:
                identifiers, identifiers_error = fetch_identifiers(
                    resource.get("identifiers_href") or "",
                    view_only=token,
                    timeout=args.timeout,
                    delay=args.delay,
                )
                if isinstance(identifiers, list):
                    for it in identifiers:
                        if not isinstance(it, dict):
                            continue
                        if str(it.get("category") or "").lower() == "doi" and it.get("value"):
                            doi = str(it.get("value"))
                            break

            # License details (best effort; many records won't have a license relationship)
            license_details = None
            license_error = None
            if isinstance(resource, dict) and resource.get("license_href"):
                license_details, license_error = fetch_license(
                    str(resource.get("license_href") or ""),
                    view_only=token,
                    timeout=args.timeout,
                    delay=args.delay,
                )

            combined = {
                "osf_id": osf_id,
                "original_url": original_url,
                "kind": kind,
                "view_only": token,
                "status": result.status,
                "error": result.error,
                "resource": resource,
                "identifiers": identifiers,
                "identifiers_error": identifiers_error,
                "license": license_details,
                "license_error": license_error,
                "file_inventory": file_inventory,
                "contributors": contributors,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "data_extensions": sorted(DATA_EXTENSIONS),
                "archive_extensions": sorted(ARCHIVE_EXTENSIONS),
            }

            out_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

            w.writerow(
                {
                    "osf_id": osf_id,
                    "original_url": original_url,
                    "html_url": (resource.get("urls") or {}).get("html") if isinstance(resource.get("urls"), dict) else "",
                    "api_url": (resource.get("urls") or {}).get("api") if isinstance(resource.get("urls"), dict) else api_url,
                    "status": result.status,
                    "error": result.error,
                    "inventory_error": inventory_error or "",
                    "contributors_error": (raw_contrib or {}).get("error") if isinstance(raw_contrib, dict) else "",
                    "identifiers_error": identifiers_error or "",
                    "contributors_url": (contributors or {}).get("contributors_href") if isinstance(contributors, dict) else "",
                    "identifiers_url": resource.get("identifiers_href") or "",
                    "kind": kind,
                    "title": resource.get("title") or "",
                    "public": resource.get("public"),
                    "doi": doi,
                    "has_potential_data_files": (file_inventory or {}).get("has_potential_data_files")
                    if isinstance(file_inventory, dict)
                    else "",
                    "total_files": (file_inventory or {}).get("total_files") if isinstance(file_inventory, dict) else "",
                    "total_bytes": (file_inventory or {}).get("total_bytes") if isinstance(file_inventory, dict) else "",
                    "n_contributors": (contributors or {}).get("n_contributors") if isinstance(contributors, dict) else "",
                    "output_json": out_rel,
                }
            )
            mf.flush()

            if args.delay > 0:
                time.sleep(args.delay)

    print(f"Wrote OSF metadata outputs to: {args.out_dir.resolve()}")
    print(f"Wrote manifest: {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

