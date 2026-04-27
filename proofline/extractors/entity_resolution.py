from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from proofline.utils import normalize_name, stable_id, json_dumps


def build_service_identity(
    repo_inventory: pd.DataFrame,
    datadog_services: pd.DataFrame,
    datadog_edges: pd.DataFrame,
    ownership: pd.DataFrame,
    api_endpoints: pd.DataFrame,
    static_edges: pd.DataFrame,
    bq_usage: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    aliases: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    owner_by_repo = {}
    if ownership is not None and not ownership.empty:
        for _, o in ownership.iterrows():
            ent = str(o.get("entity_id") or "")
            if ent.startswith("repo:") and str(o.get("owner_team") or ""):
                owner_by_repo[ent.replace("repo:", "")] = str(o.get("owner_team"))
    dd_names = set()
    if datadog_services is not None and not datadog_services.empty:
        dd_names.update(str(x) for x in datadog_services["datadog_service"].dropna().unique() if str(x) and str(x) != "__error__")
    if datadog_edges is not None and not datadog_edges.empty:
        dd_names.update(str(x) for x in datadog_edges["from_service"].dropna().unique() if str(x) and str(x) != "__error__")
        dd_names.update(str(x) for x in datadog_edges["to_service"].dropna().unique() if str(x) and str(x) and str(x) != "__error__")
    norm_to_dd = {}
    for dd in dd_names:
        norm_to_dd.setdefault(normalize_name(dd), []).append(dd)

    for _, r in repo_inventory.iterrows() if repo_inventory is not None and not repo_inventory.empty else []:
        repo_id = str(r.get("repo_id") or "")
        norm = normalize_name(repo_id)
        candidates = norm_to_dd.get(norm, [])
        # fuzzy prefix/suffix pass
        if not candidates:
            for n, names in norm_to_dd.items():
                if n and norm and (n == norm or n.startswith(norm) or norm.startswith(n)):
                    candidates.extend(names)
        dd = candidates[0] if candidates else ""
        conf = 0.9 if dd and normalize_name(dd) == norm else (0.65 if dd else 0.55)
        service_id = norm or repo_id
        api_docs = ""
        if api_endpoints is not None and not api_endpoints.empty:
            docs = api_endpoints[api_endpoints["repo_id"] == repo_id]["source_file"].dropna().unique().tolist()
            api_docs = json_dumps(docs[:20])
        rows.append({
            "service_id": service_id,
            "display_name": repo_id,
            "repo_id": repo_id,
            "repo_path": str(r.get("repo_path") or ""),
            "datadog_service": dd,
            "owner_team": owner_by_repo.get(repo_id, ""),
            "api_docs": api_docs,
            "confidence": conf,
            "evidence_refs": json_dumps(["repo_inventory", "datadog_name_match" if dd else "repo_name_only"]),
        })
        aliases.append({"canonical_id": f"service:{service_id}", "alias": repo_id, "alias_type": "repo_name", "source": "repo_inventory", "confidence": 0.8})
        if dd:
            aliases.append({"canonical_id": f"service:{service_id}", "alias": dd, "alias_type": "datadog_service", "source": "datadog", "confidence": conf})
        else:
            unresolved.append({"entity": repo_id, "entity_type": "repo", "reason": "no_datadog_service_match", "confidence": conf})
    # Datadog-only services become service identities too.
    known_aliases = {str(a["alias"]).lower() for a in aliases}
    for dd in sorted(dd_names):
        if dd.lower() in known_aliases:
            continue
        sid = normalize_name(dd)
        rows.append({
            "service_id": sid,
            "display_name": dd,
            "repo_id": "",
            "repo_path": "",
            "datadog_service": dd,
            "owner_team": "",
            "api_docs": "[]",
            "confidence": 0.6,
            "evidence_refs": json_dumps(["datadog_only"]),
        })
        aliases.append({"canonical_id": f"service:{sid}", "alias": dd, "alias_type": "datadog_service", "source": "datadog", "confidence": 0.7})
        unresolved.append({"entity": dd, "entity_type": "datadog_service", "reason": "no_repo_match", "confidence": 0.6})

    # service account candidates from BQ usage.
    if bq_usage is not None and not bq_usage.empty:
        for sa in sorted(set(str(x) for x in bq_usage["service_account"].dropna().unique() if str(x))):
            sid = guess_service_from_service_account(sa)
            aliases.append({"canonical_id": f"service:{sid}", "alias": sa, "alias_type": "service_account", "source": "bq_usage_naming", "confidence": 0.45})
    return pd.DataFrame(rows), pd.DataFrame(aliases), pd.DataFrame(unresolved)


def guess_service_from_service_account(sa: str) -> str:
    local = sa.split("@")[0]
    local = re.sub(r"^(svc|service|sa)[-_]", "", local)
    local = re.sub(r"[-_](prod|production|staging|dev|test)$", "", local)
    return normalize_name(local)


def canonicalize(raw: str, aliases: pd.DataFrame) -> str:
    if not raw:
        return ""
    raw_l = raw.lower()
    if aliases is None or aliases.empty:
        return raw
    hit = aliases[aliases["alias"].str.lower() == raw_l]
    if not hit.empty:
        return str(hit.iloc[0]["canonical_id"])
    norm = normalize_name(raw)
    hit2 = aliases[aliases["alias"].apply(lambda x: normalize_name(str(x)) == norm)]
    if not hit2.empty:
        return str(hit2.iloc[0]["canonical_id"])
    return raw
