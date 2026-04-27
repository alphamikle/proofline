from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

from proofline.utils import safe_read_text, stable_id, json_dumps, normalize_name

HTTP_METHODS = {"get", "post", "put", "patch", "delete", "head", "options"}

ROUTE_PATTERNS = [
    # Python FastAPI/Flask decorators
    ("python_route", re.compile(r"@(?:app|router|bp)\.(get|post|put|patch|delete|head|options)\(['\"]([^'\"]+)['\"]", re.I)),
    ("express_route", re.compile(r"(?:app|router)\.(get|post|put|patch|delete|head|options)\(['\"]([^'\"]+)['\"]", re.I)),
    # Java/Kotlin Spring
    ("spring_route", re.compile(r"@(GetMapping|PostMapping|PutMapping|PatchMapping|DeleteMapping|RequestMapping)\s*(?:\(\s*)?(?:value\s*=\s*)?[\"']([^\"']+)[\"']", re.I)),
    # NestJS style
    ("nestjs_route", re.compile(r"@(Get|Post|Put|Patch|Delete)\(['\"]([^'\"]+)['\"]", re.I)),
    # Go: router.GET("/path", handler)
    ("go_route", re.compile(r"\.(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\(['\"]([^'\"]+)['\"]", re.I)),
]


def parse_api_specs(repo_inventory: pd.DataFrame, repo_files: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    contracts: List[Dict[str, Any]] = []
    endpoints: List[Dict[str, Any]] = []
    service_by_repo = {r["repo_id"]: normalize_name(r["repo_id"]) for _, r in repo_inventory.iterrows()}
    api_files = repo_files[repo_files["kind"].isin(["api_contract"])] if not repo_files.empty else pd.DataFrame()
    for _, f in api_files.iterrows():
        repo_id = str(f["repo_id"])
        service_id = service_by_repo.get(repo_id, normalize_name(repo_id))
        path = Path(str(f["path"]))
        rel = str(f["rel_path"])
        text = safe_read_text(path)
        if not text:
            continue
        low = rel.lower()
        if path.suffix.lower() in {".yaml", ".yml", ".json"} or "openapi" in low or "swagger" in low or "asyncapi" in low:
            contract_type = "asyncapi" if "asyncapi" in low else "openapi"
            contract_id = stable_id(repo_id, rel, contract_type)
            contracts.append({
                "contract_id": contract_id, "service_id": service_id, "repo_id": repo_id,
                "contract_type": contract_type, "source_file": rel, "docs_url": rel,
                "commit_sha": "", "confidence": 0.85,
            })
            try:
                obj = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
            except Exception:
                obj = None
            if isinstance(obj, dict):
                paths = obj.get("paths") or {}
                if isinstance(paths, dict):
                    for pth, methods in paths.items():
                        if not isinstance(methods, dict):
                            continue
                        for method, spec in methods.items():
                            if method.lower() not in HTTP_METHODS:
                                continue
                            spec = spec if isinstance(spec, dict) else {}
                            op = spec.get("operationId") or ""
                            req = json_dumps(spec.get("requestBody", {}))
                            resp = json_dumps(spec.get("responses", {}))
                            endpoints.append({
                                "endpoint_id": stable_id(service_id, method.upper(), pth, contract_id),
                                "service_id": service_id, "repo_id": repo_id, "contract_id": contract_id,
                                "method": method.upper(), "path": str(pth), "operation_id": op,
                                "request_schema": req, "response_schema": resp, "source_file": rel,
                                "source": contract_type, "confidence": 0.9,
                            })
        elif path.suffix.lower() == ".proto":
            contract_id = stable_id(repo_id, rel, "protobuf")
            contracts.append({
                "contract_id": contract_id, "service_id": service_id, "repo_id": repo_id,
                "contract_type": "protobuf", "source_file": rel, "docs_url": rel,
                "commit_sha": "", "confidence": 0.8,
            })
            for svc in re.finditer(r"service\s+(\w+)\s*\{(?P<body>.*?)\n\}", text, re.S):
                svc_name = svc.group(1)
                body = svc.group("body")
                for rpc in re.finditer(r"rpc\s+(\w+)\s*\(([^)]+)\)\s*returns\s*\(([^)]+)\)", body):
                    rpc_name = rpc.group(1)
                    endpoints.append({
                        "endpoint_id": stable_id(service_id, "RPC", svc_name, rpc_name, rel),
                        "service_id": service_id, "repo_id": repo_id, "contract_id": contract_id,
                        "method": "RPC", "path": f"/{svc_name}/{rpc_name}", "operation_id": rpc_name,
                        "request_schema": rpc.group(2), "response_schema": rpc.group(3),
                        "source_file": rel, "source": "protobuf", "confidence": 0.85,
                    })
        elif path.suffix.lower() == ".graphql" or "schema.graphql" in low:
            contract_id = stable_id(repo_id, rel, "graphql")
            contracts.append({
                "contract_id": contract_id, "service_id": service_id, "repo_id": repo_id,
                "contract_type": "graphql", "source_file": rel, "docs_url": rel,
                "commit_sha": "", "confidence": 0.75,
            })
            for m in re.finditer(r"type\s+(Query|Mutation)\s*\{(?P<body>.*?)\}", text, re.S):
                kind = m.group(1).upper()
                for field in re.finditer(r"^\s*(\w+)\s*(?:\([^)]*\))?\s*:", m.group("body"), re.M):
                    name = field.group(1)
                    endpoints.append({
                        "endpoint_id": stable_id(service_id, kind, name, rel),
                        "service_id": service_id, "repo_id": repo_id, "contract_id": contract_id,
                        "method": kind, "path": f"/{name}", "operation_id": name,
                        "request_schema": "", "response_schema": "", "source_file": rel,
                        "source": "graphql", "confidence": 0.75,
                    })
    return pd.DataFrame(contracts), pd.DataFrame(endpoints)


def extract_static_routes(repo_inventory: pd.DataFrame, repo_files: pd.DataFrame) -> pd.DataFrame:
    endpoints: List[Dict[str, Any]] = []
    service_by_repo = {r["repo_id"]: normalize_name(r["repo_id"]) for _, r in repo_inventory.iterrows()}
    for _, f in repo_files.iterrows():
        if str(f.get("kind")) not in {"source", "source_route_hint"}:
            continue
        rel = str(f["rel_path"])
        ext = Path(rel).suffix.lower()
        if ext not in {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".kt", ".go", ".rb", ".cs"}:
            continue
        text = safe_read_text(Path(str(f["path"])))
        if not text:
            continue
        repo_id = str(f["repo_id"])
        service_id = service_by_repo.get(repo_id, normalize_name(repo_id))
        for source, pat in ROUTE_PATTERNS:
            for m in pat.finditer(text):
                method_raw, path = m.group(1), m.group(2)
                method = method_raw.upper().replace("MAPPING", "").replace("GET", "GET").replace("POST", "POST")
                if method == "REQUEST":
                    method = "ANY"
                line = text[:m.start()].count("\n") + 1
                endpoints.append({
                    "endpoint_id": stable_id(service_id, method, path, rel, line),
                    "service_id": service_id, "repo_id": repo_id, "contract_id": "",
                    "method": method, "path": path, "operation_id": "",
                    "request_schema": "", "response_schema": "", "source_file": rel,
                    "source": source, "confidence": 0.65,
                })
    return pd.DataFrame(endpoints)
