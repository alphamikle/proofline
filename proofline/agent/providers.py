from __future__ import annotations

import os
import subprocess
from typing import Any, Callable, Dict, List
from urllib.parse import urljoin

import requests


class AgentProviderError(RuntimeError):
    pass


AgentEventCallback = Callable[[str, Dict[str, Any]], None]


def complete_with_agent(
    system_prompt: str,
    user_prompt: str,
    cfg: Dict[str, Any],
    *,
    agent_name: str | None = None,
    on_event: AgentEventCallback | None = None,
) -> str | None:
    agents = agent_candidates(cfg, agent_name=agent_name)
    if not agents:
        return None
    errors: List[str] = []
    attempted = False
    for idx, agent in enumerate(agents):
        name = str(agent.get("name") or f"agent-{idx + 1}")
        provider = str(agent.get("provider", "none") or "none").lower()
        if provider in {"", "none"}:
            continue
        attempted = True
        if on_event:
            on_event("agent_attempt", {"name": name, "provider": provider, "model": agent.get("model")})
        try:
            text = _complete_with_single_agent(system_prompt, user_prompt, agent)
        except Exception as e:
            errors.append(f"{name}: {e}")
            if on_event:
                on_event("agent_error", {"name": name, "provider": provider, "error": str(e)})
            continue
        if text:
            if on_event:
                on_event("agent_success", {"name": name, "provider": provider, "model": agent.get("model")})
            return text
        errors.append(f"{name}: returned no text")
        if on_event:
            on_event("agent_empty", {"name": name, "provider": provider})
    if not attempted:
        return None
    if errors:
        raise AgentProviderError("All configured agents failed: " + "; ".join(errors))
    return None


def agent_candidates(cfg: Dict[str, Any], *, agent_name: str | None = None) -> List[Dict[str, Any]]:
    root = dict(cfg.get("agent") or {})
    configured = root.get("agents")
    if isinstance(configured, dict):
        configured = [{"name": name, **(profile or {})} for name, profile in configured.items()]
    if not isinstance(configured, list) or not configured:
        legacy = {k: v for k, v in root.items() if k not in {"active", "agents", "fallback"}}
        if "name" not in legacy:
            legacy["name"] = str(root.get("active") or "default")
        return [legacy]

    common = {k: v for k, v in root.items() if k not in {"active", "agents", "fallback"}}
    profiles: List[Dict[str, Any]] = []
    for idx, item in enumerate(configured):
        if not isinstance(item, dict):
            continue
        merged = dict(common)
        merged.update(item)
        merged.setdefault("name", f"agent-{idx + 1}")
        profiles.append(merged)

    selected = str(agent_name or os.getenv("PROOFLINE_AGENT") or root.get("active") or "").strip()
    fallback_enabled = bool(root.get("fallback", True))
    if selected:
        selected_profiles = [p for p in profiles if str(p.get("name") or "") == selected]
        if not selected_profiles:
            raise AgentProviderError(f"Configured agent not found: {selected}")
        if not fallback_enabled:
            return selected_profiles
        rest = [p for p in profiles if str(p.get("name") or "") != selected]
        return selected_profiles + rest
    return profiles if fallback_enabled else profiles[:1]


def _complete_with_single_agent(system_prompt: str, user_prompt: str, agent: Dict[str, Any]) -> str | None:
    provider = str(agent.get("provider", "none") or "none").lower()
    if provider == "command":
        provider = "cli"
    if provider == "none":
        return None
    if provider == "cli":
        return _complete_cli(system_prompt, user_prompt, agent)
    if provider == "openai":
        return _complete_openai_responses(system_prompt, user_prompt, agent)
    if provider == "openai_compatible":
        return _complete_openai_compatible(system_prompt, user_prompt, agent)
    if provider in {"anthropic", "anthropic_compatible"}:
        return _complete_anthropic_messages(system_prompt, user_prompt, agent, provider)
    raise AgentProviderError(f"Unsupported agent.provider: {provider}")


def _complete_cli(system_prompt: str, user_prompt: str, agent: Dict[str, Any]) -> str | None:
    command = agent.get("command")
    if not command:
        return None
    prompt = system_prompt + "\n\n" + user_prompt
    timeout = int(agent.get("request_timeout_seconds") or 600)
    p = subprocess.run(
        command,
        input=prompt,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    if p.returncode == 0 and p.stdout.strip():
        return p.stdout.strip()
    return None


def _complete_openai_responses(system_prompt: str, user_prompt: str, agent: Dict[str, Any]) -> str | None:
    api_key = _configured_env(agent, "api_key_env", "OPENAI_API_KEY")
    if not api_key:
        return None
    model = agent.get("model")
    if not model:
        return None
    base_url = _base_url(agent, "OPENAI_BASE_URL", "https://api.openai.com/v1")
    payload: Dict[str, Any] = {
        "model": model,
        "instructions": system_prompt,
        "input": user_prompt,
    }
    _add_common_generation_params(payload, agent, max_tokens_key="max_output_tokens")
    data = _post_json(
        urljoin(base_url.rstrip("/") + "/", "responses"),
        payload,
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        agent,
    )
    return _extract_openai_responses_text(data)


def _complete_openai_compatible(system_prompt: str, user_prompt: str, agent: Dict[str, Any]) -> str | None:
    api_key = _configured_env(agent, "api_key_env", "OPENAI_API_KEY")
    model = agent.get("model")
    base_url = _base_url(agent, "OPENAI_BASE_URL", None)
    if not model or not base_url:
        return None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    _add_common_generation_params(payload, agent, max_tokens_key="max_tokens")
    data = _post_json(
        urljoin(base_url.rstrip("/") + "/", "chat/completions"),
        payload,
        headers,
        agent,
    )
    choices = data.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    reasoning = message.get("reasoning_content")
    return reasoning.strip() if isinstance(reasoning, str) and reasoning.strip() else None


def _complete_anthropic_messages(
    system_prompt: str,
    user_prompt: str,
    agent: Dict[str, Any],
    provider: str,
) -> str | None:
    default_key_env = "ANTHROPIC_API_KEY"
    default_base = "https://api.anthropic.com/v1" if provider == "anthropic" else None
    api_key = _configured_env(agent, "api_key_env", default_key_env)
    model = agent.get("model")
    base_url = _base_url(agent, "ANTHROPIC_BASE_URL", default_base)
    if not api_key or not model or not base_url:
        return None
    payload: Dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "max_tokens": int(agent.get("max_output_tokens") or 4096),
    }
    temperature = agent.get("temperature")
    if temperature is not None:
        payload["temperature"] = float(temperature)
    data = _post_json(
        urljoin(base_url.rstrip("/") + "/", "messages"),
        payload,
        {
            "x-api-key": api_key,
            "anthropic-version": str(agent.get("anthropic_version") or "2023-06-01"),
            "Content-Type": "application/json",
        },
        agent,
    )
    parts = []
    for block in data.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
            parts.append(str(block["text"]))
    text = "\n".join(parts).strip()
    return text or None


def _post_json(url: str, payload: Dict[str, Any], headers: Dict[str, str], agent: Dict[str, Any]) -> Dict[str, Any]:
    timeout = int(agent.get("request_timeout_seconds") or 600)
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _add_common_generation_params(payload: Dict[str, Any], agent: Dict[str, Any], max_tokens_key: str) -> None:
    temperature = agent.get("temperature")
    if temperature is not None:
        payload["temperature"] = float(temperature)
    max_output_tokens = agent.get("max_output_tokens")
    if max_output_tokens is not None:
        payload[max_tokens_key] = int(max_output_tokens)


def _extract_openai_responses_text(data: Dict[str, Any]) -> str | None:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    parts = []
    for item in data.get("output") or []:
        for content in item.get("content") or []:
            text = content.get("text") if isinstance(content, dict) else None
            if text:
                parts.append(str(text))
    text = "\n".join(parts).strip()
    return text or None


def _base_url(agent: Dict[str, Any], env_name: str, default: str | None) -> str | None:
    env_value = _configured_env(agent, "base_url_env", env_name)
    return str(agent.get("base_url") or env_value or default or "").strip() or None


def _configured_env(agent: Dict[str, Any], key: str, default_name: str) -> str | None:
    configured = agent.get(key)
    if configured == "":
        return None
    return _env(str(configured or default_name))


def _env(name: str | None) -> str | None:
    if not name:
        return None
    value = os.environ.get(str(name))
    return value.strip() if value else None
