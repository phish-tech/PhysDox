"""
Unified LLM API call layer.

Supports any OpenAI-compatible API (MIMO, OpenAI, DeepSeek, etc.).
Handles reasoning models (content in reasoning_content field).
"""

import json
import re
import urllib.error
import urllib.request

from hallucination_benchmark.config import get_model_config


def llm_call(
    system_prompt: str,
    user_prompt: str,
    *,
    model_key: str = None,
    model_name_override: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    timeout: int = 180,
) -> tuple[str | None, str]:
    """Call an OpenAI-compatible LLM API.

    Returns (content, reasoning).
    - content: the model's main response (may be empty for reasoning models)
    - reasoning: the reasoning_content field (empty for non-reasoning models)

    If content is empty but reasoning exists, content falls back to reasoning.
    """
    cfg = get_model_config(model_key, model_name_override=model_name_override)
    api_key = cfg["api_key"]
    base_url = cfg["base_url"].rstrip("/")
    model = cfg["model"]
    api_style = cfg.get("api_style", "chat_completions")

    if not api_key:
        return None, f"API key not set for model '{model}'"

    headers = {"Content-Type": "application/json"}
    if api_style == "anthropic_messages":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        endpoint = f"{base_url}/v1/messages"
        payload = {
            "model": model,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    else:
        headers["Authorization"] = f"Bearer {api_key}"

    if api_style == "responses":
        endpoint = f"{base_url}/responses"
        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
    elif api_style == "chat_completions":
        endpoint = f"{base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    elif api_style != "anthropic_messages":
        return None, f"Unsupported API style: {api_style}"

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        if api_style == "anthropic_messages":
            content, reasoning = _parse_anthropic_messages_result(result)
        elif api_style == "responses":
            content, reasoning = _parse_responses_result(result)
        else:
            msg = result["choices"][0]["message"]
            content = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or ""

        # Reasoning models: if content is empty, fall back to reasoning
        if not content and reasoning:
            content = reasoning

        return content, reasoning

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        detail = f"HTTP Error {e.code}: {e.reason}"
        if body:
            detail = f"{detail}; body={body[:2000]}"
        return None, detail
    except Exception as e:
        return None, str(e)


def _parse_responses_result(result: dict) -> tuple[str, str]:
    """Extract text from OpenAI Responses-compatible API output."""
    output_text = result.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text, ""

    content_parts = []
    reasoning_parts = []
    for item in result.get("output", []) or []:
        item_type = item.get("type")
        if item_type == "reasoning":
            for summary in item.get("summary", []) or []:
                text = summary.get("text")
                if text:
                    reasoning_parts.append(text)
            continue
        if item_type != "message":
            continue
        for part in item.get("content", []) or []:
            text = part.get("text")
            if not text:
                continue
            if part.get("type") in {"output_text", "text"}:
                content_parts.append(text)
            else:
                content_parts.append(text)

    content = "\n".join(content_parts)
    reasoning = "\n".join(reasoning_parts)
    return content or reasoning, reasoning


def _parse_anthropic_messages_result(result: dict) -> tuple[str, str]:
    """Extract text from Anthropic Messages API output."""
    text_parts = []
    for part in result.get("content", []) or []:
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
            text_parts.append(part["text"])
    return "\n".join(text_parts), ""


def extract_json(text: str) -> dict | list | None:
    """Extract JSON from LLM output, handling markdown code fences."""
    if not text:
        return None

    # Strip code fences
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Try whole text first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try finding a JSON object
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try finding a JSON array
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def llm_json_call(
    system_prompt: str,
    user_prompt: str,
    *,
    model_key: str = None,
    model_name_override: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    timeout: int = 180,
) -> dict | list | None:
    """Convenience: call LLM and parse JSON response.

    Tries content first, then reasoning content.
    """
    content, reasoning = llm_call(
        system_prompt,
        user_prompt,
        model_key=model_key,
        model_name_override=model_name_override,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    result = extract_json(content)
    if result is not None:
        return result

    # Try reasoning content
    result = extract_json(reasoning)
    if result is not None:
        return result

    return None
