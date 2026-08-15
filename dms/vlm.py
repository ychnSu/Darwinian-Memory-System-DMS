"""OpenAI-compatible client for vLLM served through the local SSH tunnel."""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from urllib.request import Request, urlopen


@dataclass
class TokenUsage:
  prompt_tokens: int = 0
  completion_tokens: int = 0
  total_tokens: int = 0
  responses_with_usage: int = 0


class OpenAICompatibleVLM:
  def __init__(
      self, base_url: str | None = None,
      model: str | None = None, *, structured_output: bool = True,
  ):
    base_url = base_url or os.environ.get("DMS_VLM_BASE_URL", "http://127.0.0.1:18000/v1")
    model = model or os.environ.get("DMS_VLM_MODEL", "/autodl-tmp/model")
    self.base_url, self.model = base_url.rstrip("/"), model
    self.structured_output = structured_output
    self.usage = TokenUsage()

  def complete(
      self, prompt: str, screenshot_png: bytes | None = None, *, temperature: float = 0.0,
      max_tokens: int = 512, json_schema: dict[str, object] | None = None,
      structured_output: bool | None = None,
  ) -> str:
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    if screenshot_png:
      encoded = base64.b64encode(screenshot_png).decode("ascii")
      content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
    payload = {
        "model": self.model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    use_structured_output = (
        self.structured_output if structured_output is None else structured_output
    )
    if use_structured_output:
      payload["guided_decoding_backend"] = "outlines"
      if json_schema is None:
        payload["response_format"] = {"type": "json_object"}
      else:
        payload["guided_json"] = json_schema
    request = Request(f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=180) as response:
      data = json.load(response)
      usage = data.get("usage") or {}
      if usage:
        self.usage.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.usage.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.usage.total_tokens += int(usage.get("total_tokens") or 0)
        self.usage.responses_with_usage += 1
      return data["choices"][0]["message"]["content"]

  def snapshot_usage(self) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=self.usage.prompt_tokens,
        completion_tokens=self.usage.completion_tokens,
        total_tokens=self.usage.total_tokens,
        responses_with_usage=self.usage.responses_with_usage,
    )
