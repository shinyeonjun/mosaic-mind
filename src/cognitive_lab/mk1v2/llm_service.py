"""The speaking satellite: a GGUF language model served by llama.cpp (llama-server), called over HTTP.

Gemma 4 thinks before answering by default (the answer field comes back empty within small token budgets), so
thinking is switched off: the speaking satellite speaks; reasoning belongs to other satellites.

with LLMService("e2b") as llm:
    text = llm.chat([{"role": "user", "content": "..."}])
"""

import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SERVER = ROOT / "tools" / "llama-cpp" / "bin" / "llama-server.exe"
MODELS = {
    "e2b": ROOT / "model" / "gemma-4-E2B-it-qat-gguf" / "gemma-4-E2B_q4_0-it.gguf",
    "e4b": ROOT / "model" / "gemma-4-E4B-it-qat-gguf" / "gemma-4-E4B_q4_0-it.gguf",
}


class LLMService:
    def __init__(self, name: str, port: int = 8081, context: int = 8192, log: Path | None = None):
        self.name, self.port, self.context = name, port, context
        self.url = f"http://127.0.0.1:{port}"
        self.log = log or ROOT / "artifacts" / "logs" / f"llama-server-{name}.log"
        self.process = None
        self.calls = 0
        self.prompt_tokens = self.completion_tokens = 0
        self.seconds = 0.0

    def __enter__(self):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen([str(SERVER), "-m", str(MODELS[self.name]), "-ngl", "999", "-c", str(self.context),
                                         "--port", str(self.port), "--host", "127.0.0.1", "-np", "1"],
                                        stdout=self.log.open("w"), stderr=subprocess.STDOUT)
        for _ in range(120):
            try:
                if json.load(urllib.request.urlopen(f"{self.url}/health", timeout=2)).get("status") == "ok":
                    return self
            except Exception:
                pass
            time.sleep(1)
        raise RuntimeError(f"llama-server did not start; see {self.log}")

    def __exit__(self, *_):
        if self.process is not None:
            self.process.terminate()
            self.process.wait(timeout=30)

    def chat(self, messages: list[dict], max_tokens: int = 256, temperature: float = 0.0, schema: dict | None = None) -> str:
        body = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens, "seed": 0, "cache_prompt": False,
                "chat_template_kwargs": {"enable_thinking": False}}
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
        request = urllib.request.Request(f"{self.url}/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        reply = json.load(urllib.request.urlopen(request, timeout=600))
        self.seconds += time.perf_counter() - started
        self.calls += 1
        usage = reply.get("usage", {})
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        return reply["choices"][0]["message"]["content"] or ""

    def cost(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "seconds": round(self.seconds, 1)}
