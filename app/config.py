from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MODEL_PATH = "models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"


@dataclass(frozen=True)
class Settings:
    base_url: str = DEFAULT_BASE_URL
    model_path: str = DEFAULT_MODEL_PATH
    model_name: str = "local"
    timeout_seconds: float = 180.0
    max_iterations: int = 5

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            base_url=os.environ.get("LLAMA_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            model_path=os.environ.get("LLAMA_MODEL_PATH", DEFAULT_MODEL_PATH),
            model_name=os.environ.get("LLAMA_MODEL_NAME", "local"),
            timeout_seconds=float(os.environ.get("LLAMA_TIMEOUT_SECONDS", 180)),
            max_iterations=int(os.environ.get("LLAMA_MAX_ITERATIONS", 5)),
        )
