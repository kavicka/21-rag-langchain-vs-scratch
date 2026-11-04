"""One small interface in front of every LLM provider, chosen with LLM_PROVIDER in the environment.

Why: the RAG logic (retrieve → prompt → stream → check citations) must not care whether the tokens come
from Ollama on the Mac, AWS Bedrock (step 18) or Azure OpenAI (step 19). Switching is a config change,
not a code change. Credentials are never parameters here: AWS uses its standard credential chain
(SSO profile / role), Azure reads its key from the environment via Settings.
"""

from collections.abc import Iterator
from typing import Protocol

import httpx

from rag.config import Settings, get_settings

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


class LLM(Protocol):
    name: str

    def stream(self, messages: list[Message]) -> Iterator[str]: ...


def complete(llm: LLM, messages: list[Message]) -> str:
    return "".join(llm.stream(messages))


class OllamaLLM:
    """Local models through Ollama's HTTP API (the default: 0 €, data never leaves the machine)."""

    def __init__(self, s: Settings):
        self.s, self.name = s, f"ollama:{s.llm_model}"

    def stream(self, messages: list[Message]) -> Iterator[str]:
        import json

        body = {
            "model": self.s.llm_model,
            "messages": messages,
            "stream": True,
            "think": False,  # Qwen3 would otherwise spend hundreds of tokens "thinking" before answering
            "options": {"temperature": self.s.llm_temperature, "num_predict": self.s.llm_max_tokens, "seed": 0},
        }
        with httpx.stream("POST", f"{self.s.ollama_url}/api/chat", json=body, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    part = json.loads(line)
                    if tok := part.get("message", {}).get("content"):
                        yield tok


class TransformersLLM:
    """A Hugging Face model run in-process. Used where Ollama is not available (cloud CPU, Colab)."""

    def __init__(self, s: Settings):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.s, self.name = s, f"transformers:{s.llm_model}"
        torch.manual_seed(0)
        self.tok = AutoTokenizer.from_pretrained(s.llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(s.llm_model, torch_dtype="auto")
        self.model.eval()

    def stream(self, messages: list[Message]) -> Iterator[str]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        kwargs = {"enable_thinking": False} if "qwen3" in self.s.llm_model.lower() else {}
        ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True, **kwargs
        )
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        gen = {**ids, "max_new_tokens": self.s.llm_max_tokens, "do_sample": False, "streamer": streamer}
        Thread(target=self.model.generate, kwargs=gen, daemon=True).start()
        yield from streamer


class BedrockLLM:
    """AWS Bedrock Converse API. Prepared for step 18; not run in this step (it costs money)."""

    def __init__(self, s: Settings):
        import boto3  # uv sync --extra bedrock

        self.s, self.name = s, f"bedrock:{s.llm_model}"
        self.client = boto3.client("bedrock-runtime", region_name=s.aws_region)  # credentials: AWS default chain

    def stream(self, messages: list[Message]) -> Iterator[str]:
        system = [{"text": m["content"]} for m in messages if m["role"] == "system"]
        msgs = [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages if m["role"] != "system"]
        resp = self.client.converse_stream(
            modelId=self.s.llm_model,
            system=system,
            messages=msgs,
            inferenceConfig={"temperature": self.s.llm_temperature, "maxTokens": self.s.llm_max_tokens},
        )
        for ev in resp["stream"]:
            if delta := ev.get("contentBlockDelta", {}).get("delta", {}).get("text"):
                yield delta


class AzureOpenAILLM:
    """Azure OpenAI chat completions. Prepared for step 19; not run in this step (it costs money)."""

    def __init__(self, s: Settings):
        from openai import AzureOpenAI  # uv sync --extra azure

        if not (s.azure_openai_endpoint and s.azure_openai_api_key):
            raise RuntimeError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in the environment")
        self.s, self.name = s, f"azure_openai:{s.llm_model}"
        self.client = AzureOpenAI(
            azure_endpoint=s.azure_openai_endpoint,
            api_key=s.azure_openai_api_key.get_secret_value(),
            api_version=s.azure_openai_api_version,
        )

    def stream(self, messages: list[Message]) -> Iterator[str]:
        resp = self.client.chat.completions.create(
            model=self.s.llm_model,  # the *deployment* name in Azure
            messages=messages,
            temperature=self.s.llm_temperature,
            max_tokens=self.s.llm_max_tokens,
            stream=True,
        )
        for ev in resp:
            if ev.choices and (tok := ev.choices[0].delta.content):
                yield tok


class FakeLLM:
    """Deterministic stand-in for tests: answers with the first line of the first source and cites it."""

    name = "fake"

    def stream(self, messages: list[Message]) -> Iterator[str]:
        import re

        user = messages[-1]["content"]
        m = re.search(r"\[(NCT\d{8})\][^\n]*\n([^\n]+)", user)
        answer = f"{m.group(2)} [{m.group(1)}]" if m else "I don't know."
        for i, word in enumerate(answer.split(" ")):
            yield word if i == 0 else " " + word


PROVIDERS = {
    "ollama": OllamaLLM,
    "transformers": TransformersLLM,
    "bedrock": BedrockLLM,
    "azure_openai": AzureOpenAILLM,
}


def get_llm(s: Settings | None = None) -> LLM:
    s = s or get_settings()
    return FakeLLM() if s.llm_provider == "fake" else PROVIDERS[s.llm_provider](s)
