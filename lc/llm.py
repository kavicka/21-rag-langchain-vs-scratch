"""The LLM as a LangChain chat model, chosen with LLM_PROVIDER like in the scratch pipeline.

Held constant against the scratch run: same weights (Qwen/Qwen3-1.7B), greedy decoding, 256 new tokens,
thinking off, so the eval compares orchestration and retrieval, not two different generators.

Three things had to be done by hand to get there (each one is a finding in the README):
1. ChatHuggingFace calls tokenizer.apply_chat_template() without extra kwargs, so Qwen3's
   `enable_thinking=False` can't be passed. The template is patched so the "no thinking" branch is always taken,
   which gives the exact same prompt string as the scratch code (checked in tests/test_lc.py).
2. Generation settings are passed to the transformers pipeline itself: HuggingFacePipeline's own
   `pipeline_kwargs` field is only used by from_model_id() and silently ignored for a pipeline you pass in.
3. `return_full_text=False`, otherwise the non-streaming path returns the prompt + the answer as the "answer".
"""

from collections.abc import Iterator
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from rag.config import Settings, get_settings
from rag.llm import FakeLLM

QWEN3_THINK_SWITCH = "enable_thinking is defined and enable_thinking is false"


def no_thinking_template(template: str) -> str:
    if QWEN3_THINK_SWITCH not in template:
        raise ValueError("chat template has no Qwen3 thinking switch; is this a Qwen3 model?")
    return template.replace(QWEN3_THINK_SWITCH, "true")


def build_chat_model(s: Settings | None = None) -> BaseChatModel:
    s = s or get_settings()
    if s.llm_provider == "fake":
        return FakeChatModel()
    if s.llm_provider == "transformers":
        import torch
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        torch.manual_seed(0)
        tok = AutoTokenizer.from_pretrained(s.llm_model)
        model = AutoModelForCausalLM.from_pretrained(s.llm_model, torch_dtype="auto")
        model.eval()
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=s.llm_max_tokens,
            do_sample=False,
            return_full_text=False,
        )
        if "qwen3" in s.llm_model.lower():
            tok.chat_template = no_thinking_template(tok.chat_template)
        return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe, model_id=s.llm_model), tokenizer=tok)
    # Everything else goes through LangChain's provider registry: one line per provider, but each one needs
    # its own integration package (langchain-ollama, langchain-aws, langchain-openai).
    from langchain.chat_models import init_chat_model

    provider = {"azure_openai": "azure_openai", "bedrock": "bedrock_converse", "ollama": "ollama"}[s.llm_provider]
    extra: dict[str, Any] = {"reasoning": False, "num_predict": s.llm_max_tokens, "seed": 0}
    if provider != "ollama":
        extra = {"max_tokens": s.llm_max_tokens}
    return init_chat_model(s.llm_model, model_provider=provider, temperature=s.llm_temperature, **extra)


class FakeChatModel(BaseChatModel):
    """The scratch FakeLLM behind LangChain's chat-model interface, for tests and for timing the framework alone."""

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _text(self, messages: list[BaseMessage]) -> Iterator[str]:
        return FakeLLM().stream([{"role": "user", "content": str(messages[-1].content)}])

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kw) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage("".join(self._text(messages))))])

    def _stream(self, messages: list[BaseMessage], stop=None, run_manager=None, **kw) -> Iterator[ChatGenerationChunk]:
        for tok in self._text(messages):
            yield ChatGenerationChunk(message=AIMessageChunk(tok))
