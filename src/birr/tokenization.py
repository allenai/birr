from typing import Dict, List, Optional, Union

from transformers import AutoTokenizer, BatchEncoding

from birr.batch_inference.data_models import ChatMessage
from birr.core.config import FormatConfig, LLMModelConfig


class ModelTokenizer:
    def __init__(
        self, model_config: LLMModelConfig, format_config: FormatConfig, name_or_path: Optional[str] = None
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            name_or_path or model_config.name_or_path,
            use_fast=model_config.fast_tokenizer,
            trust_remote_code=model_config.trust_remote_code,
        )
        self._format_config = format_config

        if format_config.chat_template:
            self.tokenizer.chat_template = format_config.chat_template

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.new_line_symbol = str(format_config.new_line_symbol)

    @property
    def model_max_length(self) -> int:
        return self.tokenizer.model_max_length

    @model_max_length.setter
    def model_max_length(self, value: int):
        self.tokenizer.model_max_length = min(value, self.tokenizer.model_max_length)

    def __len__(self) -> int:
        return len(self.tokenizer)

    def _system_message(self) -> List[Dict[str, str]]:
        if self._format_config.system_message:
            return [{"role": "system", "content": self._format_config.system_message}]
        return []

    def _format_single(self, instance: Union[str, List[ChatMessage]]) -> List[Dict[str, str]]:
        prefix = self._format_config.instruction_prefix or ""
        if isinstance(instance, str):
            return [*self._system_message(), {"role": "user", "content": f"{prefix}{instance}"}]
        else:
            messages = [msg for msg in instance]
            messages[-1] = messages[-1].copy()
            messages[-1].text = f"{prefix}{messages[-1].text}"
            return [*self._system_message(), *[msg.to_dict() for msg in messages]]

    def batch_format(self, instances: List[Union[str, List[ChatMessage]]]) -> List[str]:
        chat_templated = [self._format_single(instance) for instance in instances]

        formatted = self.tokenizer.apply_chat_template(
            chat_templated,
            tokenize=False,
            add_generation_prompt=self._format_config.add_generation_prompt,
            generation_prompt=(
                self._format_config.generation_prompt if self._format_config.add_generation_prompt else None
            ),
        )
        formatted_replaced = [elem.replace("\n", self.new_line_symbol) for elem in formatted]  # pyright: ignore
        return formatted_replaced

    def batch_process(
        self,
        instances: List[Union[str, List[ChatMessage]]],
        **tokenizer_kwargs,
    ) -> BatchEncoding:
        tokenizer_kwargs.setdefault("return_attention_mask", True)
        tokenizer_kwargs.setdefault("add_special_tokens", False)

        formatted = self.batch_format(instances=instances)

        return self.tokenizer(formatted, **tokenizer_kwargs)

    def process(self, instance: Union[str, List[ChatMessage]], **tokenizer_kwargs) -> BatchEncoding:
        return self.batch_process(
            instances=[instance],
            **tokenizer_kwargs,
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)
