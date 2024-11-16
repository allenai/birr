import os
import unittest

from birr.batch_inference.data_models import ChatMessage
from birr.core.config import LLMModelConfig, FormatConfig
from birr.tokenization import ModelTokenizer


CUR_DIR = os.path.dirname(os.path.realpath(__file__))
DUMMY_ARTIFACTS_DIR = os.path.join(CUR_DIR, "fixtures", "dummy_model")


class TestModelTokenizer(unittest.TestCase):
    def test__batch_format_handles_system_message(self) -> None:
        model_config = LLMModelConfig(name_or_path=DUMMY_ARTIFACTS_DIR)
        format_config_with_sys_msg = FormatConfig(
            system_message="You are a passive aggressive and reluctant assistant of dubious intelligence.",
            add_generation_prompt=False,
        )
        format_config_without_sys_msg = FormatConfig(
            system_message=None,
            add_generation_prompt=False,
        )

        tokenizer_with_sys_msg = ModelTokenizer(model_config, format_config_with_sys_msg)
        tokenizer_without_sys_msg = ModelTokenizer(model_config, format_config_without_sys_msg)

        user_message = "I am a user, short and stout."
        formatted_with_sys_msg = tokenizer_with_sys_msg.batch_format([user_message])[0]
        formatted_without_sys_msg = tokenizer_without_sys_msg.batch_format([user_message])[0]

        expected_with_sys_msg = "\n".join(
            [
                "<|im_start|>system",
                "You are a passive aggressive and reluctant assistant of dubious intelligence.<|im_end|>",
                "<|im_start|>user",
                "I am a user, short and stout.<|im_end|>",
                "",
            ]
        )
        expected_without_sys_msg = "\n".join(
            [
                "<|im_start|>system",
                "You are a helpful assistant.<|im_end|>",
                "<|im_start|>user",
                "I am a user, short and stout.<|im_end|>",
                "",
            ]
        )

        self.assertEqual(formatted_with_sys_msg, expected_with_sys_msg)
        self.assertEqual(formatted_without_sys_msg, expected_without_sys_msg)

    def test__batch_format_handles_multiple_message_instances(self):
        model_config = LLMModelConfig(name_or_path=DUMMY_ARTIFACTS_DIR)
        messages = [
            ChatMessage(role="system", content="You are a fantastic assistant."),
            ChatMessage(role="user", content="I would like to know what time it is."),
            ChatMessage(role="assistant", content="Alas I am not a clock."),
            ChatMessage(role="user", content="Some omniscient being you are."),
        ]
        tokenizer = ModelTokenizer(model_config, FormatConfig(add_generation_prompt=False))
        formatted = tokenizer.batch_format([messages])[0]
        expected = "\n".join(
            [
                "<|im_start|>system",
                "You are a fantastic assistant.<|im_end|>",
                "<|im_start|>user",
                "I would like to know what time it is.<|im_end|>",
                "<|im_start|>assistant",
                "Alas I am not a clock.<|im_end|>",
                "<|im_start|>user",
                "Some omniscient being you are.<|im_end|>",
                "",
            ]
        )
        self.assertEqual(formatted, expected)

    def test__batch_format_handles_instruction_prefix_and_generation_prompt(self) -> None:
        model_config = LLMModelConfig(name_or_path=DUMMY_ARTIFACTS_DIR)

        chat_template = """
{%- if messages[0]['role'] == 'system' -%}
    {%- set system_message = messages[0]['content'] | trim + '\n\n' -%}
    {%- set messages = messages[1:] -%}
{%- else -%}
    {%- set system_message = '' -%}
{%- endif -%}

{{- '\n' -}}
{{- system_message -}}
{%- for message in messages -%}
    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}
        {{- raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') -}}
    {%- endif -%}

    {%- if message['role'] == 'user' -%}
        {{- '[INST] ' + message['content'] | trim + ' [/INST]' -}}
    {%- elif message['role'] == 'assistant' -%}
        {{- message['content'] | trim + eos_token + ' ' -}}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- generation_prompt -}}
{%- endif -%}"""

        format_config = FormatConfig(
            instruction_prefix="Original: ",
            generation_prompt="Rewritten:\n",
            add_generation_prompt=True,
            chat_template=chat_template,
        )

        tokenizer = ModelTokenizer(
            model_config,
            format_config,
        )

        # Simple text
        user_message = "I am a user, short and stout."
        formatted = tokenizer.batch_format([user_message])[0]
        expected = "\n".join(["", "[INST] Original: I am a user, short and stout. [/INST]Rewritten:", ""])
        self.assertEqual(formatted, expected)

        # As chat message
        user_chat_message = [ChatMessage.from_text("I am a user, short and stout.")]
        formatted = tokenizer.batch_format([user_chat_message])[0]
        expected = "\n".join(["", "[INST] Original: I am a user, short and stout. [/INST]Rewritten:", ""])
        self.assertEqual(formatted, expected)
