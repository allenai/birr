"""
Union of (De)Tokenizer and MultiModal Input Processing
functionality.

Exists as a separate actor in the pipeline to offload
cpu-bound work from the core inference component.
"""

import base64
from io import BytesIO
from typing import List, Optional

from PIL import Image

from birr.batch_inference.data_models import (
    CompletedItem,
    ImageChatMessageContent,
    PreparedInputItem,
    RawInputItem,
    TokenizedItem,
)
from birr.core.config import FormatConfig, LLMModelConfig
from birr.tokenization import ModelTokenizer


class GenerateIOProcessor:
    def __init__(self, model_config: LLMModelConfig, format_config: FormatConfig):
        self._vlm = model_config.vlm
        self._tokenizer = ModelTokenizer(
            name_or_path=model_config.name_or_path, model_config=model_config, format_config=format_config
        )

    def prepare_inputs(self, batch: List[RawInputItem]) -> List[PreparedInputItem]:
        tokenized_inputs = self.tokenize(batch)

        image_inputs = self.load_images(batch) if self._vlm else [None for _ in batch]

        return [
            PreparedInputItem(index=tokens.index, token_ids=tokens.token_ids, image_data=image_data)
            for tokens, image_data in zip(tokenized_inputs, image_inputs)
        ]

    def tokenize(self, batch: List[RawInputItem]) -> List[TokenizedItem]:
        batch_encoding = self._tokenizer.batch_process(instances=[instance.messages for instance in batch])
        batch_input_ids = batch_encoding.input_ids
        batch_attention_mask = batch_encoding.attention_mask

        final_batch_input_ids = []

        # Apply attention mask since we have prompts of unequal length
        for instance_input_ids, instance_attention_mask in zip(batch_input_ids, batch_attention_mask):
            final_instance_ids = []
            assert len(instance_input_ids) == len(instance_attention_mask)
            for token_id, keep in zip(list(instance_input_ids), list(instance_attention_mask)):
                if keep:
                    final_instance_ids.append(token_id)
            final_batch_input_ids.append(final_instance_ids)

        return [
            TokenizedItem(item.index, instance_token_ids)
            for item, instance_token_ids in zip(batch, final_batch_input_ids)
        ]

    def load_images(self, batch: List[RawInputItem]) -> List[Optional[List[Image.Image]]]:
        image_objects: List[Optional[List[Image.Image]]] = []

        for instance in batch:
            instance_image_objects = []
            for message in instance.messages:
                if isinstance(message.content, str):
                    continue
                else:
                    for content_item in message.content:
                        if isinstance(content_item, ImageChatMessageContent):
                            data = content_item.image.split(";", 1)[1]
                            assert data.startswith("base64,"), "Invalid image data"
                            decoded_data = base64.b64decode(data[7:])
                            image_object = Image.open(BytesIO(decoded_data))
                            instance_image_objects.append(image_object)

            if instance_image_objects:
                image_objects.append(instance_image_objects)
            else:
                image_objects.append(None)

        return image_objects

    def decode(self, batch: List[CompletedItem]) -> List[CompletedItem]:
        completion_outputs = []
        for item in batch:
            for completion_output in item.outputs:
                completion_outputs.append(completion_output)

        token_batch = [output.token_ids for output in completion_outputs]

        decoded = self._tokenizer.batch_decode(token_batch, skip_special_tokens=True)
        for completion_output, decoded_item in zip(completion_outputs, decoded):
            completion_output.text = decoded_item

        return batch
