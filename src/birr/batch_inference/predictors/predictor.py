import logging
import sys
from typing import List, Optional

from outlines.serve.vllm import JSONLogitsProcessor
from vllm import LLM, SamplingParams, TokensPrompt

from birr.batch_inference.data_models import (
    CompletionError,
    CompletedItem,
    CompletionOutput,
    PreparedInputItem,
)
from birr.batch_inference.predictors.base_predictor import BasePredictor


logger = logging.getLogger(__name__)


MAX_ALLOWED_CUDA_ERRORS = 2


class Predictor(BasePredictor):
    def _load_model(self) -> LLM:
        # See: https://github.com/vllm-project/vllm/pull/8001
        enable_chunked_prefill = False if self._model_config.num_scheduler_steps > 1 else None

        llm = LLM(
            self._model_config.name_or_path,
            trust_remote_code=self._model_config.trust_remote_code,
            skip_tokenizer_init=False,  # we need the tokenizer for vLLM to know about stop tokens
            max_model_len=self._model_config.max_model_len,
            tensor_parallel_size=self._model_config.tensor_parallel_size or 1,
            gpu_memory_utilization=self._model_config.gpu_memory_utilization,
            num_scheduler_steps=self._model_config.num_scheduler_steps,
            enable_chunked_prefill=enable_chunked_prefill,
        )

        self._logits_processors: Optional[List[JSONLogitsProcessor]]
        if self._generate_config.guided_decoding_json_schema:
            self._logits_processors = [
                JSONLogitsProcessor(
                    schema=self._generate_config.guided_decoding_json_schema,
                    llm=llm.llm_engine,
                )
            ]
        else:
            self._logits_processors = None

        self._accumulated_cuda_errors = 0

        return llm

    def predict(self, batch: List[PreparedInputItem]) -> List[CompletedItem]:
        tokens_prompt_batch: List[TokensPrompt] = [
            dict(
                prompt_token_ids=instance.token_ids,
                multi_modal_data=None if not instance.image_data else dict(image=instance.image_data),
            )
            for instance in batch
            if len(instance.token_ids) <= self._generate_config.max_context_length
        ]

        if not tokens_prompt_batch:
            return []

        longest_sequence = max([len(instance["prompt_token_ids"]) for instance in tokens_prompt_batch])

        if self._generate_config.max_tokens:
            max_tokens = self._generate_config.max_tokens
        else:
            max_tokens = longest_sequence

        sampling_params = SamplingParams(
            n=1,  # TODO: allow multiple outputs per input; will affect batching in a few places
            temperature=self._generate_config.temperature,
            top_k=self._generate_config.top_k,
            top_p=self._generate_config.top_p,
            max_tokens=max_tokens,
            logits_processors=self._logits_processors,
            detokenize=False,  # we defer this since it can be CPU-intensive
            presence_penalty=self._generate_config.presence_penalty,
            frequency_penalty=self._generate_config.frequency_penalty,
            repetition_penalty=self._generate_config.repetition_penalty,
        )

        try:
            outputs = self._model.generate(
                tokens_prompt_batch,
                use_tqdm=False,
                sampling_params=sampling_params,
            )
        except Exception as exc:
            if "CUDA error" in str(exc):
                self._accumulated_cuda_errors += 1
                if self._accumulated_cuda_errors >= MAX_ALLOWED_CUDA_ERRORS:
                    logger.exception(
                        """
                    CUDA errors encountered too many times -- GPU memory likely in unrecoverable state.
                    Terminating predictor...
                    """
                    )
                    sys.exit(1)
            raise exc

        unfiltered_predictions = [
            CompletedItem(
                index=instance.index,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text="",
                        token_ids=list(prediction.outputs[0].token_ids),
                        finish_reason=prediction.outputs[0].finish_reason,
                        stop_reason=prediction.outputs[0].stop_reason,
                    )
                ],
            )
            for instance, prediction in zip(batch, outputs)
        ]

        filtered_predictions = [
            prediction for prediction in unfiltered_predictions if prediction.outputs[0].finish_reason == "stop"
        ]

        if self._generate_config.drop_long_outputs:
            predictions = filtered_predictions
        else:
            predictions = unfiltered_predictions

        context_too_longs = [
            CompletedItem(index=instance.index, outputs=[], error=CompletionError.CONTEXT_TOO_LONG)
            for instance in batch
            if len(instance.token_ids) > self._generate_config.max_context_length
        ]

        if not self._generate_config.drop_long_contexts:
            predictions += context_too_longs

        return predictions
