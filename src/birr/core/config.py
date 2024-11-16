from typing import Any, Dict, List, Optional, Tuple, Union

from jsonschema.protocols import Validator as JSONSchemaValidator
from pydantic import BaseModel, ConfigDict, Field, model_validator


class LLMModelConfig(BaseModel):
    """Configuration for loading a model; includes model name and type."""

    model_config = ConfigDict(extra="forbid")

    name_or_path: str = Field(
        default="/artifacts",
        description="The model name or path to load; must be compatible with huggingface transformers.",
    )
    dtype: str = Field(default="bfloat16", description="The precision to use for the model.")
    trust_remote_code: bool = Field(
        default=False, description="Whether to trust python code packaged with model files."
    )
    vlm: bool = Field(default=False, description="Whether this is a vision-language model or not.")
    tensor_parallel_size: Optional[int] = Field(
        default=None,
        description="For big models, how many GPUs to split them between. Mutually exclusive with `PipelineConfig.predictors_per_gpu`.",
    )
    max_model_len: Optional[int] = Field(
        default=None, description="Optionally override the model's configured model length."
    )
    fast_tokenizer: bool = Field(default=True, description="Whether to use the fast tokenizer for the model.")
    gpu_memory_utilization: float = Field(
        default=0.9,
        description="Determines how much vram the model is allowed to use, includings its allocated KV cache.",
    )
    num_scheduler_steps: int = Field(
        default=8,
        description="Minimizes CPU-bound overhead within vLLM. Set to 1 to opt out (some compatibility issues in some cases with >1).",
    )


class FormatConfig(BaseModel):
    """Configuration for formatting the text that is input to the model."""

    model_config = ConfigDict(extra="forbid")

    new_line_symbol: str = Field(
        default="\n",
        description="The symbol to use for new lines in the text; default is '\\n'.",
    )
    system_message: Optional[str] = Field(
        default=None,
        description="The system message to use for formatting the text; default is no system message.",
    )
    instruction_prefix: Optional[str] = Field(
        default=None,
        description="Optional string to prepend before each user message. Note: no additional whitespace will be inserted after your prefix.",
    )
    generation_prompt: Optional[str] = Field(
        default=None,
        description="Optional string to prompt a response from the model. Your template must support `generation_prompt` as a template parameter.",
    )
    add_generation_prompt: bool = Field(
        default=True,
        description="Whether to add a generation prompt or not. Chat template must support the `add_generation_prompt` flag. Custom `generation_prompt` supplied in this config will only be used if the template chat template supports it.",
    )
    chat_template: Optional[str] = Field(
        default=None,
        description="The template to use for formatting the chat text. If None, the default chat template for the tokenizer will be used. Note: whitespace is preserved. If authoring a multi-line template, avoid adding additional newlines or leading/trailing spaces in your output by using minus signs in your block syntax. E.g. '{{-  -}}', '{%-  -%}'.",
    )


class GenerateConfig(BaseModel):
    """Configuration for output generation."""

    model_config = ConfigDict(extra="forbid")

    max_tokens: Optional[int] = Field(
        default=None,
        description="The maximum length of the generated text. If left unset, defaults to longest sequence in batch.",
    )
    max_context_length: int = Field(
        default=4096,
        description="The longest length prompt we'll attempt to generate for. Includes chat template tokens.",
    )
    temperature: float = Field(default=0.2, description="The temperature to use for generation")
    top_k: int = Field(default=50, description="The top k to use for generation")
    top_p: float = Field(default=1.0, description="The top p to use for generation")
    drop_long_contexts: bool = Field(
        default=False,
        description="If true, will discard any rows that had too many tokens for the model max context length",
    )
    drop_long_outputs: bool = Field(
        default=False, description="If true, will discard any outputs that exceed `max_tokens` in length."
    )
    guided_decoding_json_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="JSON Schema to drive guided decoding."
    )
    presence_penalty: float = Field(
        default=0.0,
        description="""Float that penalizes new tokens based on whether they appear in the generated text so far.
        Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.""",
    )
    frequency_penalty: float = Field(
        default=0.0,
        description="""Float that penalizes new tokens based on their frequency in the generated text so far. 
        Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.""",
    )
    repetition_penalty: float = Field(
        default=1.0,
        description="""Float that penalizes new tokens based on whether they appear in the prompt and the 
        generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage 
        the model to repeat tokens.""",
    )

    @model_validator(mode="after")
    def validate_guided_decoding_json_schema(self) -> "GenerateConfig":
        if self.guided_decoding_json_schema:
            JSONSchemaValidator.check_schema(self.guided_decoding_json_schema)

        return self


class PipelineConfig(BaseModel):
    """Configuration for pipeline parameters."""

    model_config = ConfigDict(extra="forbid")

    input_file_dir: str = Field(
        description="Local directory of files to process",
    )
    output_file_dir: str = Field(description="Local directory to write output files to")
    predictors_per_gpu: Optional[int] = Field(
        default=None,
        description="For having multiple copies of the model per available gpu. Mutually exclusive with `ModelConfig.tensor_parallel_size`.",
    )
    allowed_restarts_per_predictor: int = Field(
        default=0,
        ge=0,  # -1 means infinite restarts which we don't want to support; smaller values are invalid
        description="If set >0, will restart any predictor that crashes that many times to continue work.",
    )
    num_workers: int = Field(
        default=1, ge=1, description="Determines number of messages that can be worked on concurrently."
    )
    max_num_messages_per_worker: Optional[int] = Field(
        default=None, description="If set, worker actor will terminate after processing this many messages."
    )
    max_instances_per_message: Optional[int] = Field(
        default=None,
        description="For debugging/benchmarking purposes. If set, will clip the contents of a message to the set number of rows.",
    )
    num_tokenizers: int = Field(default=4, ge=1, description="How many tokenizer actors to run.")
    num_gpus: int = Field(default=0, ge=0, description="How many GPUs are available to run on. Assumes 0.")
    tokenization_batch_size: int = Field(
        default=3000, ge=1, description="How many documents to tokenize at a time."
    )
    generation_batch_size: Union[int, List[Tuple[int, int]]] = Field(
        description="How many prompts to pass for inference at a time. Either a flat number, or a list of numbers bracketed by max sequence length."
    )
    decoding_batch_size: int = Field(
        default=3000, ge=1, description="How many documents to decode into from tokens into text at a time."
    )

    @property
    def max_task_retries(self) -> int:
        if self.allowed_restarts_per_predictor > 0:
            return -1

        return 0
