import os
from typing import Optional, Union

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from birr.core.config import FormatConfig, GenerateConfig, LLMModelConfig, PipelineConfig


_CUR_DIR = os.path.dirname(os.path.realpath(__file__))


class Settings(BaseSettings):
    llm_model_config: LLMModelConfig = Field(
        default=LLMModelConfig(), description="Configuration for loading a model."
    )

    format_config: FormatConfig = Field(
        default=FormatConfig(), description="Configuration for the text that is input to the model."
    )

    generate_config: GenerateConfig = Field(
        default=GenerateConfig(), description="Configuration for generating outputs."
    )

    pipeline_config: PipelineConfig = Field(
        description="Configuration for initializing and running the batch pipeline."
    )

    dummy_mode: bool = Field(
        default=False,
        description="Whether to run in dummy mode",
    )

    @model_validator(mode="after")
    def validate_parallelism_and_multi_copy_mutual_exclusion(self) -> "Settings":
        if self.llm_model_config.tensor_parallel_size and self.pipeline_config.predictors_per_gpu:
            raise ValueError(
                "Cannot set both `LLMModelConfig.tensor_parallel_size` and `PipelineConfig.predictors_per_gpu`"
            )

        return self

    @model_validator(mode="after")
    def validate_parallelism(self) -> "Settings":
        if self.llm_model_config.tensor_parallel_size:
            if not self.pipeline_config.num_gpus:
                raise ValueError("Can't have tensor parallelism without gpus")
            if self.pipeline_config.num_gpus % self.llm_model_config.tensor_parallel_size != 0:
                raise ValueError("num_gpus must be divisible by tensor_parallelism")

        return self

    @model_validator(mode="after")
    def validate_predictors(self) -> "Settings":
        if not self.dummy_mode and self.num_predictors <= 0:
            raise ValueError("No predictors, make sure num_gpus is set correctly!")

        return self

    @model_validator(mode="after")
    def validate_has_gpus(self) -> "Settings":
        if not self.dummy_mode and self.pipeline_config.num_gpus == 0:
            raise ValueError("When not running in `dummy_mode`, `PipelineConfig.num_gpus` must be greater than 0.")

        return self

    @property
    def num_predictors(self) -> int:
        if self.llm_model_config.tensor_parallel_size:
            return int(self.pipeline_config.num_gpus / self.llm_model_config.tensor_parallel_size)

        if self.pipeline_config.predictors_per_gpu:
            return self.pipeline_config.num_gpus * self.pipeline_config.predictors_per_gpu

        return self.pipeline_config.num_gpus

    @property
    def gpus_per_predictor(self) -> Union[int, float]:
        if not self.pipeline_config.num_gpus:
            return 0

        if self.llm_model_config.tensor_parallel_size:
            return self.llm_model_config.tensor_parallel_size

        if self.pipeline_config.predictors_per_gpu:
            return 1 / self.pipeline_config.predictors_per_gpu

        return 1
