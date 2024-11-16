import os
import unittest

from birr.batch_inference.settings import Settings
from birr.core.config import LLMModelConfig, PipelineConfig


def mk_default_pipeline_config():
    return PipelineConfig(
        input_file_dir="/input/files",
        output_file_dir="/output/files",
        generation_batch_size=16,
    )


class TestSetting(unittest.TestCase):
    def setUp(self):
        self._cached_env = dict(os.environ)
        os.environ.clear()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._cached_env)

    def test__parallelism_and_multiple_copies_of_model_per_gpu_are_mutually_exclusive(self):
        with self.assertRaises(Exception):
            model_config = LLMModelConfig(tensor_parallel_size=4)
            pipeline_config = mk_default_pipeline_config()
            pipeline_config.num_gpus = 8
            pipeline_config.predictors_per_gpu = 2
            os.environ.update(
                dict(
                    LLM_MODEL_CONFIG=model_config.model_dump_json(),
                    PIPELINE_CONFIG=pipeline_config.model_dump_json(),
                )
            )
            Settings()

    def test__parallelism_validator_requires_num_gpus_be_set_and_divisible_by_parallelism(self):
        model_config = LLMModelConfig()
        pipeline_config = mk_default_pipeline_config()

        with self.assertRaises(Exception):
            model_config.tensor_parallel_size = 4
            os.environ.update(
                dict(
                    LLM_MODEL_CONFIG=model_config.model_dump_json(),
                    PIPELINE_CONFIG=pipeline_config.model_dump_json(),
                )
            )
            Settings()

        with self.assertRaises(Exception):
            pipeline_config.num_gpus = 9  # not divisible by 4
            os.environ.update(dict(PIPELINE_CONFIG=pipeline_config.model_dump_json()))
            Settings()

        pipeline_config.num_gpus = 8
        os.environ.update(dict(PIPELINE_CONFIG=pipeline_config.model_dump_json()))
        Settings()

    def test__there_must_be_at_least_one_gpu_if_not_in_dummy_mode(self):
        pipeline_config = mk_default_pipeline_config()

        with self.assertRaises(Exception):
            pipeline_config.num_gpus = 0
            os.environ.update(dict(DUMMY_MODE="0", PIPELINE_CONFIG=pipeline_config.model_dump_json()))
            Settings()

        pipeline_config.num_gpus = 1
        os.environ.update(dict(PIPELINE_CONFIG=pipeline_config.model_dump_json()))
        Settings()

    def test__gpus_per_predictor_can_assign_fractional_gpus(self):
        pipeline_config = mk_default_pipeline_config()
        pipeline_config.num_gpus = 1
        pipeline_config.predictors_per_gpu = 2

        os.environ.update(dict(PIPELINE_CONFIG=pipeline_config.model_dump_json()))

        settings = Settings()
        self.assertEqual(settings.gpus_per_predictor, 0.5)
