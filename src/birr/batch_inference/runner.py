import logging

import ray
from ray.util import ActorPool

from birr.batch_inference.generate_io_processor import GenerateIOProcessor
from birr.batch_inference.predictors.predictor import Predictor
from birr.batch_inference.queue.in_memory_queue import InMemoryQueue
from birr.batch_inference.settings import Settings
from birr.batch_inference.worker import Worker


logger = logging.getLogger(__name__)


@ray.remote(num_cpus=0.25)
class InMemoryQueueActor(InMemoryQueue):
    """Simple ray actor wrapper around the underlying birr.batch_inference.in_memory_queue class"""


@ray.remote(num_cpus=1)
class TokenizerActor(GenerateIOProcessor):
    """Simple ray actor wrapper around the underlying birr.batch_inference.tokenizer class"""


@ray.remote(num_cpus=1)
class WorkerActor(Worker):
    """Simple ray actor wrapper around the underlying birr.batch_inference.worker class"""


def main() -> None:
    settings = Settings()
    ray.init(
        object_store_memory=10**9 * 2,
        _metrics_export_port=8080,
        logging_config=ray.LoggingConfig(encoding="TEXT", log_level="INFO"),
    )
    logger.info("Starting batch inference server with settings:\n %s", settings)

    queue = InMemoryQueueActor.remote(settings.pipeline_config, dd)  # type: ignore

    tokenizer_pool = ActorPool(
        [TokenizerActor.remote(settings.llm_model_config, settings.format_config) for _ in range(settings.pipeline_config.num_tokenizers)]  # type: ignore
    )

    @ray.remote(
        num_gpus=settings.gpus_per_predictor,
        max_restarts=settings.pipeline_config.allowed_restarts_per_predictor,
        max_task_retries=settings.pipeline_config.max_task_retries,
    )
    class PredictorActor(Predictor):
        """Simple ray actor wrapper around the underlying birr.batch_inference.predictor class"""

    predictor_pool = ActorPool(
        [PredictorActor.remote(settings.llm_model_config, settings.generate_config) for _ in range(settings.num_predictors)]  # type: ignore
    )

    workers = [
        WorkerActor.remote(settings, queue, dd, tokenizer_pool, predictor_pool)  # type: ignore
        for _ in range(settings.pipeline_config.num_workers)
    ]

    # Start work loop
    ray.get([worker.run.remote() for worker in workers])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Batch inference server failed with exception: %s", e)
        raise e
