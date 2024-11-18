import logging
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import ray

from birr.batch_inference.data_models import (
    CompletedItem,
    Message,
    RawInputItem,
    PreparedInputItem,
)
from birr.batch_inference.settings import Settings
from birr.batch_inference.utils import (
    flatten,
    flatten_and_sort,
    load_instances_from_local_file,
    prediction_batches,
    simple_chunks,
    write_predictions_to_local_file,
)
from birr.batch_inference.serializer import default_serializer


logger = logging.getLogger(__name__)


class Worker:
    def __init__(self, settings: Settings, queue, tokenizers, predictors) -> None:
        self._settings = settings
        self._queue = queue
        self._tokenizers = tokenizers
        self._predictors = predictors
        self._serializer = default_serializer

        self._messages_processed = 0
        self._current_message_start: Optional[float] = None
        self._current_message: Optional[Message] = None

    def _load_instances_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        instances = load_instances_from_local_file(file_path)

        if self._settings.dummy_mode:
            logger.info("Running in dummy mode, slicing to only 100 instances")
            instances = instances[:100]

        if self._settings.pipeline_config.max_instances_per_message:
            instances = instances[: self._settings.pipeline_config.max_instances_per_message]

        return instances

    def _write_predictions_to_file(self, predictions: List[Dict[str, Any]], input_file_path: str) -> None:
        if self._settings.dummy_mode:
            logger.info("Running in dummy mode, not writing")
        else:
            write_predictions_to_local_file(
                predictions, input_file_path, self._settings.pipeline_config.output_file_dir
            )

    def _prepare_inputs_and_sort(
        self, enumerated_raw_instances: List[Tuple[int, Dict[str, Any]]]
    ) -> List[PreparedInputItem]:
        def text_iter(enumerated_instances):
            for index, instance in enumerated_instances:
                if "text" in instance:
                    yield RawInputItem.from_text(index, instance["text"])
                else:
                    yield RawInputItem.from_message_dicts(index, instance["chat_messages"])

        chunked_pumps = simple_chunks(
            text_iter(enumerated_raw_instances), self._settings.pipeline_config.tokenization_batch_size
        )
        prepared = flatten_and_sort(
            self._tokenizers.map_unordered(lambda toker, batch: toker.prepare_inputs.remote(batch), chunked_pumps)
        )


        return prepared

    def _predict(self, sorted_instances: List[PreparedInputItem]) -> Iterator[CompletedItem]:
        generation_batch_size = self._settings.pipeline_config.generation_batch_size

        if isinstance(generation_batch_size, int):
            chunked_tokes = simple_chunks(sorted_instances, generation_batch_size)
        else:
            chunked_tokes = prediction_batches(sorted_instances, generation_batch_size)

        predictions = flatten(
            self._predictors.map_unordered(lambda pred, batch: pred.predict.remote(batch), chunked_tokes)
        )

        for prediction in predictions:
            yield prediction

    def _decode(self, predictions: Iterable[CompletedItem]) -> Iterator[CompletedItem]:
        chunked_preds = simple_chunks(predictions, self._settings.pipeline_config.decoding_batch_size)
        decoded = flatten(
            self._tokenizers.map_unordered(lambda toker, batch: toker.decode.remote(batch), chunked_preds)
        )

        for item in decoded:
            yield item

    def _process_message(self, message: Message) -> None:
        enumerated_raw_instances = [
            (index, raw_instance)
            for index, raw_instance in enumerate(self._load_instances_from_file(message.object_key))
        ]

        prepared_and_sorted_instances = self._prepare_inputs_and_sort(enumerated_raw_instances)
        predictions = self._predict(prepared_and_sorted_instances)
        decoded_predictions = self._decode(predictions)

        decoded_map = {prediction.index: prediction for prediction in decoded_predictions}

        results = []
        for index, instance in enumerated_raw_instances:
            if index in decoded_map:
                results.append(self._serializer(instance, decoded_map[index].outputs, decoded_map[index].error))

        self._write_predictions_to_file(
            results,
            message.object_key,
        )

    def run(self) -> None:
        while True:
            if (
                self._settings.pipeline_config.max_num_messages_per_worker
                and self._messages_processed == self._settings.pipeline_config.max_num_messages_per_worker
            ):
                logger.info(
                    f"Worker finished processing {self._settings.pipeline_config.max_num_messages_per_worker} messages. Terminating..."
                )
                return

            try:
                message = ray.get(self._queue.get_message.remote())
                self._current_message = message
            except ray.exceptions.ActorDiedError:
                logger.exception("Queue Actor died")
                ray.actor.exit_actor()
            except Exception:
                logger.exception("Failure when fetching messages")

            if not message:
                logger.info("Out of messages, terminating...")
                return

            start = time.monotonic()
            self._current_message_start = start

            try:
                logger.info(f"Processing message: {message}")
                self._process_message(message)
                ray.get(self._queue.delete_message.remote(message))
                logger.info(f"Finished processing message: {message}")

            except ray.exceptions.ActorDiedError:
                logger.exception(f"An actor the worker requires has died while processing: {message}")
                ray.actor.exit_actor()
            except Exception:
                logger.exception(f"Error processing message: {message}")
            finally:
                self._current_message_start = None
                self._current_message = None
                self._messages_processed += 1
