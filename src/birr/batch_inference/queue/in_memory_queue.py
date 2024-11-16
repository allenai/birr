from collections import deque
from typing import Deque, Optional

from birr.batch_inference.data_models import Message
from birr.batch_inference.queue.base_queue import BaseQueue
from birr.batch_inference.utils import determine_remaining_files_to_process
from birr.core.config import PipelineConfig


class InMemoryQueue(BaseQueue):
    def __init__(self, pipeline_config: PipelineConfig) -> None:
        remaining_files_to_process = determine_remaining_files_to_process(
            input_dir=pipeline_config.input_file_dir, output_dir=pipeline_config.output_file_dir
        )

        self._queue: Deque[Message] = deque()
        for index, f in enumerate(remaining_files_to_process):
            self._queue.appendleft(
                Message(
                    message_id=f"item={index};file={f}",
                    receipt_handle="in-memory",
                    bucket_name="",
                    object_key=f,
                )
            )

    def get_message(self) -> Optional[Message]:
        if len(self._queue):
            return self._queue.pop()
        return None

    def delete_message(self, message: Message) -> None:
        pass
