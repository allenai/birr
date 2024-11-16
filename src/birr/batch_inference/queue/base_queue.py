import abc
from typing import Optional

from birr.batch_inference.data_models import Message
from birr.core.config import PipelineConfig


class BaseQueue(abc.ABC):
    def __init__(self, pipeline_config: PipelineConfig) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_message(self) -> Optional[Message]:
        """Retrieve the next message of work from this queue"""
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_message(self, message: Message) -> None:
        """Delete a message from the queue after completing the work"""
        raise NotImplementedError()
