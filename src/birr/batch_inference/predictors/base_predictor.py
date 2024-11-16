from typing import Any, List

from abc import ABC, abstractmethod

from birr.batch_inference.data_models import CompletedItem, PreparedInputItem
from birr.core.config import GenerateConfig, LLMModelConfig


class BasePredictor(ABC):
    def __init__(
        self,
        model_config: LLMModelConfig,
        generate_config: GenerateConfig,
    ):
        self._model_config = model_config
        self._generate_config = generate_config
        self._model = self._load_model()

    @abstractmethod
    def predict(self, batch: List[PreparedInputItem]) -> List[CompletedItem]:
        raise NotImplementedError

    def _load_model(self) -> Any:
        return None
