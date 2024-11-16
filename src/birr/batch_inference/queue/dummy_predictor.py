from typing import List

from birr.batch_inference.data_models import CompletedItem, CompletionOutput, PreparedInputItem
from birr.batch_inference.predictors.base_predictor import BasePredictor


class DummyPredictor(BasePredictor):
    def predict(self, batch: List[PreparedInputItem]) -> List[CompletedItem]:
        return [
            CompletedItem(
                index=item.index,
                outputs=[
                    CompletionOutput(
                        index=0,
                        text="",
                        token_ids=item.token_ids,
                        finish_reason=None,
                        stop_reason=None,
                    )
                ],
            )
            for item in batch
        ]
