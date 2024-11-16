from typing import Any, Dict, List
import unittest

from birr.batch_inference.serializer import default_serializer
from birr.batch_inference.data_models import CompletionOutput, CompletionError


def mk_input() -> Dict[str, Any]:
    return dict(text="asdfasdfasdf", id="123", metadata=dict(foo="bar"))


def mk_completion_outputs() -> List[CompletionOutput]:
    return [
        CompletionOutput(index=0, text="I am Fred", token_ids=[1, 2, 3], finish_reason="foo", stop_reason="bar"),
        CompletionOutput(
            index=1, text="You are Lucille", token_ids=[4, 5, 6], finish_reason="fizz", stop_reason="buzz"
        ),
    ]


class TestSerializers(unittest.TestCase):
    def test__default_serializer(self) -> None:
        result = default_serializer(mk_input(), mk_completion_outputs(), None)

        self.assertEqual(
            dict(
                text="asdfasdfasdf",
                id="123",
                metadata=dict(foo="bar"),
                outputs=[
                    dict(index=0, text="I am Fred", token_ids=[1, 2, 3], finish_reason="foo", stop_reason="bar"),
                    dict(
                        index=1,
                        text="You are Lucille",
                        token_ids=[4, 5, 6],
                        finish_reason="fizz",
                        stop_reason="buzz",
                    ),
                ],
            ),
            result,
        )

    def test__default_serializer_handles_completion_errors(self) -> None:
        result = default_serializer(mk_input(), [], CompletionError.CONTEXT_TOO_LONG)

        self.assertEqual(
            dict(
                text="asdfasdfasdf",
                id="123",
                metadata=dict(foo="bar"),
                outputs=None,
                completion_error="CONTEXT_TOO_LONG",
            ),
            result,
        )
