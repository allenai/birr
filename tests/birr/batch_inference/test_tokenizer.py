import unittest
from unittest.mock import Mock, patch

from birr.batch_inference.data_models import CompletedItem, CompletionOutput
from birr.batch_inference.generate_io_processor import GenerateIOProcessor


class TestGenerateIOProcessor(unittest.TestCase):
    def test__decode(self):
        with patch("birr.batch_inference.generate_io_processor.ModelTokenizer", Mock()) as MockModelTokenizer:
            mock_model_tokenizer = Mock()
            MockModelTokenizer.return_value = mock_model_tokenizer

            def batch_decode(batch, skip_special_tokens=False):
                decoded_items = []
                for item in batch:
                    decoded_items.append(" ".join(str(token) for token in item))
                return decoded_items

            mock_model_tokenizer.batch_decode.side_effect = batch_decode

            tokenizer = GenerateIOProcessor(Mock(), Mock())

            batch = [
                CompletedItem(
                    index=0,
                    outputs=[
                        CompletionOutput(index=0, text="", token_ids=[1, 2, 3]),
                        CompletionOutput(index=1, text="", token_ids=[2, 3, 4]),
                    ],
                ),
                CompletedItem(index=1, outputs=[CompletionOutput(index=0, text="", token_ids=[3, 4, 5])]),
                CompletedItem(
                    index=2,
                    outputs=[
                        CompletionOutput(
                            index=0,
                            text="",
                            token_ids=[4, 5, 6],
                        ),
                        CompletionOutput(
                            index=1,
                            text="",
                            token_ids=[5, 6, 7],
                        ),
                        CompletionOutput(
                            index=2,
                            text="",
                            token_ids=[6, 7, 8],
                        ),
                    ],
                ),
            ]

            decoded_items = tokenizer.decode(batch)

            self.assertEqual(
                [
                    CompletedItem(
                        index=0,
                        outputs=[
                            CompletionOutput(index=0, text="1 2 3", token_ids=[1, 2, 3]),
                            CompletionOutput(index=1, text="2 3 4", token_ids=[2, 3, 4]),
                        ],
                    ),
                    CompletedItem(index=1, outputs=[CompletionOutput(index=0, text="3 4 5", token_ids=[3, 4, 5])]),
                    CompletedItem(
                        index=2,
                        outputs=[
                            CompletionOutput(
                                index=0,
                                text="4 5 6",
                                token_ids=[4, 5, 6],
                            ),
                            CompletionOutput(
                                index=1,
                                text="5 6 7",
                                token_ids=[5, 6, 7],
                            ),
                            CompletionOutput(
                                index=2,
                                text="6 7 8",
                                token_ids=[6, 7, 8],
                            ),
                        ],
                    ),
                ],
                decoded_items,
            )
