import os
import unittest

from birr.batch_inference import utils
from birr.batch_inference.data_models import PreparedInputItem


class TestUtils(unittest.TestCase):
    def test__chunks_into_requested_slices(self) -> None:
        to_chunk = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        chunks = list(utils.simple_chunks(to_chunk, 3))

        self.assertEqual(chunks, [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]])

    def test__prediction_batches(self) -> None:
        batch_size_config = [(1, 4), (4, 3), (8, 1)]

        items = [
            PreparedInputItem(index=0, token_ids=[1]),  # begin batch 1
            PreparedInputItem(index=1, token_ids=[1]),
            PreparedInputItem(index=2, token_ids=[1]),
            PreparedInputItem(index=3, token_ids=[1]),
            PreparedInputItem(index=4, token_ids=[1]),  # begin batch 2
            PreparedInputItem(index=5, token_ids=[1]),
            PreparedInputItem(index=6, token_ids=[1]),
            PreparedInputItem(index=7, token_ids=[1, 2]),  # begin batch 3
            PreparedInputItem(index=8, token_ids=[1, 2]),
            PreparedInputItem(index=9, token_ids=[1, 2]),
            PreparedInputItem(index=10, token_ids=[1, 2, 3, 4]),  # begin batch 4
            PreparedInputItem(index=11, token_ids=[1, 2, 3, 4, 5]),  # begin batch 5
            PreparedInputItem(index=12, token_ids=[1, 2, 3, 4, 5, 6, 7, 8]),  # begin batch 6
            PreparedInputItem(index=13, token_ids=[1, 2, 3, 4, 5, 6, 7, 8]),  # begin batch 7
        ]

        batches = list(utils.prediction_batches(items, batch_size_config))

        expected_batches = [
            [
                PreparedInputItem(index=0, token_ids=[1]),
                PreparedInputItem(index=1, token_ids=[1]),
                PreparedInputItem(index=2, token_ids=[1]),
                PreparedInputItem(index=3, token_ids=[1]),
            ],
            [
                PreparedInputItem(index=4, token_ids=[1]),  # begin batch 2
                PreparedInputItem(index=5, token_ids=[1]),
                PreparedInputItem(index=6, token_ids=[1]),
            ],
            [
                PreparedInputItem(index=7, token_ids=[1, 2]),  # begin batch 3
                PreparedInputItem(index=8, token_ids=[1, 2]),
                PreparedInputItem(index=9, token_ids=[1, 2]),
            ],
            [
                PreparedInputItem(index=10, token_ids=[1, 2, 3, 4]),  # begin batch 4
            ],
            [
                PreparedInputItem(index=11, token_ids=[1, 2, 3, 4, 5]),  # begin batch 5
            ],
            [
                PreparedInputItem(index=12, token_ids=[1, 2, 3, 4, 5, 6, 7, 8]),  # begin batch 6
            ],
            [PreparedInputItem(index=13, token_ids=[1, 2, 3, 4, 5, 6, 7, 8])],  # begin batch 7
        ]

        self.assertEqual(batches, expected_batches)

    def test__flatten_flattens(self) -> None:
        to_flatten = [[1, 2, 3, 4], [5, 6], [7], [8, 9, 10]]

        flattened = list(utils.flatten(to_flatten))

        self.assertEqual(flattened, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test__flattens_and_sorts_ascendingly_by_num_tokens(self) -> None:
        batches = [
            [
                PreparedInputItem(index=0, token_ids=[1] * 4),
                PreparedInputItem(index=1, token_ids=[1] * 3),
                PreparedInputItem(index=2, token_ids=[1] * 5),
            ],
            [
                PreparedInputItem(index=3, token_ids=[1] * 1),
                PreparedInputItem(index=4, token_ids=[1] * 6),
            ],
            [
                PreparedInputItem(index=5, token_ids=[1] * 2),
            ],
        ]

        flattened_and_sorted = list(utils.flatten_and_sort(batches))

        indices = [item.index for item in flattened_and_sorted]

        self.assertEqual(indices, [3, 5, 1, 0, 2, 4])
