import unittest
from unittest.mock import patch

from birr.batch_inference.data_models import Message
from birr.batch_inference.queue.in_memory_queue import InMemoryQueue
from birr.core.config import PipelineConfig


class TestInMemoryQueue(unittest.TestCase):
    def test__returns_the_messages_until_exhausted(self) -> None:
        with patch("birr.batch_inference.queue.in_memory_queue.determine_remaining_files_to_process") as mock_fn:
            mock_fn.return_value = [
                "/local-dir/some/path/file1.jsonl",
                "/local-dir/some/path/file2.jsonl",
                "/local-dir/some/path/file3.jsonl",
                "/local-dir/some/path/file4.jsonl",
                "/local-dir/some/path/file5.jsonl",
                "/local-dir/some/path/file6.jsonl",
            ]

            pipeline_config = PipelineConfig(
                input_file_dir="/local-dir/some/path/input",
                output_file_dir="/local-dir/some/other/path/output",
                generation_batch_size=256,
            )

            queue = InMemoryQueue(pipeline_config)

            for index, f in enumerate(mock_fn.return_value):
                actual_message = queue.get_message()
                expected_message = Message(
                    message_id=f"item={index};file={f}",
                    receipt_handle="in-memory",
                    bucket_name="",
                    object_key=f"/local-dir/some/path/file{index+1}.jsonl",
                )

                self.assertEqual(actual_message, expected_message)

            self.assertEqual(queue.get_message(), None)
