from itertools import islice
import json
import os
from typing import Any, Dict, Iterable, Iterator, List, Tuple


from birr.batch_inference.data_models import PreparedInputItem


def simple_chunks(l: Iterable[Any], n: int) -> Iterator[List[Any]]:
    iterator = iter(l)
    while chunk := list(islice(iterator, n)):
        yield chunk


def prediction_batches(
    l: List[PreparedInputItem], batch_size_config: List[Tuple[int, int]]
) -> Iterator[List[PreparedInputItem]]:
    batch: List[PreparedInputItem] = []
    current_threshold_index = 0
    nearest_threshold_tokens, nearest_threshold_batch_size = batch_size_config[current_threshold_index]

    for item in l:
        num_tokens = len(item.token_ids)

        # We're either within current size bracket, or have no larger bracket to go up to.
        # Simply accumulate until the batch is full and then yield.
        if num_tokens <= nearest_threshold_tokens or current_threshold_index == len(batch_size_config) - 1:
            if len(batch) == nearest_threshold_batch_size:
                batch_to_yield = batch
                batch = [item]
                yield batch_to_yield
            else:
                batch.append(item)

        # We need to move to larger bracket and yield whatever is in current batch.
        else:
            current_threshold_index += 1
            nearest_threshold_tokens, nearest_threshold_batch_size = batch_size_config[current_threshold_index]
            batch_to_yield = batch
            batch = [item]
            yield batch_to_yield

    yield batch


def flatten(batches: Iterable[List[Any]]) -> Iterator[Any]:
    for batch in batches:
        for item in batch:
            yield item


def flatten_and_sort(token_batches: List[List[PreparedInputItem]]) -> List[PreparedInputItem]:
    acc = []
    for batch in token_batches:
        for item in batch:
            acc.append(item)

    return sorted(acc, key=lambda x: len(x.token_ids))


def load_instances_from_local_file(file_path: str) -> List[Dict[str, Any]]:
    instances = []
    with open(file_path, "r") as f:
        for line in f:
            content = line.strip()
            if content:
                instances.append(json.loads(content))
    return instances


def write_predictions_to_local_file(
    predictions: List[Dict[str, Any]], input_file_path: str, output_dir: str
) -> None:
    filename = input_file_path.split("/")[-1]
    output_file_path = os.path.join(output_dir, filename)
    data = "\n".join([json.dumps(prediction) for prediction in predictions])
    with open(output_file_path, "w") as f:
        f.write(data)


def determine_remaining_files_to_process(input_dir: str, output_dir: str) -> List[str]:
    input_dir_files = [path for path in os.listdir(input_dir) if os.path.isfile(path) and path.endswith(".jsonl")]
    output_dir_files = set(
        [path for path in os.listdir(output_dir) if os.path.isfile(path) and path.endswith(".jsonl")]
    )

    return [
        os.path.join(input_dir, input_file)
        for input_file in input_dir_files
        if input_dir_files not in output_dir_files
    ]
