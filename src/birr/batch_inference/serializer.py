from typing import Any, Callable, Dict, List, Optional

from birr.batch_inference.data_models import CompletionError, CompletionOutput


SerializerType = Callable[[Dict[str, Any], List[CompletionOutput], Optional[CompletionError]], Dict[str, Any]]


def default_serializer(
    input_dict: Dict[str, Any], outputs: List[CompletionOutput], completion_error: Optional[CompletionError]
) -> Dict[str, Any]:
    output = dict(**input_dict)

    if completion_error:
        output["outputs"] = None
        output["completion_error"] = completion_error.value
    else:
        output["outputs"] = [
            dict(
                index=output.index,
                text=output.text,
                token_ids=output.token_ids,
                finish_reason=output.finish_reason,
                stop_reason=output.stop_reason,
            )
            for output in outputs
        ]

    return output
