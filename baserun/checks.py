from typing import Any, Union

from baserun import Baserun


def convert_to_score(x):
    return x if isinstance(x, int) else int(x is True)


def check(
    name: str, result: Any, payload: dict[str, Any] = None, eval_type: str = None
):
    score = convert_to_score(result)
    Baserun.evals._store_eval_data(
        name=name,
        eval_type=eval_type or "match",
        result=str(result),
        score=score,
        submission=result,
        payload=payload or {"result": result},
    )

    return result


def check_equals(name: str, actual: str, compare: Union[str, list[str]]):
    expected_list = [compare] if isinstance(compare, str) else compare
    result = any(actual.startswith(item) for item in compare)

    return check(
        name=name,
        eval_type="match",
        result=str(result).lower(),
        payload={"expected": expected_list},
    )


def check_includes(name: str, actual: str, compare: Union[str, list[str]]):
    expected_list = [compare] if isinstance(compare, str) else compare
    result = any(actual in item for item in compare)

    return check(
        name=name,
        eval_type="match",
        result=str(result).lower(),
        payload={"expected": expected_list},
    )
