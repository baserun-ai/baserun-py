import json
from unittest.mock import patch

import pytest

from baserun import Baserun


def test_eval_includes(init_baserun):
    with patch("baserun.Baserun.submission_service.SubmitEval") as mock_submit_eval:
        eval_name = "TestEval"
        Baserun.evals.includes(eval_name, "Contains some words", ["Contains"])
        assert mock_submit_eval.call_count == 1
        args, kwargs = mock_submit_eval.call_args_list[0]
        submit_eval_request = args[0]

        assert submit_eval_request.eval.name == eval_name
        assert submit_eval_request.eval.type == "includes"
        assert submit_eval_request.eval.submission == "Contains some words"
        assert submit_eval_request.eval.result == "true"
        assert submit_eval_request.eval.score == 1
        assert submit_eval_request.eval.payload == json.dumps(
            {"expected": ["Contains"]}
        )

        assert submit_eval_request.run.name == "test_eval_includes"


def test_eval_not_includes(init_baserun):
    with patch("baserun.Baserun.submission_service.SubmitEval") as mock_submit_eval:
        eval_name = "TestEval"
        Baserun.evals.not_includes(
            eval_name, "Does not contain some words", ["contains"]
        )
        assert mock_submit_eval.call_count == 1
        args, kwargs = mock_submit_eval.call_args_list[0]
        submit_eval_request = args[0]

        assert submit_eval_request.eval.name == eval_name
        assert submit_eval_request.eval.type == "not_includes"
        assert submit_eval_request.eval.submission == "Does not contain some words"
        assert submit_eval_request.eval.result == "true"
        assert submit_eval_request.eval.score == 1
        assert submit_eval_request.eval.payload == json.dumps(
            {"expected": ["contains"]}
        )

        assert submit_eval_request.run.name == "test_eval_not_includes"


def test_eval_match(init_baserun):
    with patch("baserun.Baserun.submission_service.SubmitEval") as mock_submit_eval:
        eval_name = "TestEval"
        Baserun.evals.match(eval_name, "Matches some string", ["Match"])
        assert mock_submit_eval.call_count == 1
        args, kwargs = mock_submit_eval.call_args_list[0]
        submit_eval_request = args[0]

        assert submit_eval_request.eval.name == eval_name
        assert submit_eval_request.eval.type == "match"
        assert submit_eval_request.eval.submission == "Matches some string"
        assert submit_eval_request.eval.result == "true"
        assert submit_eval_request.eval.score == 1
        assert submit_eval_request.eval.payload == json.dumps({"expected": ["Match"]})

        assert submit_eval_request.run.name == "test_eval_match"


def test_eval_not_match(init_baserun):
    with patch("baserun.Baserun.submission_service.SubmitEval") as mock_submit_eval:
        eval_name = "TestEval"
        Baserun.evals.not_match(eval_name, "Does not match some string", ["matches"])
        assert mock_submit_eval.call_count == 1
        args, kwargs = mock_submit_eval.call_args_list[0]
        submit_eval_request = args[0]

        assert submit_eval_request.eval.name == eval_name
        assert submit_eval_request.eval.type == "not_match"
        assert submit_eval_request.eval.submission == "Does not match some string"
        assert submit_eval_request.eval.result == "true"
        assert submit_eval_request.eval.score == 1
        assert submit_eval_request.eval.payload == json.dumps({"expected": ["matches"]})

        assert submit_eval_request.run.name == "test_eval_not_match"


def test_eval_fuzzy_match(init_baserun):
    with patch("baserun.Baserun.submission_service.SubmitEval") as mock_submit_eval:
        eval_name = "TestEval"
        Baserun.evals.fuzzy_match(eval_name, "Fuzzy matches some string", ["Fuzz"])
        assert mock_submit_eval.call_count == 1
        args, kwargs = mock_submit_eval.call_args_list[0]
        submit_eval_request = args[0]

        assert submit_eval_request.eval.name == eval_name
        assert submit_eval_request.eval.type == "fuzzy_match"
        assert submit_eval_request.eval.submission == "Fuzzy matches some string"
        assert submit_eval_request.eval.result == "true"
        assert submit_eval_request.eval.score == 1
        assert submit_eval_request.eval.payload == json.dumps({"expected": ["Fuzz"]})

        assert submit_eval_request.run.name == "test_eval_fuzzy_match"


def test_eval_not_fuzzy_match(init_baserun):
    with patch("baserun.Baserun.submission_service.SubmitEval") as mock_submit_eval:
        eval_name = "TestEval"
        Baserun.evals.not_fuzzy_match(eval_name, "Fuzzy matches some string", ["Fizz"])
        assert mock_submit_eval.call_count == 1
        args, kwargs = mock_submit_eval.call_args_list[0]
        submit_eval_request = args[0]

        assert submit_eval_request.eval.name == eval_name
        assert submit_eval_request.eval.type == "not_fuzzy_match"
        assert submit_eval_request.eval.submission == "Fuzzy matches some string"
        assert submit_eval_request.eval.result == "true"
        assert submit_eval_request.eval.score == 1
        assert submit_eval_request.eval.payload == json.dumps({"expected": ["Fizz"]})

        assert submit_eval_request.run.name == "test_eval_not_fuzzy_match"


def test_eval_valid_json_valid(init_baserun):
    with patch("baserun.Baserun.submission_service.SubmitEval") as mock_submit_eval:
        eval_name = "TestEval"
        valid_json = json.dumps({"foo": "bar"})
        Baserun.evals.valid_json(eval_name, valid_json)
        assert mock_submit_eval.call_count == 1
        args, kwargs = mock_submit_eval.call_args_list[0]
        submit_eval_request = args[0]

        assert submit_eval_request.eval.name == eval_name
        assert submit_eval_request.eval.type == "valid_json"
        assert submit_eval_request.eval.submission == valid_json
        assert submit_eval_request.eval.result == "true"
        assert submit_eval_request.eval.score == 1
        assert submit_eval_request.eval.payload == "{}"

        assert submit_eval_request.run.name == "test_eval_valid_json_valid"


def test_eval_valid_json_invalid(init_baserun):
    with patch("baserun.Baserun.submission_service.SubmitEval") as mock_submit_eval:
        eval_name = "TestEval"
        invalid_json = json.dumps({"foo": "bar"}) + "}}}"
        Baserun.evals.valid_json(eval_name, invalid_json)
        assert mock_submit_eval.call_count == 1
        args, kwargs = mock_submit_eval.call_args_list[0]
        submit_eval_request = args[0]

        assert submit_eval_request.eval.name == eval_name
        assert submit_eval_request.eval.type == "valid_json"
        assert submit_eval_request.eval.submission == invalid_json
        assert submit_eval_request.eval.result == "false"
        assert submit_eval_request.eval.score == 0
        assert submit_eval_request.eval.payload == "{}"

        assert submit_eval_request.run.name == "test_eval_valid_json_invalid"


def test_eval_custom(init_baserun):
    def custom_eval(submission):
        return True

    with patch("baserun.Baserun.submission_service.SubmitEval") as mock_submit_eval:
        eval_name = "TestEval"
        Baserun.evals.custom(eval_name, "some comparison", custom_eval)
        assert mock_submit_eval.call_count == 1
        args, kwargs = mock_submit_eval.call_args_list[0]
        submit_eval_request = args[0]

        assert submit_eval_request.eval.name == eval_name
        assert submit_eval_request.eval.type == "custom"
        assert submit_eval_request.eval.submission == "some comparison"
        assert submit_eval_request.eval.result == "true"
        assert submit_eval_request.eval.score == 1
        assert submit_eval_request.eval.payload == "{}"

        assert submit_eval_request.run.name == "test_eval_custom"


@pytest.mark.asyncio
async def test_eval_custom_async(init_baserun):
    async def custom_eval(submission):
        return True

    with patch("baserun.Baserun.submission_service.SubmitEval") as mock_submit_eval:
        eval_name = "TestEval"
        await Baserun.evals.custom_async(eval_name, "some comparison", custom_eval)
        assert mock_submit_eval.call_count == 1
        args, kwargs = mock_submit_eval.call_args_list[0]
        submit_eval_request = args[0]

        assert submit_eval_request.eval.name == eval_name
        assert submit_eval_request.eval.type == "custom_async"
        assert submit_eval_request.eval.submission == "some comparison"
        assert submit_eval_request.eval.result == "true"
        assert submit_eval_request.eval.score == 1
        assert submit_eval_request.eval.payload == "{}"

        assert submit_eval_request.run.name == "test_eval_custom_async"
