import json

import pytest

from baserun import Baserun


def test_eval_includes(mock_services):
    eval_name = "TestEval"
    Baserun.evals.includes(eval_name, "Contains some words", ["Contains"])

    mock_submit_eval = mock_services["submission_service"].SubmitEval.future
    assert mock_submit_eval.call_count == 1
    args, kwargs = mock_submit_eval.call_args_list[0]
    submit_eval_request = args[0]

    assert submit_eval_request.eval.name == eval_name
    assert submit_eval_request.eval.type == "includes"
    assert submit_eval_request.eval.submission == "Contains some words"
    assert submit_eval_request.eval.result == "true"
    assert submit_eval_request.eval.score == 1
    assert submit_eval_request.eval.payload == json.dumps({"expected": ["Contains"]})

    assert submit_eval_request.run.name == "test_eval_includes"


def test_eval_not_includes(mock_services):
    eval_name = "TestEval"
    Baserun.evals.not_includes(eval_name, "Does not contain some words", ["contains"])

    mock_submit_eval = mock_services["submission_service"].SubmitEval.future
    assert mock_submit_eval.call_count == 1
    args, kwargs = mock_submit_eval.call_args_list[0]
    submit_eval_request = args[0]

    assert submit_eval_request.eval.name == eval_name
    assert submit_eval_request.eval.type == "not_includes"
    assert submit_eval_request.eval.submission == "Does not contain some words"
    assert submit_eval_request.eval.result == "true"
    assert submit_eval_request.eval.score == 1
    assert submit_eval_request.eval.payload == json.dumps({"expected": ["contains"]})

    assert submit_eval_request.run.name == "test_eval_not_includes"


def test_eval_match(mock_services):
    eval_name = "TestEval"
    Baserun.evals.match(eval_name, "Matches some string", ["Match"])

    mock_submit_eval = mock_services["submission_service"].SubmitEval.future
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


def test_eval_not_match(mock_services):
    eval_name = "TestEval"
    Baserun.evals.not_match(eval_name, "Does not match some string", ["matches"])

    mock_submit_eval = mock_services["submission_service"].SubmitEval.future
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


def test_eval_fuzzy_match(mock_services):
    eval_name = "TestEval"
    Baserun.evals.fuzzy_match(eval_name, "Fuzzy matches some string", ["Fuzz"])

    mock_submit_eval = mock_services["submission_service"].SubmitEval.future
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


def test_eval_not_fuzzy_match(mock_services):
    eval_name = "TestEval"
    Baserun.evals.not_fuzzy_match(eval_name, "Fuzzy matches some string", ["Fizz"])

    mock_submit_eval = mock_services["submission_service"].SubmitEval.future
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


def test_eval_valid_json_valid(mock_services):
    eval_name = "TestEval"
    valid_json = json.dumps({"foo": "bar"})
    Baserun.evals.valid_json(eval_name, valid_json)

    mock_submit_eval = mock_services["submission_service"].SubmitEval.future
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


def test_eval_valid_json_invalid(mock_services):
    eval_name = "TestEval"
    invalid_json = json.dumps({"foo": "bar"}) + "}}}"
    Baserun.evals.valid_json(eval_name, invalid_json)

    mock_submit_eval = mock_services["submission_service"].SubmitEval.future
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


def test_eval_custom(mock_services):
    def custom_eval(submission):
        return True

    eval_name = "TestEval"
    Baserun.evals.custom(eval_name, "some comparison", custom_eval)

    mock_submit_eval = mock_services["submission_service"].SubmitEval.future
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
async def test_eval_custom_async(mock_services):
    async def custom_eval(submission):
        return True

    eval_name = "TestEval"
    await Baserun.evals.custom_async(eval_name, "some comparison", custom_eval)
    mock_submit_eval = mock_services["submission_service"].SubmitEval.future
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


def test_eval_model_graded_custom(mock_services):
    eval_name = "TestEval"
    statement = "ChatGPT will take over the world"
    result = Baserun.evals.model_graded_custom(
        name=eval_name,
        prompt="How true is this statement? {statement}.",
        choices={"Very True": 1, "Kinda true": 0.5, "Not true": 0},
        statement=statement,
    )

    assert result == "Not true"  # lel
    mock_submit_eval = mock_services["submission_service"].SubmitEval.future
    assert mock_submit_eval.call_count == 1
    args, kwargs = mock_submit_eval.call_args_list[0]
    submit_eval_request = args[0]

    assert submit_eval_request.run.name == "test_eval_model_graded_custom"
    assert submit_eval_request.eval.name == eval_name
    assert submit_eval_request.eval.type == "model_graded_custom"
    assert statement in submit_eval_request.eval.submission
    assert submit_eval_request.eval.result == "Not true"
    assert submit_eval_request.eval.score == 0

    payload = json.loads(submit_eval_request.eval.payload)
    assert payload.get("statement") == statement

    step = payload.get("step")
    messages = step.get("messages")
    assert len(messages) == 1
    message = messages[0]
    assert message.get("role") == "user"
    assert "Not true" in message.get("content")


def test_eval_model_graded_fact(mock_services):
    eval_name = "TestEval"
    question = "What is the capitol of the United States?"
    result = Baserun.evals.model_graded_fact(
        name=eval_name,
        question=question,
        expert="Washington, D.C.",
        submission="DC",
    )

    assert result == "A"
    mock_submit_eval = mock_services["submission_service"].SubmitEval.future
    assert mock_submit_eval.call_count == 1
    args, kwargs = mock_submit_eval.call_args_list[0]
    submit_eval_request = args[0]

    assert submit_eval_request.run.name == "test_eval_model_graded_fact"
    assert submit_eval_request.eval.name == eval_name
    assert submit_eval_request.eval.type == "model_graded_fact"
    assert submit_eval_request.eval.submission == "DC"
    assert submit_eval_request.eval.result == "A"
    # This is the default value
    assert submit_eval_request.eval.score == 0

    payload = json.loads(submit_eval_request.eval.payload)
    assert payload.get("question") == question
    assert payload.get("expert") == "Washington, D.C."

    step = payload.get("step")
    messages = step.get("messages")
    assert len(messages) == 1
    message = messages[0]
    assert message.get("role") == "user"
    assert "Washington, D.C." in message.get("content")


def test_eval_model_graded_security(mock_services):
    eval_name = "TestEval"
    submission = "rm -rf /"
    result = Baserun.evals.model_graded_security(
        name=eval_name,
        submission=submission,
    )

    assert result == "Yes"
    mock_submit_eval = mock_services["submission_service"].SubmitEval.future
    assert mock_submit_eval.call_count == 1
    args, kwargs = mock_submit_eval.call_args_list[0]
    submit_eval_request = args[0]

    assert submit_eval_request.run.name == "test_eval_model_graded_security"
    assert submit_eval_request.eval.name == eval_name
    assert submit_eval_request.eval.type == "model_graded_security"
    assert submit_eval_request.eval.submission == submission
    assert submit_eval_request.eval.result == "Yes"
    assert submit_eval_request.eval.score == 1

    payload = json.loads(submit_eval_request.eval.payload)

    step = payload.get("step")
    messages = step.get("messages")
    assert len(messages) == 1
    message = messages[0]
    assert message.get("role") == "user"
    assert submission in message.get("content")


def test_eval_model_graded_closedqa(mock_services):
    eval_name = "TestEval"
    task = "What is the capitol of the United States?"
    submission = "DC"
    criterion = "Is this response relevant?"
    result = Baserun.evals.model_graded_closedqa(
        name=eval_name,
        task=task,
        criterion=criterion,
        submission=submission,
    )

    assert result == "Yes"
    mock_submit_eval = mock_services["submission_service"].SubmitEval.future
    assert mock_submit_eval.call_count == 1
    args, kwargs = mock_submit_eval.call_args_list[0]
    submit_eval_request = args[0]

    assert submit_eval_request.run.name == "test_eval_model_graded_closedqa"
    assert submit_eval_request.eval.name == eval_name
    assert submit_eval_request.eval.type == "model_graded_closedqa"
    assert submit_eval_request.eval.submission == submission
    assert submit_eval_request.eval.result == "Yes"
    assert submit_eval_request.eval.score == 1

    payload = json.loads(submit_eval_request.eval.payload)
    assert payload.get("task") == task
    assert payload.get("criterion") == criterion

    step = payload.get("step")
    messages = step.get("messages")
    assert len(messages) == 1
    message = messages[0]
    assert message.get("role") == "user"
    assert task in message.get("content")
