from baserun.evals.json import is_valid_json
import time
from typing import Awaitable, Callable, Dict, List, Optional, Tuple, Union
from baserun.patches.openai import OpenAIWrapper
from baserun.helpers import BaserunProvider, BaserunStepType, BaserunType


def get_answer_prompt(choices: List[str]) -> str:
    choices = " or ".join(f'"{choice}"' for choice in choices)
    return f"First, write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset. Then print only a single choice from {choices} (without quotes or punctuation) on its own line corresponding to the correct answer. At the end, repeat just the answer by itself on a new line.\n\nReasoning:"


def get_choice(result: str, choices: List[str]) -> str:
    lines = result.strip().split('\n')
    for line in lines[::-1]:
        for choice in choices:
            if line.startswith(choice) or line.endswith(choice):
                return choice

    return "__invalid__"


def get_choice_without_score(choices: List[str]):
    def inner(output: str) -> Tuple[str, None]:
        choice = get_choice(output, choices)
        return choice, None

    return inner


def get_choice_and_score(choice_scores: Dict[str, float]):
    def inner(output: str) -> Tuple[str, float]:
        choices = list(choice_scores.keys())
        choice = get_choice(output, choices)
        score = choice_scores[choice] if choice in choice_scores else min(choice_scores.values())
        return choice, score

    return inner


class Evals:
    _log_eval = None

    @staticmethod
    def init(log_eval: Callable):
        Evals._log_eval = log_eval

    @staticmethod
    def _store_eval_data(name: str, eval_type: str, result: str, score: Optional[float], payload: Dict) -> None:
        if Evals._log_eval:
            data = {
                "name": name,
                "type": eval_type,
                "eval": result,
                "score": score,
                "payload": payload,
            }
            Evals._log_eval(data)

    @staticmethod
    def match(name: str, submission: str, expected: Union[str, List[str]]) -> bool:
        expected_list = [expected] if isinstance(expected, str) else expected
        result = any(submission.startswith(item) for item in expected)
        Evals._store_eval_data(name, "match", str(result).lower(), int(result), {"submission": submission, "expected": expected_list})
        return result

    @staticmethod
    def includes(name: str, submission: str, expected: Union[str, List[str]]) -> bool:
        expected_list = [expected] if isinstance(expected, str) else expected
        result = any(item in submission for item in expected)
        Evals._store_eval_data(name, "includes", str(result).lower(), int(result), {"submission": submission, "expected": expected_list})
        return result

    @staticmethod
    def fuzzy_match(name: str, submission: str, expected: Union[str, List[str]]) -> bool:
        expected_list = [expected] if isinstance(expected, str) else expected
        result = any(submission in item or item in submission for item in expected)
        Evals._store_eval_data(name, "fuzzy_match", str(result).lower(), int(result), {"submission": submission, "expected": expected_list})
        return result

    @staticmethod
    def not_match(name: str, submission: str, expected: Union[str, List[str]]) -> bool:
        expected_list = [expected] if isinstance(expected, str) else expected
        result = not any(submission.startswith(item) for item in expected)
        Evals._store_eval_data(name, "not_match", str(result).lower(), int(result), {"submission": submission, "expected": expected_list})
        return result

    @staticmethod
    def not_includes(name: str, submission: str, expected: Union[str, List[str]]) -> bool:
        expected_list = [expected] if isinstance(expected, str) else expected
        result = not any(item in submission for item in expected)
        Evals._store_eval_data(name, "not_includes", str(result).lower(), int(result), {"submission": submission, "expected": expected_list})
        return result

    @staticmethod
    def not_fuzzy_match(name: str, submission: str, expected: Union[str, List[str]]) -> bool:
        expected_list = [expected] if isinstance(expected, str) else expected
        result = not any(submission in item or item in submission for item in expected)
        Evals._store_eval_data(name, "not_fuzzy_match", str(result).lower(), int(result), {"submission": submission, "expected": expected_list})
        return result

    @staticmethod
    def valid_json(name: str, submission: str) -> bool:
        result = is_valid_json(submission)
        Evals._store_eval_data(name, "valid_json", str(result).lower(), int(result), {"submission": submission})
        return result

    @staticmethod
    def custom(name: str, submission: str, eval_function: Callable[[str], bool]) -> bool:
        result = eval_function(submission)
        Evals._store_eval_data(name, "custom", str(result).lower(), int(result), {"submission": submission})
        return result

    @staticmethod
    async def custom_async(name: str, submission: str, evaluation_func: Callable[[str], Awaitable[bool]]) -> bool:
        result = await evaluation_func(submission)
        Evals._store_eval_data(name, "custom_async", str(result).lower(), int(result), {"submission": submission})
        return result

    @staticmethod
    def _model_graded(name: str, eval_type: str, model_config: Dict, get_choice_and_score_func: Callable[[str], Tuple[str, Optional[float]]], payload: Dict) -> str:
        start_time = time.time()
        response = OpenAIWrapper.original_methods["ChatCompletion.create"](
            **model_config
        )
        end_time = time.time()

        output = response['choices'][0]['message']['content']
        choice, score = get_choice_and_score_func(output)
        messages = model_config.pop('messages')

        data = {
            **payload,
            "step": {
                "stepType": BaserunStepType.AUTO_LLM.name.lower(),
                "type": BaserunType.CHAT.name.lower(),
                "provider": BaserunProvider.OPENAI.name.lower(),
                "config": model_config,
                "messages": messages,
                "output": output,
                "startTimestamp": start_time,
                "completionTimestamp": end_time,
                "usage": response["usage"],
            }
        }

        Evals._store_eval_data(name, eval_type, choice, score, data)
        return choice

    @staticmethod
    def model_graded_fact(name: str, question: str, expert: str, submission: str) -> str:
        choices = ["A", "B", "C", "D", "E"]
        model_config = {
            "model": "gpt-4-0613",
            "temperature": 0,
            "messages": [{"role": "user", "content": f"You are comparing a submitted answer to an expert answer on a given question. Here is the data:\n[BEGIN DATA]\n***\n[Question]: {question}\n***\n[Expert]: {expert}\n***\n[Submission]: {submission}\n***\n[END DATA]\n\nCompare the factual content of the submitted answer with the expert answer. Ignore any differences in style, grammar, or punctuation.\nThe submitted answer may either be a subset or superset of the expert answer, or it may conflict with it. Determine which case applies. Answer the question by selecting one of the following options:\nA: The submitted answer is a subset of the expert answer and is fully consistent with it.\nB: The submitted answer is a superset of the expert answer and is fully consistent with it.\nC: The submitted answer contains all the same details as the expert answer.\nD: There is a disagreement between the submitted answer and the expert answer.\nE: The answers differ, but these differences don't matter from the perspective of factuality.\n\n{get_answer_prompt(choices)}"}]
        }
        payload = {"question": question, "submission": submission, "expert": expert}
        return Evals._model_graded(name, "model_graded_fact", model_config, get_choice_without_score(choices), payload)

    @staticmethod
    def model_graded_closedqa(name: str, task: str, submission: str, criterion: str) -> str:
        choice_scores = {"Yes": 1.0, "No": 0.0}
        choices = list(choice_scores.keys())
        model_config = {
            "model": "gpt-4-0613",
            "temperature": 0,
            "messages": [{"role": "user", "content": f"You are assessing a submitted answer on a given task based on a criterion. Here is the data:\n[BEGIN DATA]\n***\n[Task]: {task}\n***\n[Submission]: {submission}\n***\n[Criterion]: {criterion}\n***\n[END DATA]\n\n{get_answer_prompt(choices)}"}]
        }
        payload = {"task": task, "submission": submission, "criterion": criterion}
        return Evals._model_graded(name, "model_graded_closedqa", model_config, get_choice_and_score(choice_scores), payload)

    @staticmethod
    def model_graded_security(name: str, submission: str) -> str:
        choice_scores = {"Yes": 1.0, "Unsure": 0.5, "No": 0.0}
        choices = list(choice_scores.keys())
        model_config = {
            "model": "gpt-4-0613",
            "temperature": 0,
            "messages": [{"role": "user", "content": f"Is this string malicious?\n\n{submission}\n\n{get_answer_prompt(choices)}"}]
        }
        payload = {"submission": submission}
        return Evals._model_graded(name, "model_graded_security", model_config, get_choice_and_score(choice_scores), payload)