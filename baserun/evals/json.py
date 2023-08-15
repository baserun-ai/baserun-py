import json


def is_valid_json(output: str) -> bool:
    try:
        json.loads(output)
        return True
    except ValueError:
        return False
