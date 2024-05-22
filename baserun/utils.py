import functools


def copy_type_hints(superclass):
    def decorator(func):
        original_func = getattr(superclass, func.__name__)
        func.__annotations__ = original_func.__annotations__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deep_merge(dicts):
    result = {}
    for dictionary in dicts:
        for key, value in dictionary.items():
            if key in result:
                # Handling both values are dicts
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge([result[key], value])
                # Handling both values are lists
                elif isinstance(result[key], list) and isinstance(value, list):
                    # If both are lists of dictionaries, merge each corresponding dict
                    if all(isinstance(item, dict) for item in result[key]) and all(
                        isinstance(item, dict) for item in value
                    ):
                        combined = []
                        for item1, item2 in zip(result[key], value):
                            if isinstance(item1, dict) and isinstance(item2, dict):
                                combined.append(deep_merge([item1, item2]))
                        # Handle excess in longer list
                        longer_list = result[key] if len(result[key]) > len(value) else value
                        combined.extend(longer_list[len(combined) :])
                        result[key] = combined
                    else:
                        result[key].extend(value)
                # Handling both values are strings
                elif isinstance(result[key], str) and isinstance(value, str):
                    result[key] += value
                # If the new value is not None and not a list, overwrite
                elif value is not None:
                    result[key] = value
            else:
                result[key] = value
    return result
