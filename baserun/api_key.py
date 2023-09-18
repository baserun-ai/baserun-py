def get_api_key():
    from . import api_key
    if not api_key:
        raise ValueError("No BASERUN_API_Key provided. You can set your API key in code using 'baserun.api_key = <API-KEY>', or you can set the environment variable BASERUN_API_KEY=<API-KEY>")
    return api_key
