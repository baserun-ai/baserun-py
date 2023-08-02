# Baserun

**[Baserun](https://baserun.ai)** is the collaborative workspace for AI teams. Our mission is to simplify the testing, debugging, and evaluation of LLM features to help you get your app production-ready.

## Quick Start

Install `baserun`

```bash
pip install baserun
```

Get your API key from the [Baserun dashboard](https://baserun.ai/settings) and set it as an environment variable:

```bash
export BASERUN_API_KEY="your_api_key_here"
```

Use our pytest plugin and start immediately logging to Baserun. By default all OpenAI completion and chat requests will be logged to Baserun. Logs are aggregated by test.

```python
# test_module.py

import openai

def test_paris_trip():
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": "What are three activities to do in Paris?"
            }
        ],
    )
    
    assert "Eiffel Tower" in response['choices'][0]['message']['content']
```

To run the test and log to baserun:

```bash
pytest --baserun test_module.py
...
========================Baserun========================
Test results available at: https://baserun.ai/runs/<id>
=======================================================
```

## Custom logs

### log
Logs a custom message to Baserun. If Baserun is not initialized, this function will have no effect.

#### Parameters
* message (str): The custom log message to be recorded.
* payload (Union[str, Dict]): The log's additional data, which can be either a string or a dictionary.

```python
import baserun

def test_custom_log():
    ...
    baserun.log("CustomEvent", payload={"key": "value"})
```