# Baserun

**[Baserun](https://baserun.ai)** is an evaluation platform for LLM applications. Our mission is to simplify the testing, debugging, and evaluation of LLM features, helping you get your app production-ready.

## Installation

To install `baserun`, use pip:

```bash
pip install baserun
```

## Setting Up Your API Key

Before running your tests, you'll need to set up the `BASERUN_API_KEY` environment variable with the API key from your Baserun dashboard:

For UNIX-based systems (Linux/MacOS):
```bash
export BASERUN_API_KEY='your_api_key_here'
```

For Windows:
```bash
setx BASERUN_API_KEY "your_api_key_here"
```
This will ensure that your tests can authenticate and send data to Baserun.

## Quick Start

**Automatic logging with pytest and openai**: by default all openai completion and chat requests will be logged to Baserun.
Logs will be aggregated by test and openai logs will include latency and token usage.

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
    
    return response['choices'][0]['message']['content']
```

To run the test and log to baserun:

```bash
pytest --baserun test_module.py
```

## Baserun API

### log
Logs a custom message to Baserun during a test run.

#### Parameters
* message (str): The custom log message to be recorded.
* payload (dict, optional): Additional data associated with the event. Defaults to None.

```python
import baserun

def some_function():
    baserun.log("SomeEvent", payload={"key": "value"})
```


## Running tests with pytest
To execute your tests and send logs to Baserun, simply use the --baserun plugin with pytest. When using the --baserun plugin each test will automatically group logs together and then send them to Baserun.

```bash
pytest --baserun your_test_module.py
```

## Running tests with unittest
When you derive your test case from `BaserunTestCase` all test methods will automatically group logs together and then send them to Baserun.
```python
# unittest_module.py

import baserun

class YourTest(baserun.BaserunTestCase):
    
    def test_example(self):
        ...
        baserun.log("ExampleEvent", payload={"key": "value"})
```

To run your unittests:

```bash
python -m unittest unittest_module.py
```

## Custom tests
In order to group logs together for custom tests use the `@baserun.test` decorator which will group all logs within the decorated function and then send them to Baserun once the test run completes.

```python
# custom_module.py

import baserun

@baserun.test
def custom_test():
    ...
    baserun.log("CustomEvent", payload={"key": "value"})

custom_test()
```

### Using Baserun CLI
To run a Python module with custom tests:

```bash
baserun custom_module
```

The Baserun CLI will handle initializing itself and then flushing logs to Baserun. 

### Explicit Initialization and Flushing
For cases where you want more control:
```python
# custom_module.py

import baserun

baserun.init(api_url='YOUR_API_URL')

@baserun.test
def custom_test():
    ...
    baserun.log("CustomEvent", payload={"key": "value"})
    
custom_test()

# At the end of your script or before exit
baserun.flush()
```

To run your custom script:

```bash
python custom_module.py
```