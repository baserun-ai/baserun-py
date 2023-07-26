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

**Log with the decorator**: Simply use `@baserun.test` as a decorator for your test functions. All baserun logs within the decorated function will be sent to Baserun as part of your test run.

```python
import baserun

@baserun.test
def test_my_function():
    ...
    baserun.log("Your log message")
```

## Running tests with pytest
To execute your tests and send logs to Baserun, simply use the --baserun flag with pytest.

```bash
pytest --baserun your_test_module.py
```

## Running tests with unittest
When you derive your test case from `BaserunTestCase`, there's no need to use the `@baserun.test` decorator. All test methods within a class that inherits from `BaserunTestCase` will automatically send logs to Baserun.
```python
import baserun

class YourTest(baserun.BaserunTestCase):
    
    def test_example(self):
        ...
        baserun.log("Your unittest log message")
```

To run your unittests:

```bash
python -m unittest your_unittest_module.py
```

## Using Baserun CLI
To run a Python module with Baserun logging enabled:

```bash
baserun your_module_name
```

## Explicit Initialization
For cases where you want more control:
```python
import baserun

baserun.init(api_url='YOUR_API_URL')

@baserun.test
def some_function():
    ...
    baserun.log("Your log message for explicit initialization")
    
some_function()

# At the end of your script or before exit
baserun.flush()
```