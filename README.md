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
import openai

def test_paris_trip():
    activities_prompt = (
        "What are three activities to do in Paris? Return CSV of the activities:"
    )
    activities_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=[
            {
                "role": "User",
                "content": activities_prompt
            }
        ],
    )
    
    activities_csv = activities_response['choices'][0]['message']['content']
    
    itinerary_prompt = (
        f"Plan a concise, one day itinerary for these activities: {activities_csv}"
    )
    itinerary_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=[
            {
                "role": "User", 
                "content": itinerary_prompt
            }
        ],
    )
    
    return itinerary_response['choices'][0]['message']['content']
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

### log_llm_chat
Logs an interaction with a LLM chat API to Baserun during a test run.

#### Parameters
* config (dict): A dictionary containing the configuration for the LLM model, including the model and any other relevant parameters.
* messages (list of dict): A list of messages representing the conversation between the user and the LLM model. Each message is represented as a dictionary with keys role and content, where role can be "system", "user", or "assistant", and content is the text of the message. Use prompt templates (e.g., "{variable}") in the content to specify variables that will be substituted at runtime. 
* output (str): The output response from the LLM model. 
* variables (dict, optional): A dictionary of variables used in the chat messages.

```python
import baserun
import openai


@baserun.test
def log_llm_chat_example():
    config = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7
    }

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in {year}?"}
    ]

    variables = {"year": "2020"}

    response = openai.ChatCompletion.create(
        **config,
        messages=[
            {"role": message["role"], "content": message["content"].format(variables)}
            for message in messages
        ],
    )

    baserun.log_llm_chat(
        config=config,
        messages=messages,
        output=response['choices'][0]['message']['content'],
        variables=variables
    )
```

### log_llm_completion
Logs an interaction with a LLM completion API to Baserun during a test run.

#### Parameters
* config (dict): A dictionary containing the configuration for the LLM model, including the model and any other relevant parameters.
* prompt (str): The input prompt for the LLM model. Use prompt templates (e.g., "{variable}") in the prompt to specify variables that will be substituted at runtime.
* output (str): The output response from the LLM model. 
* variables (dict, optional): A dictionary of variables used in the completion prompt.

```python
import baserun
import openai

@baserun.test
def log_llm_completion_example():
    config = {
        "model": "text-davinci-003",
        "temperature": 0.8,
        "max_tokens": 100
    }
    prompt = "Once upon a time, there was a {character} who {action}."
    variables = {"character": "brave knight", "action": "fought dragons"}

    response = openai.Completion.create(
        **config,
        prompt=prompt.format(variables),
    )

    baserun.log_llm_completion(
        config=config,
        prompt=prompt,
        output=response['choices'][0]['text'],
        variables=variables
    )
```

## Running tests with pytest
To execute your tests and send logs to Baserun, simply use the --baserun plugin with pytest. When using the --baserun plugin each test will automatically group logs together and then send them to Baserun.

```bash
pytest --baserun your_test_module.py
```

## Running tests with unittest
When you derive your test case from `BaserunTestCase` all test methods will automatically group logs together and then send them to Baserun.
```python
import baserun

class YourTest(baserun.BaserunTestCase):
    
    def test_example(self):
        ...
        baserun.log("ExampleEvent", payload={"key": "value"})
```

To run your unittests:

```bash
python -m unittest your_unittest_module.py
```

## Custom tests
In order to group logs together for custom tests use the `@baserun.test` decorator which will group all logs within the decorated function and then send them to Baserun once the test run completes.

```python
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
baserun your_module_name
```

The Baserun CLI will handle initializing itself and then flushing logs to Baserun. 

### Explicit Initialization and Flushing
For cases where you want more control:
```python
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