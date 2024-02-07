# Baserun


[![](https://img.shields.io/badge/Visit%20Us-baserun.ai-brightgreen)](https://baserun.ai)
[![](https://img.shields.io/badge/View%20Documentation-Docs-yellow)](https://docs.baserun.ai)
[![](https://img.shields.io/badge/Join%20our%20community-Discord-blue)](https://discord.gg/xEPFsvSmkb)
[![Twitter](https://img.shields.io/twitter/follow/baserun.ai?style=social)](https://twitter.com/baserunai)

**[Baserun](https://baserun.ai)** is the testing and observability platform for LLM apps.

# Quick Start

## 1. Install Baserun

```bash
pip install baserun
```

## 2. Set up Baserun in your application

###  Set the Baserun API key
Create an account at [https://baserun.ai](https://baserun.ai). Then generate an API key for your project in the [settings](https://baserun.ai/settings) tab. Set it as an environment variable:

```bash
export BASERUN_API_KEY="your_api_key_here"
```

Or set `baserun.api_key` to its value:

```python
baserun.api_key = "br-..."
```

###  Initialize Baserun

At some point during your application's startup you need to call `baserun.init()`. This sets up the observability system and enables Baserun. If `init` is not called, Baserun will be disabled.

## 3. Set up your traces

A trace comprises a series of events executed within an your application. Tracing enables Baserun to display an LLM chain’s entire lifecycle, whether synchronous or asynchronous.

To start tracing add the `@baserun.trace` decorator to the function you want to observe (e.g. a request/response handler or your `main` function).

Here is a simple example. In this case, Baserun is initialized at application startup and the `answer_question` function is traced. The LLM call within that function will now be traced.

```python
import sys
from openai import OpenAI
import baserun


@baserun.trace
def answer_question(question: str) -> str:
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    baserun.init()
    print(answer_question(sys.argv[-1]))
```

## 4. (Optional) Set up User Sessions

If your application involves interaction with a user and you wish to associate logs and traces with a particular user, you can use User Sessions. You can do this using `with_sessions`:

```python
from openai import OpenAI
import baserun

@baserun.trace
def use_sessions(prompt="What is the capitol of the US?") -> str:
    client = OpenAI()
    with baserun.with_session(user_identifier="example@test.com"):
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        content = completion.choices[0].message.content
        return content
```


## 5. (Optional) Set up your test suite

Use our [pytest](https://docs.pytest.org) plugin and start immediately testing with Baserun. By default all OpenAI and Anthropic requests will be automatically logged.

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

## 6. (Optional) Set up checks

Baserun supports checks (also more broadly known as "evaluations"). These are assertions that the LLM response you received matches whatever criteria you require. To use a check, you can use `baserun.check` like so:

```python
from openai import OpenAI
import baserun

client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What is the capital of the United States?"}],
)
content = completion.choices[0].message.content
baserun.check(name="capital_answer", result="Washington" in content)
```

## Further Documentation
For a deeper dive on all capabilities and more advanced usage, please refer to our [Documentation](https://docs.baserun.ai).

## License

[MIT License](https://github.com/baserun-ai/baserun-py/blob/main/LICENSE)
