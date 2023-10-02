# Baserun


[![](https://img.shields.io/badge/Visit%20Us-baserun.ai-brightgreen)](https://baserun.ai)
[![](https://img.shields.io/badge/View%20Documentation-Docs-yellow)](https://docs.baserun.ai)
[![](https://img.shields.io/badge/Join%20our%20community-Discord-blue)](https://discord.gg/xEPFsvSmkb)
[![Twitter](https://img.shields.io/twitter/follow/baserun.ai?style=social)](https://twitter.com/baserunai)

**[Baserun](https://baserun.ai)** is the testing and observability platform for LLM apps.

## Quick Start

### 1. Install Baserun

```bash
pip install baserun
```

### 2. Generate an API key
Create an account at [https://baserun.ai](https://baserun.ai). Then generate an API key for your project in the [settings](https://baserun.ai/settings) tab. Set it as an environment variable:

```bash
export BASERUN_API_KEY="your_api_key_here"
```

Or set `baserun.api_key` to its value:

```python
baserun.api_key = "br-..."
```

### 3. Start testing

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

### Production usage

You can use Baserun for production observability as well. To do so, simply call `baserun.init()` somewhere during your application's startup, and add the `@baserun.trace` decorator to the function you want to observe (e.g. a request/response handler). For example,

```python
import sys
import openai
import baserun


@baserun.trace
def answer_question(question: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": question}],
    )
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":
    baserun.init()
    print(answer_question(sys.argv[-1]))
```

## Documentation
For a deeper dive on all capabilities and more advanced usage, please refer to our [Documentation](https://docs.baserun.ai).

## Contributing

Contributions to baserun-py are welcome! Below are some guidelines to help you get started.

### Dependencies
Install the dependencies:
```bash
pip install -r requirements.txt
```

Install the dev dependencies with:
```bash
pip install -r requirements-dev.txt
```

### Tests

You can run tests using `pytest`. Note is that in pytest the remote server is mocked, so network requests are not actually made to Baserun's backend.

If you want to emulate production tracing, we have a utility for that:

```bash
python tests/testing_functions.py {function_to_test}
```

Take a look at the list of functions in that file: any function with the `@baserun.trace` decorator can be used.

### gRPC and Protobuf
If you're making changes to `baserun.proto`, you'll need to compile those changes. Run the following command:

```
python -m grpc_tools.protoc -Ibaserun --python_out=baserun --pyi_out=baserun --grpc_python_out=baserun baserun/v1/baserun.proto
```

### A Note on Breaking Changes
Be cautious when making breaking changes to protobuf definitions. These could impact backward compatibility and require corresponding server-side changes, so be sure to discuss it with our maintainers.

## License

[MIT License](https://github.com/baserun-ai/baserun-py/blob/main/LICENSE)
