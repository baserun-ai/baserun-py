# Baserun


[![](https://img.shields.io/badge/Visit%20Us-baserun.ai-brightgreen)](https://baserun.ai)
[![](https://img.shields.io/badge/View%20Documentation-Docs-yellow)](https://docs.baserun.ai)
[![](https://img.shields.io/badge/Join%20our%20community-Discord-blue)](https://discord.gg/xEPFsvSmkb)
[![Twitter](https://img.shields.io/twitter/follow/baserun.ai?style=social)](https://twitter.com/baserunai)

**[Baserun](https://baserun.ai)** is the testing and observability platform for LLM apps.

## Quick Start

### 1. Generate an API key
Create an account at [https://baserun.ai](https://baserun.ai). Then generate an API key for your project in the [settings](https://baserun.ai/settings) tab and set it as an environment variable.

```bash
export BASERUN_API_KEY="your_api_key_here"
```

### 2. Install Baserun

```bash
pip install baserun
```

### 3. Start testing

```bash
export BASERUN_API_KEY="your_api_key_here"
```

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

## Documentation
For a deeper dive on all capabilities and more advanced usage, please refer to our [Documentation](https://docs.baserun.ai).

## License

[MIT License](https://github.com/baserun-ai/baserun-py/blob/main/LICENSE)