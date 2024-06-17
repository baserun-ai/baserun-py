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

## 2. Set the Baserun API key
Create an account at [https://baserun.ai](https://baserun.ai). Then generate an API key for your project in the [settings](https://baserun.ai/settings) tab. Set it as an environment variable:

```bash
export BASERUN_API_KEY="your_api_key_here"
```

## Usage

In order to have Baserun trace your LLM Requests, all you need to do is import `OpenAI` from `baserun` instead of `openai`. Creating an OpenAI client object automatically starts the trace, and all future LLM requests made with this client object will be captured.

<CodeGroup>

```python python
from baserun import OpenAI


def example():
    client = OpenAI()
    completion = client.chat.completions.create(
        name="Paris Activities",
        model="gpt-4o",
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": "What are three activities to do in Paris?"
            }
        ],
    )


if __name__ == "__main__":
    print(example())
```

### Alternate init method

If, for some reason, you don't wish to use Baserun's OpenAI client, you can simply wrap your normal OpenAI client using `init`.

```python python
from baserun import init

client = init(OpenAI())
```

</CodeGroup>

## Configuring the trace

When you start a trace by initializing an OpenAI object, there are several _optional_ parameters you can set for that trace:

- `name`: A customized name for the trace
- `result`: Some end result or output for the trace
- `user`: A username or user ID to associate with this trace.
- `session`: A session ID to associate with this trace.
- `trace_id`: A previously-generated or custom UUID (e.g. to continue a previous trace)

```python
from baserun import OpenAI

def example():
    client = OpenAI(result="What are three activities to do in Paris?")
    client.name = "Example"
    client.user = "user123"
    client.session = "session123"

    completion = client.chat.completions.create(
        name="Paris Activities",
        model="gpt-4o",
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": "What are three activities to do in Paris?"
            }
        ],
    )
    client.result = "Done"
```

## Evals

### Evaluating Completions

You can perform evals directly on a completion object. The `includes` eval is used here as an example, and checks if a string is included in the completion's output. The argument passed to `eval()` is a name or label used for your reference.

```python
from baserun import OpenAI

def example():
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": "What are three activities to do in Paris?"
            }
        ],
    )
    client.eval("include_eiffel_tower").includes("Eiffel Tower")
```

## Tags

You can add tags either to the traced OpenAI object or to the completion. There are several different types of tags:

- `log`: Any arbitrary logs you want to attach to a trace or completion
- `feedback`: Any score-based feedback given from users (e.g. thumbs up/down, star rating)
- `variable`: Any variables used, e.g. while rendering a template
- `custom`: Any arbitrary attributes you want to attach to a trace or completion

Each tag type has functions on traced OpenAI objects and completions. Each tag function can accept a `metadata` parameter which is an arbitrary dictionary with any values you might want to capture.

```python
from baserun import OpenAI

def example():
    client = OpenAI()
    client.log("Gathering user input")
    city = input()
    completion = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": f"What are three activities to do in {city}?"
            }
        ],
    )
    completion.variable("city", city)
    user_score = input()
    client.feedback("User Score", score=user_score, metadata={"My key": "My value"})
```

### Adding tags to a completed trace or completion

After a trace has been completed you may wish to add additional tags to a trace or completion. For example, you might have user feedback that is gathered well after the fact. To add these tags, you need to store the `trace_id`, and, if the tag is for a completion, the `completion_id`. You can then use the `tag`, `log`, or `feedback` functions to submit those tags.

```python
from baserun import OpenAI, log, feedback

client = OpenAI(name="trace to be resumed")
completion = client.chat.completions.create(
    name="completion to be resumed",
    model="gpt-4o",
    messages=[{"role": "user", "content": "What are three activities to do in Paris?"}],
)

# Store these values
trace_id = client.trace_id
completion_id = completion.completion_id

# A few moments later...
log("Tagging resumed", trace_id=trace_id, completion_id=completion_id)
feedback("User satisfaction", 0.9, trace_id=trace_id, completion_id=completion_id)
```

## Unsupported models
Baserun ships with support for OpenAI and Anthropic. If you use another provider or library, you can still use Baserun by manually creating "generic" objects. Notably, generic completions _must be submitted_ explicitly using `submit_to_baserun`. Here's what that looks like:

```python
question = "What is the capital of the US?"
response = call_my_custom_model(question)

client = GenericClient(name="My Traced Client")
completion = GenericCompletion(
    model="my custom model",
    name="My Completion",
    input_messages=[GenericInputMessage(content=question, role="user")],
    choices=[GenericChoice(message=GenericCompletionMessage(content=response))],
    client=client,
    trace_id=client.trace_id,
)
completion.submit_to_baserun()
```

## Datasets

Baserun has built-in support for the `datasets` library by HuggingFace. You can use the `Dataset` class to submit datasets. See the [HuggingFace documentation](https://huggingface.co/docs/datasets/index) to learn more about the `datasets` library.

Once you have loaded your dataset, you can submit it to Baserun by using the `submit_dataset` function.

```python
from datasets import Dataset
from baserun import submit_dataset


data_samples = {
    "question": ["When was the first super bowl?"],
    "answer": ["The first Super Bowl was held on January 15, 1967. It took place at the Los Angeles Memorial Coliseum in Los Angeles, California."],
    "contexts": [
        [
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,"
        ],
    ],
    "ground_truth": [
        "The first Super Bowl was held on January 15, 1967",
    ],
}

dataset = Dataset.from_dict(data_samples)
submit_dataset(dataset, "questions")
```

### Using Datasets for evals

Once you have submitted a dataset, you can use the `get_dataset` function to retrieve it. The retrieved dataset can automatically create scenarios from your data. From there, you can easily evaluate these scenarios by using the `evaluate` function:

```python
from baserun import OpenAI, evaluate

dataset = await get_dataset(name="capital questions")
question = "What is the capital of {country}?"

client = OpenAI()
experiment = Experiment(dataset=dataset, client=client, name="Dataset online eval run")
for scenario in experiment.scenarios:
    evaluators = [Includes(scenario=scenario, expected="{city}"), Correctness(scenario=scenario, question=question)]

    completion = client.chat.completions.create(
        name=scenario.name,
        model="gpt-4o",
        messages=scenario.format_messages([{"role": "user", "content": question}]),
        variables=scenario.input,
    )
    output = completion.choices[0].message.content
    client.output = output
    scenario.actual = output

    evaluate(evaluators, scenario, completion=completion)
```


## Further Documentation
For a deeper dive on all capabilities and more advanced usage, please refer to our [Documentation](https://docs.baserun.ai).

## License

[MIT License](https://github.com/baserun-ai/baserun-py/blob/main/LICENSE)
