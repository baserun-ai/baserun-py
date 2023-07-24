# Baserun

**[Baserun](https://baserun.ai)** is an evaluation platform for LLM applications. Our goal is to simplify the testing, debugging, and evaluation of LLM features helping make your app production ready.

## Installation

To install `baserun`, use pip:

```bash
pip install baserun
```

## Quick Start

1. **Initialization**: Begin by initializing Baserun with your API key:

```python
import baserun

baserun.init(api_key="YOUR_API_KEY")
```

2. **Logging**: 

* **Using as a decorator**: To log a test with Baserun, use baserun.test() as a decorator for your functions. The function name will be automatically inferred and added to the metadata, but you can also provide additional metadata if needed. Any log within the callstack provided with the extra baserun_payload property will be sent to Baserun.
```python
import baserun
import logging

@baserun.test(metadata={"custom_key": "custom_value"})
def my_function():
    logger = logging.getLogger(__name__)
    logger.info("Your log message", extra={"baserun_payload": {
        "input": "What is the capital of the United States?",
        "output": "Washington, D.C."
    }})

my_function()
```

* **Using Context Manager**: you can also use baserun.test() as a context. Likewise any logs generated within that context that are annotated with the `baserun_payload` property will be sent to Baserun. 

```python
import baserun
import logging

with baserun.test():
    logger = logging.getLogger(__name__)
    logger.info("Your log message", extra={"baserun_payload": {
        "input": "What is the capital of the United States?",
        "output": "Washington, D.C."
    }})
```

## Configuration

- **Setting Log Level**: Ensure your logging level is set to `INFO` to capture and forward the relevant log messages:

```python
logging.basicConfig(level=logging.INFO)
```
