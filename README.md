# Baserun

**[Baserun](https://baserun.ai)** is an evaluation platform for LLM applications. Our goal is to simplify the testing, debugging, and evaluation of LLM features helping get your app production ready.

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

2. **Logging**: To log a test with Baserun enable the baserun.test() context and then annotate logs with the `baserun_payload` extra property. This will ensure all logs associated with a test are correlated in Baserun. 

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
