import json
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import uuid4

from pydantic import BaseModel, Field


class Tag(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    target_type: str
    target_id: str
    tag_type: str
    key: str
    value: str
    timestamp: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data, timestamp=datetime.now(timezone.utc))

    def model_dump(self, *args, **kwargs):
        dumped = super().model_dump()
        dumped["timestamp"] = dumped.pop("timestamp").isoformat()
        return dumped


class Variable(Tag):
    def __init__(self, **data):
        key = data.pop("name")
        super().__init__(**data, tag_type="variable", key=key)

    @property
    def name(self) -> str:
        return self.key

    @name.setter
    def name(self, value: str):
        self.key = value


class Transform(Tag):
    def __init__(self, **data):
        value = json.dumps(
            {
                "input": data.pop("input"),
                "output": data.pop("output"),
            }
        )
        super().__init__(**data, tag_type="transform", value=value)

    @property
    def input(self) -> str:
        return json.loads(self.value).get("input")

    @input.setter
    def input(self, input_value: Variable):
        self.value = json.dumps({"input": input_value, "output": self.output})

    @property
    def name(self) -> str:
        return self.key

    @name.setter
    def name(self, value: str):
        self.key = value

    @property
    def output(self) -> str:
        return json.loads(self.value).get("output")

    @output.setter
    def output(self, output_value: Variable):
        self.value = json.dumps(
            {
                "input": self.input,
                "output": output_value,
            }
        )


class Log(Tag):
    def __init__(self, **data):
        key = data.pop("name", data.pop("key", str(uuid4())))
        super().__init__(**data, tag_type="log", key=key)

    @property
    def message(self) -> str:
        return self.value

    @message.setter
    def message(self, value: str):
        self.value = value

    @property
    def name(self) -> str:
        return self.key

    @name.setter
    def name(self, value: str):
        self.key = value
