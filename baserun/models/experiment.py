from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from uuid import uuid4

from datasets import Dataset
from pydantic import BaseModel, ConfigDict, Field, root_validator

if TYPE_CHECKING:
    from baserun.mixins import ClientMixin
    from baserun.models.scenario import Scenario
    from baserun.wrappers.generic import GenericClient


class Experiment(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str = Field(default_factory=lambda: str(uuid4()))
    dataset: Dataset = Field(exclude=True)
    client: Union["GenericClient", "ClientMixin"]
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    start_timestamp: Optional[datetime] = Field(default_factory=datetime.now)

    @root_validator(pre=True)
    def initialize_fields(cls, values):
        dataset = values.get("dataset")

        if "name" not in values and dataset is not None:
            values["name"] = dataset.info.dataset_name

        return values

    @property
    def scenarios(self) -> List["Scenario"]:
        from baserun.models.scenario import Scenario

        raw_scenarios = self.dataset.to_list()
        scenarios = []
        # Ensure that each row has the right fields
        for i, raw_scenario in enumerate(raw_scenarios):
            raw_scenario["experiment"] = self
            if "id" not in raw_scenario:
                raw_scenario["id"] = str(uuid4())
            if "name" not in raw_scenario:
                raw_scenario["name"] = f"{self.name} Scenario {i + 1}"
            if "input" not in raw_scenario:
                raw_scenario["input"] = {}
            if isinstance(raw_scenario.get("expected"), str):
                raw_scenario["expected"] = {"content": raw_scenario["expected"]}

            scenarios.append(Scenario(**raw_scenario))

        return scenarios

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        dumped = super().model_dump(*args, **kwargs)
        dumped["dataset"] = {
            "fingerprint": self.dataset._fingerprint,
            "name": self.dataset.info.dataset_name,
        }

        dumped["start_timestamp"] = self.start_timestamp.isoformat() if self.start_timestamp else None
        return dumped
