from typing import List, Optional

from pydantic import BaseModel

from baserun.wrappers.generic import GenericInputMessage


class Experiment(BaseModel):
    id: str
    dataset_id: str
    dataset_version_id: Optional[str] = None


class ExperimentCase(BaseModel):
    id: str
    experiment_id: str
    input_messages: List[GenericInputMessage]
