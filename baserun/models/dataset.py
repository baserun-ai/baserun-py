from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class DatasetVersionMetadata(BaseModel):
    id: str
    creation_timestamp: datetime


class DatasetMetadata(BaseModel):
    id: str
    name: str
    length: int
    versions: List[DatasetVersionMetadata] = Field(default_factory=list)
    features: List[str] = Field(default_factory=list)
    creation_timestamp: datetime
    created_by: str
