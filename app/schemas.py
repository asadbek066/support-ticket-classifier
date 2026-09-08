from typing import List

from pydantic import BaseModel, ConfigDict, Field


class Ticket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str = Field(default="", max_length=300)
    description: str = Field(default="", max_length=20_000)
    source_channel: str = Field(default="", max_length=32)
    customer_type: str = Field(default="", max_length=32)
    language: str = Field(default="en", max_length=16)


class ClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    confidence: float
    queue: str
    reason: str = Field(default="", max_length=2_000)
    human_review: bool


class BatchClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tickets: List[Ticket] = Field(min_length=1, max_length=100)


class BatchClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: List[ClassificationResponse]
    total: int
    processing_time_ms: float
