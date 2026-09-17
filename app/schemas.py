import unicodedata
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

HIDDEN_CHARACTER_CATEGORIES = {"Cc", "Cf", "Zl", "Zp"}
ALLOWED_CONTROL_CHARACTERS = {"\t", "\n", "\r"}


def _reject_hidden_characters(value: str) -> str:
    """Reject control/format characters that can spoof audit or UI text."""
    for character in value:
        if character in ALLOWED_CONTROL_CHARACTERS:
            continue
        if unicodedata.category(character) in HIDDEN_CHARACTER_CATEGORIES:
            raise ValueError("value contains control or format characters")
    return value


TicketText = Annotated[str, AfterValidator(_reject_hidden_characters)]


class Ticket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: TicketText = Field(default="", max_length=300)
    description: TicketText = Field(default="", max_length=20_000)
    source_channel: TicketText = Field(default="", max_length=32)
    customer_type: TicketText = Field(default="", max_length=32)
    language: TicketText = Field(default="en", max_length=16)


class ClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    confidence: float
    queue: str
    reason: str = Field(default="", max_length=2_000)
    human_review: bool


class BatchClassificationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tickets: list[Ticket] = Field(min_length=1, max_length=100)


class BatchClassificationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[ClassificationResponse]
    total: int
    degraded: int = 0
    processing_time_ms: float
