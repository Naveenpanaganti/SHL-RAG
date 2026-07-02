"""
Pydantic models for request/response schema.
Schema is non-negotiable per assignment spec — do not alter field names.

Spec says:
- recommendations is EMPTY [] when gathering context or refusing (NOT null)
- recommendations is array of 1-10 when committed to a shortlist
- end_of_conversation is true only when agent considers task complete
"""

from typing import List
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str = Field(..., description="Single letter code(s): A, B, C, D, K, P, S")


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(
        default_factory=list,
        description="Empty list when clarifying/refusing; 1-10 items when committed"
    )
    end_of_conversation: bool = False
