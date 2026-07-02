"""
Tests for POST /chat endpoint.

These are schema compliance tests — they verify the response shape
is correct regardless of LLM output. Actual recommendation quality
is evaluated by the SHL harness.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

# Mock the LLM so tests don't require API keys
MOCK_CLARIFY_RESPONSE = """{
  "reply": "Could you tell me more about the role?",
  "recommendations": null,
  "end_of_conversation": false
}"""

MOCK_RECOMMEND_RESPONSE = """{
  "reply": "Here are 2 assessments for you.",
  "recommendations": [
    {
      "name": "Test Assessment",
      "url": "https://www.shl.com/products/product-catalog/view/test/",
      "test_type": "K"
    }
  ],
  "end_of_conversation": false
}"""


@pytest.fixture
def client_with_mock_index():
    """
    Provide a test client with mocked vectorstore and LLM.
    Avoids needing a real catalog.json or API keys in CI.
    """
    mock_catalog = [
        {
            "name": "Test Assessment",
            "url": "https://www.shl.com/products/product-catalog/view/test/",
            "test_type": "K",
            "description": "A test assessment",
            "keys": [],
            "job_levels": [],
            "languages": [],
            "duration": "10 minutes",
        }
    ]
    mock_name_map = {item["name"].lower(): item for item in mock_catalog}

    with patch("app.vectorstore.build_index"), \
         patch("app.vectorstore.get_catalog", return_value=mock_catalog), \
         patch("app.chat.get_catalog", return_value=mock_catalog), \
         patch("app.chat.get_name_map", return_value=mock_name_map), \
         patch("app.chat.retrieve", return_value=mock_catalog), \
         patch("app.vectorstore.get_index", return_value=(None, None)):
        from app.main import app
        from fastapi.testclient import TestClient
        yield TestClient(app)


def test_chat_schema_clarify(client_with_mock_index):
    """Response must match schema when clarifying."""
    with patch("app.chat.call_llm", new_callable=AsyncMock, return_value=MOCK_CLARIFY_RESPONSE):
        response = client_with_mock_index.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "I need an assessment"}]},
        )
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "recommendations" in data
    assert "end_of_conversation" in data


def test_chat_schema_recommend(client_with_mock_index):
    """Recommendations must be a list when provided."""
    with patch("app.chat.call_llm", new_callable=AsyncMock, return_value=MOCK_RECOMMEND_RESPONSE):
        response = client_with_mock_index.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "I need a Java assessment for mid-level engineers"}
                ]
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["end_of_conversation"], bool)


def test_chat_empty_messages_returns_400():
    """Empty messages list should return 400."""
    from app.main import app
    from fastapi.testclient import TestClient
    with patch("app.vectorstore.build_index"):
        c = TestClient(app)
        response = c.post("/chat", json={"messages": []})
    assert response.status_code == 422  # Pydantic min_length validation


def test_chat_turn_cap_enforced(client_with_mock_index):
    """At 8 turns, end_of_conversation must be true."""
    messages = []
    for i in range(4):
        messages.append({"role": "user", "content": f"User message {i}"})
        messages.append({"role": "assistant", "content": f"Assistant reply {i}"})

    eoc_response = """{
      "reply": "Here is your final shortlist.",
      "recommendations": null,
      "end_of_conversation": false
    }"""

    with patch("app.chat.call_llm", new_callable=AsyncMock, return_value=eoc_response):
        response = client_with_mock_index.post(
            "/chat", json={"messages": messages}
        )
    assert response.status_code == 200
    data = response.json()
    # At exactly 8 turns, the system forces end_of_conversation = true
    assert data["end_of_conversation"] is True
