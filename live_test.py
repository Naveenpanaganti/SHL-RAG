"""Quick diagnostic against the live Render deployment."""
import json
import urllib.request
import urllib.error

BASE = "https://shl-rag-jv6t.onrender.com"

def post_chat(messages):
    body = json.dumps({"messages": messages}).encode()
    req = urllib.request.Request(
        f"{BASE}/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

# Health
with urllib.request.urlopen(f"{BASE}/health", timeout=10) as r:
    print(f"GET /health → {r.status} {r.read().decode()}")

# Chat
status, body = post_chat([{"role": "user", "content": "I need an assessment for a Java developer"}])
print(f"POST /chat → {status}")
print(json.dumps(body, indent=2) if isinstance(body, dict) else body[:500])
