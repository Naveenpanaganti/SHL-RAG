"""
Diagnose the live production error.
Temporarily exposes the real error detail by catching the raw HTTP response.
"""
import json
import urllib.request
import urllib.error

BASE = "https://shl-rag-jv6t.onrender.com"

def post_chat_raw(messages, timeout=45):
    body = json.dumps({"messages": messages}).encode()
    req = urllib.request.Request(
        f"{BASE}/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        return e.code, raw
    except Exception as e:
        return 0, str(e)

print("=== Health check ===")
try:
    with urllib.request.urlopen(f"{BASE}/health", timeout=10) as r:
        print(f"  {r.status} — {r.read().decode()}")
except Exception as e:
    print(f"  FAILED: {e}")

tests = [
    ("Simple role", [{"role": "user", "content": "Hiring a Java developer"}]),
    ("JD paste", [{"role": "user", "content": "Here is the job description: We are hiring a Java backend developer responsible for Spring Boot, REST APIs, Microservices, SQL, stakeholder communication, mentoring junior engineers, and code reviews."}]),
    ("Vague", [{"role": "user", "content": "I need an assessment"}]),
]

for label, messages in tests:
    print(f"\n=== {label} ===")
    status, body = post_chat_raw(messages)
    print(f"  HTTP {status}")
    if isinstance(body, str):
        print(f"  BODY: {body[:300]}")
    else:
        print(f"  reply: {body.get('reply','')[:120]}")
        print(f"  recommendations: {len(body.get('recommendations',[]))} items")
        print(f"  end_of_conversation: {body.get('end_of_conversation')}")
