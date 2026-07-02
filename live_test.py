"""Diagnostic — test JD paste with detail exposed."""
import json
import urllib.request
import urllib.error

BASE = "https://shl-rag-jv6t.onrender.com"

def post_chat(messages, timeout=45):
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
        return e.code, e.read().decode()
    except Exception as e:
        return 0, str(e)

# JD paste — this was failing
status, body = post_chat([{
    "role": "user",
    "content": "Here is the job description: We are hiring a Java backend developer responsible for Spring Boot, REST APIs, Microservices, SQL, stakeholder communication, mentoring junior engineers, and code reviews."
}])
print(f"HTTP {status}")
print(body if isinstance(body, str) else json.dumps(body, indent=2))
