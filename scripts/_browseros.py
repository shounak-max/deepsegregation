"""Helper script to interact with BrowserOS neo MCP endpoint."""

import json
import os
import sys
import urllib.request

SESS_FILE = os.path.expanduser("~/.gemini/antigravity-ide/brain/85194025-ab2a-4dd3-b902-4ec2e66e7c4e/scratch/neo_sess.json")
MCP_URL = "http://127.0.0.1:9010/mcp"


def get_session():
    if os.path.exists(SESS_FILE):
        try:
            with open(SESS_FILE, "r") as f:
                return json.load(f).get("session_id")
        except Exception:
            pass
    return None


def save_session(session_id):
    os.makedirs(os.path.dirname(SESS_FILE), exist_ok=True)
    with open(SESS_FILE, "w") as f:
        json.dump({"session_id": session_id}, f)


def run_code(js_code, timeout=30):
    session_id = get_session()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if session_id:
        headers["mcp-session-id"] = session_id
    else:
        # Initialize
        init_body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "antigravity", "version": "1.0"},
            },
        }
        req = urllib.request.Request(
            MCP_URL, data=json.dumps(init_body).encode("utf-8"), headers=headers
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            session_id = resp.headers.get("mcp-session-id")
            save_session(session_id)
            headers["mcp-session-id"] = session_id

        # Notification
        notif_body = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        req = urllib.request.Request(
            MCP_URL, data=json.dumps(notif_body).encode("utf-8"), headers=headers
        )
        urllib.request.urlopen(req, timeout=10)

    # Call run tool
    call_body = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "tools/call",
        "params": {"name": "run", "arguments": {"code": js_code, "timeout": timeout * 1000}},
    }
    req = urllib.request.Request(
        MCP_URL, data=json.dumps(call_body).encode("utf-8"), headers=headers
    )
    with urllib.request.urlopen(req, timeout=timeout + 5) as resp:
        session_id = resp.headers.get("mcp-session-id", session_id)
        if session_id:
            save_session(session_id)
        raw = resp.read().decode("utf-8")
        for line in raw.splitlines():
            if line.startswith("data: "):
                try:
                    payload = json.loads(line[6:])
                    if "result" in payload:
                        return payload["result"]
                    if "error" in payload:
                        return payload
                except Exception:
                    pass
        return {"raw": raw}


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    if action == "inspect":
        res = run_code(
            """
        const pages = await browser.pages.list();
        const claudePage = pages.find(p => p.url.includes('claude.ai'));
        if (!claudePage) {
            return { error: 'No claude tab found', pages };
        }
        const snap = await browser.observe(claudePage.pageId).snapshot();
        return {
            title: claudePage.title,
            url: claudePage.url,
            pageId: claudePage.pageId,
            text: snap.text
        };
        """
        )
        print(json.dumps(res, indent=2))
