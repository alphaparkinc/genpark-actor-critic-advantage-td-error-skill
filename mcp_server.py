import sys
import json
from client import TabularActorCritic

ac = TabularActorCritic(["s0", "s1", "s2"], ["act_a", "act_b"])

def handle_rpc(line):
    global ac
    try:
        req = json.loads(line)
    except Exception:
        return
    req_id = req.get("id")
    method = req.get("method")
    params = req.get("params", {})

    if method == "initialize":
        res = {
            "protocolVersion": "2024-11-05",
            "serverInfo": {"name": "genpark-actor-critic-advantage-td-error-skill", "version": "1.0.0"},
            "capabilities": {"tools": {}}
        }
    elif method == "tools/list":
        res = {
            "tools": [
                {
                    "name": "get_policy_probs",
                    "description": "Get current softmax action probabilities for a state",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"state": {"type": "string"}},
                        "required": ["state"]
                    }
                },
                {
                    "name": "update_step",
                    "description": "Perform actor-critic TD-error and policy gradient update",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "state": {"type": "string"},
                            "action": {"type": "string"},
                            "reward": {"type": "number"},
                            "next_state": {"type": "string"}
                        },
                        "required": ["state", "action", "reward", "next_state"]
                    }
                }
            ]
        }
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})
        if tool_name == "get_policy_probs":
            probs = ac.policy_probs(args.get("state", "s0"))
            res = {"content": [{"type": "text", "text": json.dumps({"action_probabilities": dict(zip(ac.actions, probs))})}]}
        elif tool_name == "update_step":
            data = ac.update(
                state=args.get("state"),
                action=args.get("action"),
                reward=args.get("reward"),
                next_state=args.get("next_state")
            )
            res = {"content": [{"type": "text", "text": json.dumps(data)}]}
        else:
            res = {"isError": True, "content": [{"type": "text", "text": f"Unknown tool {tool_name}"}]}
    else:
        res = {"error": {"code": -32601, "message": "Method not found"}}

    resp = {"jsonrpc": "2.0", "id": req_id, "result": res.get("result", res)}
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()

def main():
    for line in sys.stdin:
        if line.strip():
            handle_rpc(line.strip())

if __name__ == "__main__":
    main()
