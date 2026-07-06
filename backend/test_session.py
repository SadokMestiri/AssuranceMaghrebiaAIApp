"""
Quick smoke-test for multi-turn session memory.
Run with:  python test_session.py
The backend must be running at http://localhost:8000.
"""
import json
import uuid
import requests

BASE = "http://localhost:8000/agent/query"
SESSION_ID = str(uuid.uuid4())   # fresh session for this run

TURNS = [
    # Turn 1 — normal question with explicit branch
    {"question": "donne-moi les KPI de la branche AUTO", "branch": "AUTO"},
    # Turn 2 — anaphoric follow-up (no branch specified; should inherit context via history)
    {"question": "et pour IRDS ?"},
    # Turn 3 — topic switch to verify history doesn't pollute unrelated questions
    {"question": "quel est le nombre d'impayés total ?"},
]


def ask(turn: dict) -> dict:
    payload = {
        "session_id": SESSION_ID,
        "question": turn["question"],
        "branch": turn.get("branch"),
    }
    r = requests.post(BASE, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


if __name__ == "__main__":
    print(f"Session ID: {SESSION_ID}\n{'='*60}")

    for i, turn in enumerate(TURNS, start=1):
        print(f"\n[Turn {i}] Q: {turn['question']}")
        result = ask(turn)
        agent = result.get("agent", {})
        print(f"  Intent  : {agent.get('intent')} (confidence={agent.get('intent_confidence')})")
        print(f"  Tools   : {agent.get('invoked_tools')}")
        answer = agent.get("answer", "")
        # Print first 300 chars of the answer
        print(f"  Answer  : {answer[:300]}{'...' if len(answer) > 300 else ''}")
