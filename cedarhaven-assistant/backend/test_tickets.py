import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ["TICKET_DB"] = os.path.join(_tmp, "t.db")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import tickets  # noqa: E402

app = FastAPI()
app.include_router(tickets.router)
client = TestClient(app)

TOKEN_A = "a" * 40
TOKEN_B = "b" * 40
GOOD = {
    "title": "Pothole on Maple Street", "description": "Large pothole in the right lane near the school.",
    "category": "Roads & Potholes", "priority": "Medium", "location": "Maple St and 2nd Ave",
}


def H(t):
    return {"X-Client-Token": t}


def test_requires_token():
    assert client.get("/api/tickets").status_code == 401
    assert client.get("/api/tickets", headers=H("short")).status_code == 401


def test_validation_errors():
    r = client.post("/api/tickets", json={"title": "x"}, headers=H(TOKEN_A))
    assert r.status_code == 422
    errs = r.json()["detail"]["errors"]
    assert {"title", "description", "category", "priority", "location"} <= set(errs)


def test_future_date_rejected():
    r = client.post("/api/tickets", json={**GOOD, "date_first_noticed": "2999-01-01"}, headers=H(TOKEN_A))
    assert r.status_code == 422


def test_create_persist_and_retrieve():
    r = client.post("/api/tickets", json=GOOD, headers=H(TOKEN_A))
    assert r.status_code == 200
    t = r.json()
    assert t["id"].startswith("CH-") and t["status"] == "Submitted"
    got = client.get(f"/api/tickets/{t['id']}", headers=H(TOKEN_A))
    assert got.status_code == 200 and got.json()["title"] == GOOD["title"]
    assert t["id"] in [x["id"] for x in client.get("/api/tickets", headers=H(TOKEN_A)).json()["tickets"]]


def test_access_control():
    t = client.post("/api/tickets", json={**GOOD, "location": "Elm Rd and Oak Ave"}, headers=H(TOKEN_A)).json()
    assert client.get(f"/api/tickets/{t['id']}", headers=H(TOKEN_B)).status_code == 404
    assert t["id"] not in [x["id"] for x in client.get("/api/tickets", headers=H(TOKEN_B)).json()["tickets"]]


def test_duplicate_detection():
    body = {**GOOD, "location": "Birch Lane 5"}
    assert client.post("/api/tickets", json=body, headers=H(TOKEN_A)).status_code == 200
    r = client.post("/api/tickets", json=body, headers=H(TOKEN_A))
    assert r.status_code == 409 and "duplicate" in r.json()["detail"]
    ok = client.post("/api/tickets", json={**body, "confirm_duplicate": True}, headers=H(TOKEN_A))
    assert ok.status_code == 200
    # another user is not told about it and can create freely
    assert client.post("/api/tickets", json=body, headers=H(TOKEN_B)).status_code == 200


def test_draft_does_not_invent_location():
    r = client.post("/api/tickets/draft", headers=H(TOKEN_A),
                    json={"messages": [{"role": "user", "content": "My bin was not collected today"}]})
    d = r.json()
    assert r.status_code == 200
    assert d["location"] is None and "location" in d["missing"]
    assert d["category"] == "Waste & Recycling"


def test_draft_urgent_priority_and_notice():
    d = client.post("/api/tickets/draft", headers=H(TOKEN_A),
                    json={"messages": [{"role": "user", "content": "There is a sinkhole on Pine Street"}]}).json()
    assert d["priority"] == "Urgent" and d["emergency_notice"]
    assert d["location"] == "Pine Street"


def test_ids_unique():
    ids = {tickets._new_id() for _ in range(500)}
    assert len(ids) == 500
