"""Real PR fixtures; only labelled fixture-bot identities are mapped in memory."""

import copy
import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, ".ci-shared")
from scripts import check_ai_review_disposition as c

REPO = "liaoxiang0304/VR_DexCopilot"
BOT = "github-actions[bot]"
MARKER = re.compile(r"\[SANDBOX AI FIXTURE: (CODEX|INFISCALE)\]")
ROLE_LOGINS = {"CODEX": c.CODEX, "INFISCALE": "infiscale-bot"}
MANIFEST = Path("docs/ai-review-sandbox/pr-scenarios-20261028.json")
BANNER = (
  "**SANDBOX SIMULATION:** CODEX/INFISCALE roles come from labelled "
  "github-actions fixture Reviews. No real AI completion is claimed. "
  "All PR commits, threads, replies, editors and Issues are read live.\n\n"
)


def logical_actor(role):
  return {
    "login": ROLE_LOGINS[role],
    "__typename": "Bot" if role == "CODEX" else "User",
  }


class ScenarioAPI(c.GitHub):
  def __init__(self, item):
    super().__init__(os.environ["GH_TOKEN"])
    self.item = item
    self.race_armed = False
    self.issue_reads = 0

  def request(self, method, path, body=None):
    if method == "PATCH" and "/check-runs/" in path and (body or {}).get("output"):
      body = copy.deepcopy(body)
      body["output"]["summary"] = BANNER + body["output"]["summary"]
      if body["output"]["title"] in {"Passed", "Failed"}:
        body["output"]["title"] = "SIMULATED " + body["output"]["title"]
    if method == "POST" and path == f"/repos/{REPO}/check-runs":
      body = copy.deepcopy(body)
      body["output"]["summary"] = BANNER + body["output"]["summary"]
    result = super().request(method, path, body)
    issue_path = f"/repos/{REPO}/issues/{self.item.get('issue_number', 0)}"
    if method == "GET" and path == issue_path and self.race_armed:
      self.issue_reads += 1
      if self.issue_reads == 1:
        assert result["state"] == "open"
        self.request("PATCH", issue_path, {"state": "closed"})
    return result


def mapped_snapshot(original, api, repository, number):
  data = original(api, repository, number)
  reviews = api.pages(f"/repos/{repository}/pulls/{number}/reviews")
  roles = {}
  for review in reviews:
    match = MARKER.search(review.get("body") or "")
    if review["user"]["login"] == BOT and match:
      roles[review["node_id"]] = match[1]
  for review in data["reviews"]:
    if c.actor_login(review.get("author")) == BOT and review["id"] in roles:
      review["author"] = logical_actor(roles[review["id"]])
  for comment in [
    *data["comments"],
    *(comment for thread in data["threads"] for comment in thread["comments"]),
  ]:
    match = MARKER.search(comment["body"])
    if c.actor_login(comment.get("author")) == BOT and match:
      comment["author"] = logical_actor(match[1])
  return data


def roots(api, number):
  return [
    comment
    for comment in api.pages(f"/repos/{REPO}/pulls/{number}/comments")
    if not comment.get("in_reply_to_id")
    and comment["user"]["login"] == BOT
    and MARKER.search(comment["body"])
  ]


def seed(api, item, pr):
  number = item["number"]
  published = api.pages(f"/repos/{REPO}/pulls/{number}/reviews")
  existing = {
    MARKER.search(r.get("body") or "")[1]
    for r in published
    if r["user"]["login"] == BOT and MARKER.search(r.get("body") or "")
  }
  roles = ["CODEX"] if item["slug"] == "one-complete" else ["CODEX", "INFISCALE"]
  for role in roles:
    if role in existing:
      continue
    body = (
      f"[SANDBOX AI FIXTURE: {role}]\n\nSIMULATED completed review. "
      "Signed by GitHub Actions; not a real Codex or Infiscale review."
    )
    payload = {"commit_id": pr["head"]["sha"], "event": "COMMENT", "body": body}
    if item["slug"] not in {"clean", "one-complete"}:
      payload["comments"] = [
        {
          "path": item["path"],
          "line": 3 if role == "CODEX" else 5,
          "side": "RIGHT",
          "body": (
            f"[SANDBOX AI FIXTURE: {role}]\n\n"
            f"Synthetic finding for scenario {item['id']}. "
            "Record a disposition according to the PR's scenario."
          ),
        }
      ]
    created = api.request("POST", f"/repos/{REPO}/pulls/{number}/reviews", payload)
    print("Fixture review:", created["html_url"], flush=True)
  if item["slug"] == "bot-reply":
    comments = api.pages(f"/repos/{REPO}/pulls/{number}/comments")
    for root in roots(api, number):
      if not any(
        x.get("in_reply_to_id") == root["id"] and x["user"]["login"] == BOT
        for x in comments
      ):
        api.request(
          "POST",
          f"/repos/{REPO}/pulls/{number}/comments/{root['id']}/replies",
          {
            "body": (
              "Disposition: Rejected\nReason: SANDBOX Bot reply "
              "must not count as an authorized human reply."
            )
          },
        )


def tamper(api, item):
  if item["slug"] != "bot-editor":
    return
  comments = api.pages(f"/repos/{REPO}/pulls/{item['number']}/comments")
  replies = [
    x
    for x in comments
    if x.get("in_reply_to_id") and x["user"]["login"] == "liaoxiang0304"
  ]
  assert len(replies) == 2
  for reply in replies:
    api.request(
      "PATCH",
      f"/repos/{REPO}/pulls/comments/{reply['id']}",
      {
        "body": (
          "Disposition: Rejected\nReason: SANDBOX template edited by the "
          "fixture Bot; editor authorization must fail."
        )
      },
    )
  data = c.snapshot(api, REPO, item["number"])
  edited = [
    x
    for thread in data["threads"]
    for x in thread["comments"]
    if c.actor_login(x.get("author")) == "liaoxiang0304"
  ]
  assert len(edited) == 2 and all(
    c.actor_login(c.content_actor(x)) == BOT for x in edited
  )
  print("Actual editor recorded as github-actions[bot] for both human replies.")


def evaluate(api, item, pr):
  number = item["number"]
  if item["slug"] == "issue-race":
    issue_path = f"/repos/{REPO}/issues/{item['issue_number']}"
    if api.request("GET", issue_path)["state"] != "open":
      api.request("PATCH", issue_path, {"state": "open"})
    api.race_armed = True
  original = c.snapshot
  with patch.object(
    c,
    "snapshot",
    side_effect=lambda api, repository, number: mapped_snapshot(
      original, api, repository, number
    ),
  ):
    data = c.snapshot(api, REPO, number)
    completion = {
      role: c.review_complete(data, login) for role, login in ROLE_LOGINS.items()
    }
    api.issue_reads = 0
    ok = c.check_pr(api, REPO, number, set(ROLE_LOGINS.values()))
  api.race_armed = False
  head = pr["head"]["sha"]
  checks = api.pages(
    f"/repos/{REPO}/commits/{head}/check-runs?check_name=AI%20Review%20%2F%20Disposition%20Check&filter=all",
    "check_runs",
  )
  own = [
    x
    for x in checks
    if (x.get("external_id") or "").startswith(f"ai-review-disposition:{number}:")
  ]
  latest = max(own, key=lambda x: tuple(map(int, x["external_id"].split(":")[-2:])))
  assert (
    latest["external_id"] == f"ai-review-disposition:{number}:"
    f"{os.environ['GITHUB_RUN_ID']}:{os.environ['GITHUB_RUN_ATTEMPT']}"
  ), "A newer writer owns this Check; redispatch after ordinary runs settle"
  assert latest["conclusion"] == item["expected"]
  assert ok == (item["expected"] == "success")
  assert "evaluation failed" not in latest["output"]["summary"]
  assert completion == {"CODEX": True, "INFISCALE": item["slug"] != "one-complete"}
  if item["slug"] == "issue-race":
    assert api.issue_reads == 2
  report = {
    "scenario": item["id"],
    "slug": item["slug"],
    "pr": number,
    "url": item["url"],
    "head": head,
    "expected": item["expected"],
    "actual": latest["conclusion"],
    "check_url": latest["html_url"],
    "completion_roles": completion,
    "simulation": "Only fixture actor identities mapped; all PR evidence read live",
    "issue_validation_reads": api.issue_reads,
    "summary": latest["output"]["summary"],
  }
  Path(f"scenario-{number}.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
  )
  print("SCENARIO RESULT", json.dumps(report), flush=True)


def main():
  assert os.environ["GITHUB_REPOSITORY"] == REPO
  number = int(os.environ["PR_NUMBER"])
  item = next(
    x for x in json.loads(MANIFEST.read_text())["cases"] if x["number"] == number
  )
  api = ScenarioAPI(item)
  pr = api.request("GET", f"/repos/{REPO}/pulls/{number}")
  assert pr["state"] == "open" and pr["draft"] and pr["head"]["ref"] == item["branch"]
  assert (
    pr["head"]["repo"]["full_name"] == REPO
    and pr["base"]["ref"] == "test/ai-review-merge-target"
  )
  operation = os.environ["OPERATION"]
  if operation == "seed":
    seed(api, item, pr)
  elif operation == "tamper":
    tamper(api, item)
  elif operation == "evaluate":
    try:
      evaluate(api, item, pr)
    finally:
      c.cleanup_file(api, REPO, os.environ["AI_REVIEW_CHECKS_FILE"])
  else:
    raise ValueError("Unsupported operation")


if __name__ == "__main__":
  main()
