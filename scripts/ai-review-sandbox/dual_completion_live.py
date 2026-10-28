"""Sandbox test: simulated review inputs, real Checks/Issues API, no PR writes."""

import contextlib
import copy
import io
import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, ".ci-shared")
from scripts import check_ai_review_disposition as c
from scripts.tests import test_ai_review_dual_completion as matrix

REPO = os.environ["GITHUB_REPOSITORY"]
NUMBER = 6
HEAD = os.environ["GITHUB_SHA"]
LAB_NAME = "AI Review / Dual Completion Simulation (sandbox only)"
c.CHECK_NAME = LAB_NAME


class SandboxAPI(c.GitHub):
  def __init__(self):
    super().__init__(os.environ["GH_TOKEN"])
    self.check_ids = set()
    self.issue_number = None
    self.close_during_read = False
    self.tracking_reads = 0

  def request(self, method, path, body=None):
    prefix = f"/repos/{REPO}"
    if method in {"POST", "PATCH"}:
      if path == prefix + "/check-runs":
        assert body["name"] == LAB_NAME and body["head_sha"] == HEAD
      elif path.startswith(prefix + "/check-runs/"):
        assert int(path.rsplit("/", 1)[1]) in self.check_ids
        assert body.get("name", LAB_NAME) == LAB_NAME
      elif path == prefix + "/issues":
        assert body["title"].startswith("[Sandbox disposable] Dual-completion")
      else:
        assert self.issue_number and path == prefix + f"/issues/{self.issue_number}"
        assert method == "PATCH" and set(body) == {"state"}
    result = super().request(method, path, body)
    if method == "GET" and "/commits/" in path and "/check-runs?" in path:
      self.check_ids.update(
        item["id"]
        for item in result["check_runs"]
        if item["name"] == LAB_NAME and item["head_sha"] == HEAD
      )
    if method == "POST" and path == prefix + "/check-runs":
      self.check_ids.add(result["id"])
    if method == "GET" and path == prefix + f"/pulls/{NUMBER}":
      # Simulated head on the default branch keeps every experimental Check
      # off the real PR's head and outside its required business gate.
      result = copy.deepcopy(result)
      result["head"]["sha"] = HEAD
    if (
      method == "GET"
      and self.issue_number
      and path == prefix + f"/issues/{self.issue_number}"
    ):
      self.tracking_reads += 1
      if self.close_during_read and self.tracking_reads == 1:
        assert result["state"] == "open"
        self.request("PATCH", path, {"state": "closed"})
    return result


def main():
  assert REPO == "liaoxiang0304/VR_DexCopilot"
  assert set(os.environ["CONFIGURED_BOTS"].split(",")) == matrix.BOTS
  api = SandboxAPI()
  real_pr = c.GitHub.request(api, "GET", f"/repos/{REPO}/pulls/{NUMBER}")
  assert (
    real_pr["state"] == "open"
    and real_pr["head"]["ref"] == "test/ai-review-e2e-shared-reviewed"
  )
  real_head = real_pr["head"]["sha"]
  assert HEAD != real_head
  author = real_pr["user"]["login"]
  real_checks_path = f"/repos/{REPO}/commits/{real_head}/check-runs?filter=all"
  before = api.pages(real_checks_path, "check_runs")
  summary_path = os.environ.pop("GITHUB_STEP_SUMMARY")
  outcomes = []
  # These require a second actual collaborator/editor account for live proof.
  simulated_permissions_only = {
    "maintainer",
    "read_only_user",
    "maintainer_permission_revoked",
    "human_editor",
    "unauthorized_editor",
  }

  def remap(data):
    encoded = json.dumps(data)
    for old, new in (
      ("org/repo", REPO),
      (matrix.f.HEAD, HEAD),
      (matrix.f.FIX, real_head),
      ("contributor", author),
      ("/issues/42", f"/issues/{api.issue_number}"),
    ):
      encoded = encoded.replace(old, new)
    return json.loads(encoded)

  def execute(name, data, expected, complete, blocked=0):
    assert (
      tuple(c.review_complete(data, bot) for bot in (matrix.f.CODEX, "infiscale-bot"))
      == complete
    )
    with (
      patch.object(c, "snapshot", side_effect=lambda *args: copy.deepcopy(data)),
      contextlib.redirect_stdout(io.StringIO()),
    ):
      actual = c.check_pr(api, REPO, NUMBER, matrix.BOTS)
    runs = api.pages(
      f"/repos/{REPO}/commits/{HEAD}/check-runs?check_name=AI%20Review%20%2F%20Dual%20Completion%20Simulation%20%28sandbox%20only%29&filter=all",
      "check_runs",
    )
    assert len(runs) == 1
    check = runs[0]
    assert check["name"] == LAB_NAME and check["head_sha"] == HEAD
    expected_conclusion = "success" if expected else "failure"
    assert actual == expected and check["conclusion"] == expected_conclusion, (
      name,
      actual,
      check["conclusion"],
    )
    assert "evaluation failed" not in check["output"]["summary"], name
    completion_errors = check["output"]["summary"].count(
      "waiting for this review round"
    )
    assert completion_errors == sum(not item for item in complete), name
    counts = re.search(
      r"Findings: ([0-9]+); blocked dispositions: ([0-9]+)", check["output"]["summary"]
    )
    assert counts and int(counts[2]) == blocked, name
    outcomes.append({
      "scenario": name,
      "both_complete": all(complete),
      "expected": expected_conclusion,
      "actual": check["conclusion"],
      "check_id": check["id"],
      "findings": int(counts[1]),
      "blocked": int(counts[2]),
    })
    print("SIMULATED INPUT:", name, "=>", check["conclusion"], flush=True)

  try:
    issue = api.request(
      "POST",
      f"/repos/{REPO}/issues",
      {
        "title": f"[Sandbox disposable] Dual-completion {os.environ['GITHUB_RUN_ID']}",
        "body": (
          "Automated conditional-input test only. AI completions are simulated. "
          "No real AI Review is claimed. This issue is closed by cleanup."
        ),
      },
    )
    api.issue_number = issue["number"]
    for case in matrix.scenarios():
      if case["name"] in simulated_permissions_only:
        continue
      if case["name"] == "deferred_closed_issue":
        api.request(
          "PATCH", f"/repos/{REPO}/issues/{api.issue_number}", {"state": "closed"}
        )
      execute(
        case["name"],
        remap(case["data"]),
        case["expected"],
        case["complete"],
        case["blocked"],
      )
    # Real Issue state changes between the two permission/Tracking validations.
    api.request("PATCH", f"/repos/{REPO}/issues/{api.issue_number}", {"state": "open"})
    api.tracking_reads = 0
    api.close_during_read = True
    data = remap(matrix.baseline((matrix.DISPOSITIONS["deferred"], matrix.REJECTED)))
    execute("issue_closed_between_validation_reads", data, False, (True, True), 1)
    assert api.tracking_reads == 2
    api.close_during_read = False
    after = api.pages(real_checks_path, "check_runs")
    assert {item["id"]: item for item in after} == {item["id"]: item for item in before}
    assert len(api.check_ids) == 1
    report = {
      "input_level": "Both AI completions and review/comment snapshots are simulated",
      "real_api": [
        "Checks create/update/read",
        "Tracking Issue open/close/read",
        "real PR Checks unchanged",
      ],
      "not_live_verified": sorted(simulated_permissions_only),
      "shared_ref": os.environ["SHARED_REF"],
      "production_pr_head": real_head,
      "simulation_head": HEAD,
      "production_checks_before": len(before),
      "production_checks_after": len(after),
      "tracking_issue": issue["html_url"],
      "scenario_count": len(outcomes),
      "outcomes": outcomes,
    }
    Path("dual-completion-live-results.json").write_text(
      json.dumps(report, indent=2), encoding="utf-8"
    )
    with open(summary_path, "w", encoding="utf-8") as stream:
      stream.write(
        "# Conditional AI completion test\n\n"
        "Both AI completions are SIMULATED. No real Reviews were submitted.\n\n"
      )
      stream.write(
        f"{len(outcomes)} scenarios passed. "
        f"Production PR Checks unchanged ({len(before)}).\n\n"
      )
      stream.write("| Scenario | Expected | Actual |\n| --- | --- | --- |\n")
      for item in outcomes:
        stream.write(
          f"| {item['scenario']} | {item['expected']} | {item['actual']} |\n"
        )
    api.request(
      "PATCH",
      f"/repos/{REPO}/check-runs/{next(iter(api.check_ids))}",
      {
        "status": "completed",
        "conclusion": "success",
        "completed_at": c.datetime.now(c.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output": {
          "title": "SIMULATED scenario assertions passed",
          "summary": (
            f"{len(outcomes)} conditional-input scenarios passed. "
            "Both AI completions were simulated. No production PR Checks changed."
          ),
        },
      },
    )
    print(
      "LIVE RESULT",
      json.dumps({key: value for key, value in report.items() if key != "outcomes"}),
      flush=True,
    )
  except Exception:
    for check_id in api.check_ids:
      api.request(
        "PATCH",
        f"/repos/{REPO}/check-runs/{check_id}",
        {
          "status": "completed",
          "conclusion": "failure",
          "output": {
            "title": "Simulation assertion failed",
            "summary": (
              "Conditional-input sandbox test failed; inspect the job log. "
              "This is not a production PR result."
            ),
          },
        },
      )
    raise
  finally:
    if api.issue_number:
      api.request(
        "PATCH", f"/repos/{REPO}/issues/{api.issue_number}", {"state": "closed"}
      )


if __name__ == "__main__":
  main()
