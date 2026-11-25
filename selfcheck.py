"""Offline checks for sie_helper and eval_harness. Uses a fake SIE client: 0 credits, no network.

Run: uv run selfcheck.py      (exits non-zero if a check fails)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import sie_helper
from eval_harness import evaluate, grounded_rate, majority_baseline, metrics, threshold_sweep, write_report


class FakeClient:
    """Mimics the result shapes observed on Serverless on 23 Nov 2026."""

    def __init__(self, credits: int = 1, fail: bool = False):
        self.calls = 0
        self.credits = credits
        self.fail = fail

    def _req(self):
        return {"id": f"req-{self.calls}", "usage": {"input_tokens": 10, "credits_charged": self.credits}}

    def encode(self, model, items, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        req = self._req()
        return [{"dense": [1.0, float(i)], "model": model, "request": req} for i, _ in enumerate(items)]

    def score(self, model, query, items, **kw):
        self.calls += 1
        return {"model": model, "scores": [{"item_id": f"item-{i}", "score": 1.0 / (i + 1), "rank": i}
                                           for i, _ in enumerate(items)][::-1],
                "usage": {"input_tokens": 5, "pairs": len(items), "credits_charged": self.credits}}

    def extract(self, model, items, labels=None, output_schema=None, **kw):
        self.calls += 1
        req = self._req()
        return [{"entities": [{"text": "x", "label": (labels or ["y"])[0]}], "request": req} for _ in items]

    def close(self):
        pass


failures: list[str] = []


def check(name: str, cond: bool) -> None:
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    sie_helper.CACHE_DIR, sie_helper.RUNS_DIR = tmp / ".cache", tmp / "runs"
    sie_helper.USAGE_LOG = sie_helper.RUNS_DIR / "usage.jsonl"
    os.environ.pop("TOOLKIT_CREDIT_CAP", None)
    try:
        fake = FakeClient()
        sie = sie_helper.SIE(client=fake, credit_cap=100, cache=True)

        v = sie.encode("m", ["a", "b"])
        check("encode returns one vector per text", len(v) == 2 and v[1] == [1.0, 1.0])
        sie.encode("m", ["a", "b"])
        check("same call twice hits the cache (1 live call)", fake.calls == 1 and sie.cache_hits == 1)
        check("usage of a shared request block counted once", sie_helper.logged_credits() == 1)

        sie.extract("m", ["t"], labels=["person"])
        sie.extract("m", ["t"], labels=["amount"])
        check("different labels -> different cache entries", fake.calls == 3)
        sie.extract("m", ["t"], output_schema={"a": 1})
        sie.extract("m", ["t"], output_schema={"a": 2})
        check("different schema -> different cache entries", fake.calls == 5)

        s = sie.score("m", "q", ["d0", "d1", "d2"])
        check("score results come back in input order", [r["index"] for r in s] == [0, 1, 2])
        check("top-level score usage is logged", sie_helper.logged_credits() == 6)

        failing = FakeClient(fail=True)
        bad = sie_helper.SIE(client=failing, credit_cap=100, cache=True)
        before = len(list(sie_helper.CACHE_DIR.glob("*.json")))
        raised = 0
        for _ in range(2):
            try:
                bad.encode("m", ["zzz"])
            except RuntimeError:
                raised += 1
        check("errors propagate", raised == 2)
        check("errors are not cached (no file written, second try calls again)",
              len(list(sie_helper.CACHE_DIR.glob("*.json"))) == before and failing.calls == 2)

        capped = sie_helper.SIE(client=FakeClient(), credit_cap=6, cache=True)
        try:
            capped.encode("m", ["new text"])
            check("credit cap blocks new live calls", False)
        except sie_helper.CreditCapReached:
            check("credit cap blocks new live calls", True)
        check("credit cap still serves cached calls", capped.encode("m", ["a", "b"])[0] == [1.0, 0.0])

        m = metrics(["a", "a", "b", "b"], ["a", "b", "b", None])
        check("accuracy", m["accuracy"] == 0.5 and m["abstained"] == 1)
        check("precision and recall differ where they should",
              m["per_class"]["a"]["precision"] == 1.0 and m["per_class"]["a"]["recall"] == 0.5
              and m["per_class"]["b"]["precision"] == 0.5)
        sw = threshold_sweep(["a", "b"], [{"label": "a", "score": 0.9}, {"label": "a", "score": 0.1}], steps=2)
        check("sweep: high threshold keeps only the confident, correct one", sw[-1]["coverage"] == 0.5 and sw[-1]["accuracy_on_kept"] == 1.0)
        check("sweep is skipped when there are no scores", threshold_sweep(["a"], [{"label": "a"}]) is None)

        g = grounded_rate(["10 peti", "Parle-G", "45 days"], "bhaiya 10 peti parle g bhej do")
        check("grounded_rate finds literal values, flags invented ones", g["grounded"] == 2 and g["invented"] == ["45 days"])

        g = grounded_rate(["5", "10", "1000", "4.2", "दस पेटी"], "15 boxes, 100 bags, Rs 1,000, rate 4.25, दस पेटी")
        check("grounded_rate matches whole words only", g["invented"] == ["5", "10", "4.2"])
        g = grounded_rate(["4 2"], "rate 4.2")
        check("grounded_rate keeps decimal points", g["grounded"] == 0)

        no_body_usage = [{"dense": [0.0], "request": {"id": "r", "credits_debited": 4}}]
        check("credits fall back to credits_debited", sie_helper._usage_of(no_body_usage).get("credits_charged") == 4)

        labels = tmp / "labels.yaml"
        labels.write_text("task: t\nitems:\n  - {id: a, text: 'pay later', label: payment}\n"
                          "  - {id: b, text: 'send goods', label: order}\n  - {id: c, text: 'send', label: order}\n",
                          encoding="utf-8")
        res = evaluate(labels, {"majority": majority_baseline,
                                "keyword": lambda its: [{"label": "payment" if "pay" in i["text"] else "order", "score": 1.0} for i in its]})
        check("evaluate: keyword beats majority on toy set",
              res["engines"]["keyword"]["metrics"]["accuracy"] == 1.0 and res["engines"]["majority"]["metrics"]["accuracy"] < 1.0)
        rep = write_report(res, tmp / "report")
        check("write_report writes report.md and results.json",
              (rep / "report.md").exists() and json.loads((rep / "results.json").read_text(encoding="utf-8"))["n"] == 3)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'all checks passed' if not failures else f'{len(failures)} check(s) failed'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
