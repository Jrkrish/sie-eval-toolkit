"""End-to-end harness demo: keyword baseline vs SIE GLiNER2.5-Decide, one batched SIE request.

Run from the toolkit folder:  uv run examples/intent_demo/run_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from eval_harness import evaluate, majority_baseline, write_report  # noqa: E402
from sie_helper import SIE  # noqa: E402

KEYWORDS = {
    "order": ["send", "bhej", "need", "chahiye", "boxes", "peti", "bags"],
    "payment": ["payment", "paisa", "dues", "cheque", "upi", "transfer"],
    "complaint": ["broken", "kharab", "galat", "late", "bad"],
}


def keyword(items):
    out = []
    for it in items:
        t = it["text"].lower()
        hits = {c: sum(k in t for k in ks) for c, ks in KEYWORDS.items()}
        best = max(hits, key=hits.get)
        out.append({"label": best if hits[best] else "other"})
    return out


def sie_decide(items):
    sie = SIE()
    res = sie.extract("fastino/GLiNER2.5-Decide", [it["text"] for it in items], output_schema={
        "intent": {"type": "choice", "instructions": "What is this business message mainly about?",
                   "criteria": {"order": "asks for goods to be sent or supplied",
                                "payment": "about paying money, dues, cheques or transfers",
                                "complaint": "unhappy about goods, delivery or the bill",
                                "other": "greetings, timings, general questions or anything else"}},
    })
    sie.close()
    out = []
    for r in res:
        d = (r.get("data") or {}).get("intent") or {}
        probs = d.get("probabilities") or {}
        out.append({"label": d.get("choice"), "score": max(probs.values()) if probs else None})
    return out


if __name__ == "__main__":
    res = evaluate(HERE / "labels.yaml", {"majority": majority_baseline, "keyword": keyword, "sie_decide": sie_decide})
    out = write_report(res)
    for name, e in res["engines"].items():
        print(f"{name:12} accuracy {e['metrics']['accuracy']:.0%}  ({e['seconds']}s)")
    print(f"\nreport: {out / 'report.md'}")
