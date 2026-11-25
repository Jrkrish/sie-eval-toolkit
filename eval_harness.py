"""Reusable baseline-vs-SIE evaluation harness.

The approach (hand labels, a simple baseline next to SIE, a threshold sweep, every item shown)
follows the evaluation in https://github.com/Cappybara12/mcp-use. The code here is written from scratch.

Works for any problem statement: hand labels in a YAML file, any number of predictors,
one report with accuracy, per-class precision/recall, a threshold sweep and every item's answers.

Labels file (YAML):
    task: intent              # free text, shown in the report
    items:
      - {id: m1, text: "...", label: order}
      - {id: m2, text: "...", label: payment}

A predictor is any function   f(items: list[dict]) -> list[dict]
returning, per item, {"label": <predicted label or None>, "score": <float, optional>}.
Batch inside the predictor: cheap SIE calls seem to cost at least ~1 credit per request, so one
request for all items saves credits there (token-heavy models like Decide scale with tokens anyway).

Extraction check (no labels needed): grounded_rate(values, source_text) is the share of
extracted values that literally appear in the source. 1 - grounded_rate = invented values.

Usage from code:
    from eval_harness import evaluate, write_report
    res = evaluate("labels.yaml", {"keyword": kw_fn, "sie": sie_fn})
    write_report(res)            # runs/eval-<timestamp>/report.md + results.json
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import yaml

HERE = Path(__file__).resolve().parent
Predictor = Callable[[list[dict]], list[dict]]


def load_labels(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = data.get("items") or []
    ids = [it["id"] for it in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ids in {path}")
    for it in items:
        if "text" not in it or "label" not in it:
            raise ValueError(f"Item {it.get('id')} needs 'text' and 'label'")
    return {"task": data.get("task", Path(path).stem), "items": items}


def _norm(x: Any) -> Any:
    return x.strip().lower() if isinstance(x, str) else x


def metrics(gold: list, pred: list) -> dict:
    n = len(gold)
    correct = sum(_norm(g) == _norm(p) for g, p in zip(gold, pred))
    classes = sorted({str(_norm(g)) for g in gold} | {str(_norm(p)) for p in pred if p is not None})
    per_class = {}
    for c in classes:
        tp = sum(str(_norm(g)) == c and str(_norm(p)) == c for g, p in zip(gold, pred))
        fp = sum(str(_norm(g)) != c and str(_norm(p)) == c for g, p in zip(gold, pred))
        fn = sum(str(_norm(g)) == c and str(_norm(p)) != c for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        per_class[c] = {"precision": prec, "recall": rec, "support": tp + fn}
    return {"n": n, "accuracy": correct / n if n else None, "abstained": sum(p is None for p in pred),
            "per_class": per_class}


def threshold_sweep(gold: list, preds: list[dict], steps: int = 11) -> list[dict] | None:
    """For predictors with a score: abstain (label None) below a threshold; see the trade-off."""
    scores = [p.get("score") for p in preds]
    if any(not isinstance(s, (int, float)) for s in scores):
        return None
    lo, hi = min(scores), max(scores)
    rows = []
    for i in range(steps):
        t = lo + (hi - lo) * i / (steps - 1) if steps > 1 else lo
        kept = [(g, p["label"]) for g, p in zip(gold, preds) if p["score"] >= t]
        acc = sum(_norm(g) == _norm(p) for g, p in kept) / len(kept) if kept else None
        rows.append({"threshold": round(t, 4), "coverage": len(kept) / len(gold), "accuracy_on_kept": acc})
    return rows


def evaluate(labels_path: str | Path, predictors: dict[str, Predictor]) -> dict:
    data = load_labels(labels_path)
    items, gold = data["items"], [it["label"] for it in data["items"]]
    out: dict = {"task": data["task"], "labels_file": str(labels_path), "n": len(items), "engines": {}}
    for name, fn in predictors.items():
        t0 = time.perf_counter()
        preds = fn(items)
        if len(preds) != len(items):
            raise ValueError(f"Predictor {name} returned {len(preds)} results for {len(items)} items")
        out["engines"][name] = {
            "seconds": round(time.perf_counter() - t0, 2),
            "metrics": metrics(gold, [p.get("label") for p in preds]),
            "sweep": threshold_sweep(gold, preds),
            "predictions": preds,
        }
    out["items"] = [{"id": it["id"], "text": it["text"], "gold": it["label"],
                     **{name: out["engines"][name]["predictions"][i].get("label") for name in predictors}}
                    for i, it in enumerate(items)]
    return out


def _fmt(x: Any) -> str:
    return "-" if x is None else (f"{x:.0%}" if isinstance(x, float) else str(x))


def write_report(res: dict, out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else HERE / "runs" / f"eval-{time.strftime('%Y%m%dT%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    names = list(res["engines"])
    lines = [f"# Eval: {res['task']}", "", f"{res['n']} labelled items from `{res['labels_file']}`.", "",
             "| Engine | Accuracy | Abstained | Seconds |", "|---|---|---|---|"]
    for n in names:
        m = res["engines"][n]["metrics"]
        lines.append(f"| {n} | {_fmt(m['accuracy'])} | {m['abstained']} | {res['engines'][n]['seconds']} |")
    lines += ["", "## Per class (precision / recall)", "", "| Class | " + " | ".join(names) + " |",
              "|---|" + "---|" * len(names)]
    classes = sorted({c for n in names for c in res["engines"][n]["metrics"]["per_class"]})
    for c in classes:
        cells = []
        for n in names:
            pc = res["engines"][n]["metrics"]["per_class"].get(c, {})
            cells.append(f"{_fmt(pc.get('precision'))} / {_fmt(pc.get('recall'))}")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    for n in names:
        sw = res["engines"][n]["sweep"]
        if sw:
            lines += ["", f"## Threshold sweep: {n}", "", "| Threshold | Coverage | Accuracy on kept |", "|---|---|---|"]
            lines += [f"| {r['threshold']} | {_fmt(r['coverage'])} | {_fmt(r['accuracy_on_kept'])} |" for r in sw]
    lines += ["", "## Every item", "", "| id | gold | " + " | ".join(names) + " | text |", "|---|---|" + "---|" * len(names) + "---|"]
    for it in res["items"]:
        marks = [f"{_fmt(it[n])}{'' if _norm(it[n]) == _norm(it['gold']) else ' ✗'}" for n in names]
        text = it["text"].replace("|", "/")[:70]
        lines.append(f"| {it['id']} | {it['gold']} | " + " | ".join(marks) + f" | {text} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ---- extraction grounding (no labels needed) ------------------------------------------------
def _canon(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"(?<=\d),(?=\d)", "", s)  # 1,000 and 1000 are the same number
    s = re.sub(r"(?<=\d)\.(?=\d)", "\x00", s)  # keep decimal points: 4.2 is not "4 2"
    s = re.sub(r"[^\w\x00ऀ-ॿ]+", " ", s.replace("_", " "))
    return s.replace("\x00", ".").strip()


def grounded_rate(values: list[str], source: str) -> dict:
    """Share of extracted values that literally occur in the source as whole words.

    Whole words, so "5" is not found inside "15" and "10" is not found inside "100".
    """
    vals = [v for v in values if _canon(v)]
    src = f" {_canon(source)} "
    hits = [v for v in vals if f" {_canon(v)} " in src]
    return {"n": len(vals), "grounded": len(hits), "rate": len(hits) / len(vals) if vals else None,
            "invented": [v for v in vals if v not in hits]}


def majority_baseline(items: list[dict]) -> list[dict]:
    """Always predicts the most common gold label. Any engine must beat this to mean anything."""
    top = Counter(it["label"] for it in items).most_common(1)[0][0]
    return [{"label": top} for _ in items]
