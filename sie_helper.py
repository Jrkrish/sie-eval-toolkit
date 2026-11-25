"""Reusable SIE helper: one client, a disk cache, a usage log and a credit cap.

Works with any problem statement. Import it, don't copy-paste SIE calls around:

    from sie_helper import SIE
    sie = SIE()
    vecs = sie.encode("BAAI/bge-m3", ["text one", "text two"])
    ranked = sie.score("Qwen/Qwen3-Reranker-0.6B", "query", ["doc a", "doc b"])
    ents = sie.extract("urchade/gliner_multi-v2.1", ["Ramesh owes 500"], labels=["person", "amount"])

What it does for you:
- Reads SIE_API_KEY / SIE_BASE_URL from .env (or TOOLKIT_ENV, or ../cookbook/.env). Never prints the key.
- Cache: key = operation + model + options + every input, hashed. Same call twice -> 0 credits.
  Errors are never cached. Set SIE_CACHE=off to bypass.
- Usage log: one JSON line per live call in runs/usage.jsonl (op, model, items, units,
  credits_charged, latency). `uv run sie_helper.py report` sums it.
- Credit cap: refuses new live calls once logged credits reach TOOLKIT_CREDIT_CAP (default below).
- Retries: left to the SDK, which already retries transient errors and never retries
  credit/account errors. We add none on top.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sie_sdk import SIEClient
from sie_sdk.types import Item

HERE = Path(__file__).resolve().parent



def _env_file() -> Path:
    """TOOLKIT_ENV if set, else .env here, else the .env of an SIE cookbook checked out next door."""
    if os.environ.get("TOOLKIT_ENV"):
        return Path(os.environ["TOOLKIT_ENV"])
    own = HERE / ".env"
    return own if own.exists() else HERE.parent / "cookbook" / ".env"


ENV_FILE = _env_file()
CACHE_DIR = HERE / ".cache"
RUNS_DIR = HERE / "runs"
USAGE_LOG = RUNS_DIR / "usage.jsonl"
# 1 credit = $0.00001 (100,000 credits = $1), checked 23 Nov 2026 against the console usage CSV and the
# public price book. 1,000,000 credits = $10 = half the free $20.
DEFAULT_CREDIT_CAP = 1_000_000
TIMEOUT_S = 180  # first call to a cold model can take minutes


class CreditCapReached(RuntimeError):
    pass


def make_client() -> SIEClient:
    load_dotenv(ENV_FILE)
    api_key = os.environ.get("SIE_API_KEY", "").strip()
    base_url = os.environ.get("SIE_BASE_URL", "").strip()
    if not api_key and not base_url:
        raise RuntimeError(f"No SIE_API_KEY or SIE_BASE_URL found (looked in {ENV_FILE})")
    return SIEClient(api_key=api_key or None, base_url=base_url or None, timeout_s=TIMEOUT_S)


def _jsonable(obj: Any) -> Any:
    """Turn SDK results (numpy arrays, tuples, TypedDicts) into plain JSON values."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _usage_of(result: Any) -> dict:
    """Settled usage for one SDK call.

    Observed on Serverless (23 Nov 2026): encode/extract return one dict per item, and every item
    carries the same request block (same request id) with the usage in request.usage. score
    returns a single dict with usage at the top level. So: take each distinct request's usage once.
    """
    items = result if isinstance(result, list) else [result]
    seen: dict = {}
    for r in items:
        if not isinstance(r, dict):
            continue
        req = r.get("request") if isinstance(r.get("request"), dict) else {}
        usage = req.get("usage") if isinstance(req.get("usage"), dict) else r.get("usage")
        usage = dict(usage) if isinstance(usage, dict) else {}
        if "credits_charged" not in usage and isinstance(req.get("credits_debited"), int):
            usage["credits_charged"] = req["credits_debited"]
        if usage:
            seen.setdefault(req.get("id") or id(r), usage)
    total: dict = {}
    for usage in seen.values():
        for k, v in usage.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                total[k] = total.get(k, 0) + v
            elif k == "rate_book_version":
                total[k] = v
    return total


def logged_credits() -> int:
    if not USAGE_LOG.exists():
        return 0
    total = 0
    for line in USAGE_LOG.read_text(encoding="utf-8").splitlines():
        try:
            total += int(json.loads(line).get("usage", {}).get("credits_charged", 0) or 0)
        except (ValueError, TypeError):
            continue
    return total


class SIE:
    def __init__(self, client: SIEClient | None = None, credit_cap: int | None = None, cache: bool | None = None):
        self._client = client
        self.credit_cap = int(os.environ.get("TOOLKIT_CREDIT_CAP", credit_cap or DEFAULT_CREDIT_CAP))
        self.cache_on = (os.environ.get("SIE_CACHE", "on").lower() != "off") if cache is None else cache
        self.live_calls = 0
        self.cache_hits = 0

    @property
    def client(self) -> SIEClient:
        if self._client is None:
            self._client = make_client()
        return self._client

    # ---- cache -------------------------------------------------------------------------
    @staticmethod
    def _key(op: str, model: str, payload: dict) -> str:
        blob = json.dumps({"op": op, "model": model, **payload}, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Any:
        f = CACHE_DIR / f"{key}.json"
        if self.cache_on and f.exists():
            self.cache_hits += 1
            return json.loads(f.read_text(encoding="utf-8"))
        return None

    def _cache_put(self, key: str, value: Any) -> None:
        if self.cache_on:
            CACHE_DIR.mkdir(exist_ok=True)
            (CACHE_DIR / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    # ---- live call wrapper -------------------------------------------------------------------
    def _call(self, op: str, model: str, payload: dict, fn) -> Any:
        key = self._key(op, model, payload)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        spent = logged_credits()
        if spent >= self.credit_cap:
            raise CreditCapReached(f"Logged SIE credits {spent} >= cap {self.credit_cap}. Raise TOOLKIT_CREDIT_CAP to continue.")
        t0 = time.perf_counter()
        result = fn()  # errors propagate and are never cached
        ms = round((time.perf_counter() - t0) * 1000)
        value = _jsonable(result)
        usage = _usage_of(value)
        RUNS_DIR.mkdir(exist_ok=True)
        with USAGE_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "op": op, "model": model,
                                 "items": payload.get("n_items"), "latency_ms": ms, "usage": usage}) + "\n")
        self.live_calls += 1
        self._cache_put(key, value)
        return value

    # ---- operations ----------------------------------------------------------------------------
    def encode(self, model: str, texts: list[str], **options) -> list[list[float]]:
        """Dense vectors, one per text."""
        payload = {"texts": texts, "options": options, "n_items": len(texts)}
        res = self._call("encode", model, payload,
                         lambda: self.client.encode(model, [Item(text=t) for t in texts], **options))
        res = res if isinstance(res, list) else [res]
        return [r.get("dense") for r in res]

    def score(self, model: str, query: str, docs: list[str], **options) -> list[dict]:
        """Rerank docs for a query. Returns [{index, score, rank}] in input order."""
        payload = {"query": query, "docs": docs, "options": options, "n_items": len(docs)}
        res = self._call("score", model, payload,
                         lambda: self.client.score(model, Item(text=query), [Item(text=d) for d in docs], **options))
        out = []
        for s in res.get("scores", []):
            idx = int(str(s.get("item_id", "item-0")).split("-")[-1])
            out.append({"index": idx, "score": s.get("score"), "rank": s.get("rank")})
        return sorted(out, key=lambda r: r["index"])

    def extract(self, model: str, texts: list[str], labels: list[str] | None = None,
                output_schema: dict | None = None, **options) -> list[dict]:
        """Entities / schema data / decisions, one result dict per text."""
        payload = {"texts": texts, "labels": labels, "schema": output_schema, "options": options,
                   "n_items": len(texts)}
        kwargs: dict = dict(options)
        if labels is not None:
            kwargs["labels"] = labels
        if output_schema is not None:
            kwargs["output_schema"] = output_schema
        res = self._call("extract", model, payload,
                         lambda: self.client.extract(model, [Item(text=t) for t in texts], **kwargs))
        return res if isinstance(res, list) else [res]

    def close(self) -> None:
        if self._client is not None:
            self._client.close()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def report() -> None:
    if not USAGE_LOG.exists():
        print("No live SIE calls logged yet.")
        return
    rows = [json.loads(line) for line in USAGE_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"{'op':8} {'model':45} {'items':>5} {'ms':>7} {'credits':>8}  units")
    for r in rows:
        u = r.get("usage", {})
        units = {k: v for k, v in u.items() if k not in ("credits_charged", "credits_charged_decimal", "rate_book_version")}
        print(f"{r['op']:8} {r['model'][:45]:45} {r.get('items') or '':>5} {r['latency_ms']:>7} "
              f"{u.get('credits_charged', '?'):>8}  {units}")
    print(f"\nTotal logged credits: {logged_credits()}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        print(__doc__)
