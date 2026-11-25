"""Free check: which candidate models does SIE Serverless serve? (list_models costs no credits.)

Run: uv run probe_models.py
Writes runs/served_models.json and prints a served / not-served table for the candidates.
"""

from __future__ import annotations

import json
from pathlib import Path

from sie_helper import make_client

CANDIDATES = {
    "embed": ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-m3", "intfloat/multilingual-e5-large",
              "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B"],
    "rerank": ["cross-encoder/ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-v2-m3",
               "jinaai/jina-reranker-v2-base-multilingual", "Qwen/Qwen3-Reranker-0.6B", "Qwen/Qwen3-Reranker-4B"],
    "extract": ["urchade/gliner_multi-v2.1", "fastino/gliner2-large-v1", "knowledgator/gliformer-base-v1",
                "numind/NuNER_Zero"],
    "decide/classify": ["fastino/GLiNER2.5-Decide", "fastino/GLiNER2.5-multi-Decide", "convaiinnovations/laya",
                        "convaiinnovations/laya-multilingual", "knowledgator/gliclass-multilang-mini"],
    "guard/pii": ["Qwen/Qwen3Guard-Gen-4B", "Qwen/Qwen3Guard-Gen-0.6B", "superlinked/pii-redact",
                  "knowledgator/opir-multitask-multilang-v1.0"],
    "ocr/docs": ["docling", "lightonai/LightOnOCR-3-4B", "zai-org/GLM-OCR"],
    "speech": ["openai/whisper-large-v3-turbo", "nvidia/parakeet-tdt-0.6b-v3"],
    "translate": ["google/madlad400-3b-mt"],
    "generate": ["Qwen/Qwen3-0.6B", "Qwen/Qwen3.8-27B-FP8"],
}


def main() -> None:
    client = make_client()
    models = client.list_models()
    names = set()
    for m in models:
        name = m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
        if name:
            names.add(name)
    out = Path("runs")
    out.mkdir(exist_ok=True)
    (out / "served_models.json").write_text(json.dumps(sorted(names), indent=2), encoding="utf-8")
    print(f"Serverless lists {len(names)} models (full list: runs/served_models.json)\n")
    lower = {n.lower(): n for n in names}
    for cap, cands in CANDIDATES.items():
        print(f"[{cap}]")
        for c in cands:
            hit = lower.get(c.lower()) or next((n for n in names if n.lower().endswith(c.lower().split("/")[-1])), None)
            print(f"  {'SERVED ' if hit else 'missing'}  {c}" + (f"  (as {hit})" if hit and hit != c else ""))
    client.close()


if __name__ == "__main__":
    main()
