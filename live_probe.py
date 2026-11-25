"""One tiny live call per text capability, to learn real output shapes, latency and credit cost.

Each call carries one English and one Hinglish line. One Hinglish sample is an observation,
not a finding. Results are cached, so re-running costs 0 credits.

Run: uv run live_probe.py      then: uv run sie_helper.py report
"""

from __future__ import annotations

import json

from sie_helper import SIE, cosine

EN = "Please send 10 cartons of Parle-G by Friday, payment in 30 days."
HI = "bhaiya 10 peti parle-g bhej do shukravar tak, payment 30 din mein kar denge"


def show(title: str, value) -> None:
    text = json.dumps(value, ensure_ascii=False, default=str)
    print(f"\n=== {title}\n{text[:900]}{' ...' if len(text) > 900 else ''}")


def main() -> None:
    sie = SIE()
    try:
        v = sie.encode("BAAI/bge-m3", [EN, HI, "The weather is sunny today."])
        show("encode bge-m3: cosine(EN,HI) vs cosine(EN,unrelated)",
             {"dims": len(v[0]), "EN~HI": round(cosine(v[0], v[1]), 3), "EN~weather": round(cosine(v[0], v[2]), 3)})

        s = sie.score("Qwen/Qwen3-Reranker-0.6B", "biscuit order with credit terms",
                      [HI, "The weather is sunny today.", EN])
        show("score Qwen3-Reranker-0.6B (input order)", s)

        e = sie.extract("urchade/gliner_multi-v2.1", [EN, HI], labels=["product", "quantity", "date", "payment term"])
        show("extract gliner_multi-v2.1 (labels)", e)

        j = sie.extract("fastino/gliner2-large-v1", [EN, HI], output_schema={
            "type": "object",
            "properties": {"product": {"type": "string"}, "quantity": {"type": "string"},
                           "credit_days": {"type": "string"}},
        })
        show("extract gliner2-large-v1 (schema)", j)

        d = sie.extract("fastino/GLiNER2.5-Decide", [EN, HI], output_schema={
            "intent": {"type": "choice", "instructions": "What does the sender want?",
                       "criteria": {"order": "wants goods sent", "payment": "talks about paying",
                                    "complaint": "unhappy about something"}},
            "asks_credit": {"type": "noul", "instructions": "Does the sender ask to pay later?"},
        })
        show("decide GLiNER2.5-Decide", d)
    finally:
        print(f"\nlive calls: {sie.live_calls}, cache hits: {sie.cache_hits}")
        sie.close()


if __name__ == "__main__":
    main()
