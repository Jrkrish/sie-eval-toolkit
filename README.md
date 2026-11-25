# SIE eval toolkit

A small Python toolkit for building with [SIE](https://github.com/superlinked/sie), the Superlinked
Inference Engine. It does two things:

1. **Calls SIE safely.** One helper with a disk cache, a usage log and a credit cap.
2. **Measures whether SIE helps.** A harness that runs a simple baseline and SIE on the same
   labelled examples and shows where each one is right and wrong.

It isn't an official Superlinked project.

## Quickstart

You need [uv](https://docs.astral.sh/uv/), which installs Python 3.12 for you.

```bash
uv sync
uv run selfcheck.py            # 21 offline checks: no key, no network, no credits
```

To call SIE, copy `.env.example` to `.env`, add an
[SIE API key](https://console.superlinked.com/console/keys), then:

```bash
uv run probe_models.py         # which models your server lists (no inference calls)
uv run live_probe.py           # one small live call for each of five capabilities
uv run examples/intent_demo/run_demo.py
uv run sie_helper.py report    # what each live call cost
```

## The helper: `sie_helper.py`

```python
from sie_helper import SIE

sie = SIE()
vectors = sie.encode("BAAI/bge-m3", ["text one", "text two"])
ranked = sie.score("Qwen/Qwen3-Reranker-0.6B", "query", ["doc a", "doc b"])
found = sie.extract("urchade/gliner_multi-v2.1", ["Ramesh owes 500"], labels=["person", "amount"])
```

- **Cache.** The key covers the operation, model, options, labels, schema and every input, so changing
  a label can't return an old answer. Errors are never cached. Set `SIE_CACHE=off` to skip it.
- **Usage log.** Each successful live call adds a line to `runs/usage.jsonl` with the model, units,
  credits and latency. Calls that fail or time out aren't logged, though the server may still charge for
  them, and neither are charges reported only in an HTTP header. The credit cap can't see those either,
  so check the console for the final figure.
- **Credit cap.** New live calls stop once logged credits reach `TOOLKIT_CREDIT_CAP` (default
  1,000,000). It's checked before each call, so one large call can go past it.
- **Retries** are left to the SDK.

## The harness: `eval_harness.py`

Write a labels file:

```yaml
task: intent
items:
  - {id: m1, text: "Please send 20 boxes tomorrow.", label: order}
  - {id: m2, text: "I'll pay on Monday.", label: payment}
```

A predictor is a function that takes all the items and returns one dict per item:
`{"label": ..., "score": ...}`, where `score` is optional. Then:

```python
from eval_harness import evaluate, write_report

results = evaluate("labels.yaml", {"baseline": my_baseline, "sie": my_sie_predictor})
write_report(results)   # runs/eval-<time>/report.md and results.json
```

The report has accuracy, precision and recall per class, a threshold sweep for predictors that return
a score, and every item's answers.

`grounded_rate(values, source)` gives the share of extracted values that appear in the source as whole
words. It catches invented values. It doesn't show that a value was put in the right field, and a
correct value written differently from the source counts as invented.

## The demo proves the harness, not SIE

`examples/intent_demo` has 16 made-up business messages in English and Hinglish. The messages, the
labels and most of the keyword baseline were written together, while looking at the same items, so the
scores say nothing reliable about either engine. For a real comparison, write the baseline before you see
the test items, and have someone label them independently.

## Notes

- Models on SIE Serverless change. Run `probe_models.py` rather than trusting a list.
- Only five models have been called live with this toolkit: `BAAI/bge-m3`,
  `Qwen/Qwen3-Reranker-0.6B`, `urchade/gliner_multi-v2.1`, `fastino/gliner2-large-v1` and
  `fastino/GLiNER2.5-Decide`.
- A model that hasn't been used for a while can take several minutes to answer its first call, or
  time out. Warm it up before a live demo.

## Credits

The evaluation approach (hand labels, a baseline beside SIE, a threshold sweep, every item shown) follows
[Cappybara12/mcp-use](https://github.com/Cappybara12/mcp-use). No code was copied from it.

## Licence

[Apache 2.0](LICENSE)
