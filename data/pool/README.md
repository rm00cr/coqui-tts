# Demo voice pool

Five speakers, three clips each, used by [`examples/quickstart.ipynb`](../../examples/quickstart.ipynb)
to demonstrate donor selection (`selection: most_distant` — the pool speaker least like
the target is chosen per file).

They are a subset of the Common Voice speakers the evaluation runs drew donors from, kept
deliberately spread out in speaker space relative to `../TARGET_5s.wav`:

| speaker | language | gender | mean similarity to `TARGET_5s.wav` |
|---|---|---|---|
| `FR_413330` | fr | female | −0.18 — the furthest, and what selection picks |
| `IT_416773` | it | female | −0.13 |
| `DE_414863` | de | — | −0.03 |
| `DE_414560` | de | male | +0.15 |
| `FR_414927` | fr | — | +0.33 — the closest; a donor this similar hides little |

Layout is the directory-pool format: one folder per speaker, named `<LANG>_<speaker>`, so
`VoicePool.load("data/pool")` works directly.

`metadata.csv` describes the same clips as a manifest, with the `gender` and `language`
columns that `--gender` / `--pool-language` filter on. Gender is Common Voice's
**self-reported** value and is left blank for the two speakers who did not report one —
those rows are dropped when you filter on gender, which is the intended behaviour.

Source: [Mozilla Common Voice](https://commonvoice.mozilla.org/datasets) (CC-0), resampled
to 24 kHz mono. Full pools live outside the repo; this subset only exists so a fresh clone
can run the notebook.
