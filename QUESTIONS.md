# QUESTIONS — S6E10

Open questions about the project, the data, or the loop. Every blocking
question must be resolved before the round continues. Non-blocking
questions get answered when convenient.

## Open (non-blocking)

(none right now)

## Resolved

- **Q: do we have the original "Airline Passenger Satisfaction" CSV?**
  - A: no, the data dir only has train/test/sample_submission. Queue
    item A is SKIP. Going to B.
- **Q: do we have a GPU?**
  - A: no. CPU-only. CatBoost will run on CPU and be slow (5-10 min
    per fold). Run sparingly.
- **Q: do we have pytabkit / RealMLP?**
  - A: TBD — not installed by default. Will use sklearn-MLP as a
    fallback in queue item E.
- **Q: what's the OOF AUC ceiling we should treat as a leakage alarm?**
  - A: > 0.963. If a round produces that, audit the last change
    before doing anything else.
