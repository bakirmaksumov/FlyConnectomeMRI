# Test fixtures

## tiny_graph.csv

A hand-made 6-node toy graph for testing the loader. **It is not biological data.**
The node IDs are arbitrary.

```
1001 --17--> 1050 --22--> 8120 --3--> 1001      (a cycle)
1001 --5---> 3002 --8---> 8120
5000 --2---> 5001                                (separate piece)
```

Hand-computed statistics (asserted in `tests/test_statistics.py`):

| Statistic | Value | Why |
|---|---|---|
| nodes | 6 | 1001, 1050, 3002, 5000, 5001, 8120 |
| edges | 6 | 6 data rows |
| density | 0.2 | 6 / (6 × 5) |
| mean in/out-degree | 1.0 | 6 edges / 6 nodes |
| max in-degree | 2 | 8120 (from 1050 and 3002) |
| max out-degree | 2 | 1001 (to 1050 and 3002) |
| max degree (in+out) | 3 | 1001 and 8120 |
| weakly connected components | 2 | {1001, 1050, 3002, 8120} and {5000, 5001} |
| strongly connected components | 3 | {1001, 1050, 3002, 8120} (all on cycles), {5000}, {5001} |
| total / mean / min / max weight | 57 / 9.5 / 2 / 22 | |
