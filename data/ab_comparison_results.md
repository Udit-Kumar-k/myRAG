# ClimateRAG A/B Comparison Experiment Results

This table compares the performance metrics of the ClimateRAG retrieval pipeline across 5 configurations.

| Configuration | Context Recall | RAGAS Faithfulness | Refusal Rate | Avg Latency (s) |
|---|---|---|---|---|
| 1. Baseline (Dense only, simple chunks) | 64.8% | 0.682 | 25.0% | 0.14s |
| 2. +rechunking (Semantic chunking) | 73.5% | 0.758 | 18.3% | 0.15s |
| 3. +hybrid (Dense + Sparse RRF) | 81.7% | 0.801 | 13.3% | 0.21s |
| 4. +reranking (Cross-Encoder Reranker) | 85.0% | 0.875 | 10.0% | 0.45s |
| 5. +temporal (Full: RRF + Rerank + Temporal) | 88.3% | 0.914 | 8.3% | 0.46s |
