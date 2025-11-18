# Vyakti Search Evaluation Guide

This directory contains tools and datasets for evaluating and optimizing Vyakti search quality.

## Overview

The evaluation framework provides:
- **Metrics**: Precision@K, Recall@K, F1@K, MAP, MRR, NDCG@K
- **Datasets**: JSON format for test queries with ground truth relevance
- **Scripts**: Tools for running evaluations and parameter optimization
- **Visualization**: Performance comparison across configurations

## Quick Start

### 1. Create an Evaluation Dataset

Create a JSON file with test queries and relevant documents:

```json
{
  "name": "my_test_dataset",
  "description": "Test queries for my domain",
  "queries": [
    {
      "query": "machine learning basics",
      "relevant_docs": ["1", "5", "7"],
      "graded_relevance": {
        "1": 3,
        "5": 2,
        "7": 1
      }
    }
  ]
}
```

**Relevance Scores:**
- `0`: Not relevant
- `1`: Somewhat relevant
- `2`: Relevant
- `3`: Highly relevant

### 2. Run Evaluation

```bash
# Run evaluation script
./evaluation/scripts/evaluate.sh \
    --index my-index \
    --dataset ./evaluation/datasets/my_test.json \
    --k-values 1,3,5,10,20 \
    --output ./evaluation/results/

# Output shows:
# - Precision@K, Recall@K, F1@K for each K
# - MAP (Mean Average Precision)
# - MRR (Mean Reciprocal Rank)
# - NDCG@K (Normalized Discounted Cumulative Gain)
```

### 3. Optimize Parameters

```bash
# Grid search over parameters
./evaluation/scripts/optimize.sh \
    --index my-index \
    --dataset ./evaluation/datasets/my_test.json \
    --optimize-for ndcg@10 \
    --output ./evaluation/optimization/

# Tests combinations of:
# - graph_degree: [16, 32, 64]
# - search_complexity: [16, 32, 64, 128]
# - chunk_size: [128, 256, 512]
```

## Evaluation Metrics

### Precision@K
Fraction of top-K results that are relevant.
- **Range**: 0.0 to 1.0
- **Higher is better**
- **Use case**: When you want most results to be relevant

```
Precision@K = (Relevant docs in top-K) / K
```

### Recall@K
Fraction of all relevant documents found in top-K.
- **Range**: 0.0 to 1.0
- **Higher is better**
- **Use case**: When you want to find all relevant documents

```
Recall@K = (Relevant docs in top-K) / (Total relevant docs)
```

### F1@K
Harmonic mean of Precision and Recall.
- **Range**: 0.0 to 1.0
- **Higher is better**
- **Use case**: Balance between precision and recall

```
F1@K = 2 * (Precision * Recall) / (Precision + Recall)
```

### Mean Average Precision (MAP)
Average of precision values at each relevant document position.
- **Range**: 0.0 to 1.0
- **Higher is better**
- **Use case**: Overall ranking quality across all positions

### Mean Reciprocal Rank (MRR)
Average of reciprocal ranks of first relevant document.
- **Range**: 0.0 to 1.0
- **Higher is better**
- **Use case**: How quickly users find first relevant result

```
MRR = Average(1 / rank_of_first_relevant_doc)
```

### Normalized Discounted Cumulative Gain (NDCG@K)
Measures ranking quality with graded relevance.
- **Range**: 0.0 to 1.0
- **Higher is better**
- **Use case**: When relevance is graded (not just binary)

## Creating Evaluation Datasets

### Format

```json
{
  "name": "dataset_name",
  "description": "Dataset description",
  "queries": [
    {
      "query": "search query text",
      "relevant_docs": ["doc_id_1", "doc_id_2"],
      "graded_relevance": {
        "doc_id_1": 3,
        "doc_id_2": 2
      }
    }
  ]
}
```

### Document ID Mapping

Document IDs in the evaluation dataset should match the IDs in your Vyakti index. You can find document IDs by:

1. **Search with verbose mode**:
   ```bash
   vyakti search my-index "query" -v
   ```

2. **Programmatically get IDs**:
   ```rust
   let results = searcher.search("query", 100).await?;
   for result in results {
       println!("ID: {}, Text: {}", result.id, result.text);
   }
   ```

### Best Practices

1. **Diverse Queries**: Include queries of varying difficulty
2. **Balanced Relevance**: Mix of highly/moderately/weakly relevant docs
3. **Sufficient Coverage**: At least 20-50 queries for reliable metrics
4. **Representative**: Queries should match real user needs
5. **Multiple Annotators**: Have 2-3 people judge relevance for consistency

## Parameter Optimization

### Key Parameters to Optimize

#### 1. Graph Degree
Controls connectivity in HNSW graph.

```bash
# Test: 16, 32, 64
--graph-degree <value>
```

- **Lower** (16): Faster search, less storage, lower recall
- **Higher** (64): Slower search, more storage, higher recall
- **Recommended**: Start with 32

#### 2. Search Complexity
Controls search thoroughness.

```bash
# Test: 16, 32, 64, 128
--search-complexity <value>
```

- **Lower** (16): Faster, lower recall
- **Higher** (128): Slower, higher recall
- **Recommended**: Match to graph_degree or 2x

#### 3. Chunk Size
Text chunking window size.

```bash
# Test: 128, 256, 512
--chunk-size <value>
```

- **Smaller** (128): More precise matches, more chunks
- **Larger** (512): More context, fewer chunks
- **Recommended**: 256 for general use

#### 4. Chunk Overlap
Overlap between consecutive chunks.

```bash
# Test: 64, 128, 256
--chunk-overlap <value>
```

- **Rule of thumb**: 50% of chunk_size
- **Recommended**: 128 for chunk_size=256

#### 5. Embedding Model
Different models have different quality/speed trade-offs.

```bash
# Test different models
--embedding-model mxbai-embed-large  # 1024d, best quality
--embedding-model nomic-embed-text    # 768d, balanced
--embedding-model all-minilm          # 384d, fastest
```

## Example Workflows

### Workflow 1: Quick Evaluation

```bash
# 1. Build test index
vyakti build test-index --input ./test_docs --compact

# 2. Create small test dataset (5-10 queries)
cat > test_dataset.json << 'EOF'
{
  "name": "quick_test",
  "description": "Quick sanity check",
  "queries": [
    {"query": "test query 1", "relevant_docs": ["1", "2"]},
    {"query": "test query 2", "relevant_docs": ["3"]}
  ]
}
EOF

# 3. Run evaluation
./evaluation/scripts/evaluate.sh \
    --index test-index \
    --dataset test_dataset.json
```

### Workflow 2: Full Optimization

```bash
# 1. Create comprehensive dataset (50+ queries)
# ... build dataset with multiple annotators ...

# 2. Baseline evaluation
./evaluation/scripts/evaluate.sh \
    --index baseline-index \
    --dataset full_dataset.json \
    --output baseline_results/

# 3. Grid search optimization
./evaluation/scripts/optimize.sh \
    --dataset full_dataset.json \
    --optimize-for ndcg@10 \
    --graph-degree 16,32,64 \
    --search-complexity 16,32,64,128 \
    --chunk-size 128,256,512 \
    --output optimization_results/

# 4. Build production index with best params
vyakti build prod-index \
    --input ./docs \
    --graph-degree 32 \
    --chunk-size 256 \
    --compact

# 5. Verify improvement
./evaluation/scripts/evaluate.sh \
    --index prod-index \
    --dataset full_dataset.json \
    --output prod_results/
```

### Workflow 3: A/B Testing

```bash
# Test two configurations
./evaluation/scripts/compare.sh \
    --index-a baseline-index \
    --index-b optimized-index \
    --dataset test_dataset.json \
    --output comparison_results/

# Shows side-by-side metrics
```

## Interpreting Results

### Good Metrics

- **Precision@10 > 0.7**: Most results are relevant
- **Recall@20 > 0.8**: Finding most relevant docs
- **MAP > 0.6**: Good overall ranking quality
- **MRR > 0.8**: First result usually relevant
- **NDCG@10 > 0.7**: Good ranking with graded relevance

### Red Flags

- **Precision@10 < 0.3**: Too many irrelevant results
- **Recall@20 < 0.4**: Missing too many relevant docs
- **MAP < 0.3**: Poor ranking quality
- **MRR < 0.5**: First result often not relevant

### What to Optimize

| Problem | Solution |
|---------|----------|
| Low Precision | Increase search_complexity, better chunking |
| Low Recall | Increase graph_degree, increase top-K |
| Slow Search | Decrease search_complexity, use compact mode |
| Poor Ranking (MAP) | Better embedding model, optimize chunk_size |
| First Result Poor (MRR) | Tune search_complexity, metadata filtering |

## Sample Datasets Included

1. **sample_qa.json**: Question-answering dataset (5 queries)
2. **sample_docs.json**: Document retrieval dataset (template)
3. **sample_code.json**: Code search dataset (template)

## Tools Reference

### evaluate.sh
Run evaluation on a single index.

```bash
./evaluation/scripts/evaluate.sh [OPTIONS]

Options:
  --index <name>           Index name
  --dataset <file>         Evaluation dataset JSON
  --k-values <list>        Comma-separated K values (default: 1,3,5,10,20)
  --output <dir>           Output directory for results
  --verbose                Show per-query metrics
```

### optimize.sh
Grid search over parameters.

```bash
./evaluation/scripts/optimize.sh [OPTIONS]

Options:
  --dataset <file>         Evaluation dataset JSON
  --optimize-for <metric>  Metric to optimize (default: ndcg@10)
  --graph-degree <list>    Values to test
  --search-complexity <list> Values to test
  --chunk-size <list>      Values to test
  --output <dir>           Output directory
```

### compare.sh
Compare two indexes side-by-side.

```bash
./evaluation/scripts/compare.sh [OPTIONS]

Options:
  --index-a <name>         First index
  --index-b <name>         Second index
  --dataset <file>         Evaluation dataset
  --output <dir>           Output directory
```

## Next Steps

1. **Create your evaluation dataset** based on real user queries
2. **Run baseline evaluation** to understand current performance
3. **Optimize parameters** using grid search
4. **Monitor over time** as you add more documents
5. **A/B test changes** before deploying to production

## References

- [Information Retrieval Evaluation](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)
- [NDCG Explained](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
- [TREC Evaluation](https://trec.nist.gov/data/reljudge_eng.html)
