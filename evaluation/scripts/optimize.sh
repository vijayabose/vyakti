#!/bin/bash
# Vyakti Parameter Optimization Script
#
# Performs grid search over key parameters to find optimal configuration.

set -e

# Default values
DATASET_FILE=""
INPUT_DOCS=""
OPTIMIZE_METRIC="ndcg@10"
OUTPUT_DIR="./evaluation/optimization"
INDEX_DIR=".vyakti"
EMBEDDING_MODEL="mxbai-embed-large"
EMBEDDING_DIM=1024

# Parameter ranges
GRAPH_DEGREES="16,32,64"
SEARCH_COMPLEXITIES="16,32,64,128"
CHUNK_SIZES="128,256,512"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_FILE="$2"
            shift 2
            ;;
        --input)
            INPUT_DOCS="$2"
            shift 2
            ;;
        --optimize-for)
            OPTIMIZE_METRIC="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --index-dir)
            INDEX_DIR="$2"
            shift 2
            ;;
        --embedding-model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --embedding-dimension)
            EMBEDDING_DIM="$2"
            shift 2
            ;;
        --graph-degree)
            GRAPH_DEGREES="$2"
            shift 2
            ;;
        --search-complexity)
            SEARCH_COMPLEXITIES="$2"
            shift 2
            ;;
        --chunk-size)
            CHUNK_SIZES="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset <file>            Evaluation dataset JSON file (required)"
            echo "  --input <dir>               Input documents directory (required)"
            echo "  --optimize-for <metric>     Metric to optimize (default: ndcg@10)"
            echo "  --output <dir>              Output directory (default: ./evaluation/optimization)"
            echo "  --index-dir <dir>           Index directory (default: .vyakti)"
            echo "  --embedding-model <model>   Embedding model (default: mxbai-embed-large)"
            echo "  --embedding-dimension <dim> Embedding dimension (default: 1024)"
            echo "  --graph-degree <list>       Comma-separated values to test (default: 16,32,64)"
            echo "  --search-complexity <list>  Comma-separated values to test (default: 16,32,64,128)"
            echo "  --chunk-size <list>         Comma-separated values to test (default: 128,256,512)"
            echo "  --help                      Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$DATASET_FILE" ]; then
    echo -e "${RED}Error: --dataset is required${NC}"
    exit 1
fi

if [ -z "$INPUT_DOCS" ]; then
    echo -e "${RED}Error: --input is required${NC}"
    exit 1
fi

if [ ! -f "$DATASET_FILE" ]; then
    echo -e "${RED}Error: Dataset file not found: $DATASET_FILE${NC}"
    exit 1
fi

if [ ! -d "$INPUT_DOCS" ]; then
    echo -e "${RED}Error: Input directory not found: $INPUT_DOCS${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║      VYAKTI PARAMETER OPTIMIZATION                       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Dataset: ${YELLOW}$DATASET_FILE${NC}"
echo -e "  Input: ${YELLOW}$INPUT_DOCS${NC}"
echo -e "  Optimize for: ${YELLOW}$OPTIMIZE_METRIC${NC}"
echo -e "  Output: ${YELLOW}$OUTPUT_DIR${NC}"
echo ""

# Check if vyakti-evaluate binary exists
if [ ! -f "./target/release/vyakti-evaluate" ]; then
    echo -e "${YELLOW}⚠️  Building vyakti-evaluate binary...${NC}"
    cargo build --release --bin vyakti-evaluate
fi

# Check if vyakti binary exists
if [ ! -f "./target/release/vyakti" ]; then
    echo -e "${YELLOW}⚠️  Building vyakti CLI...${NC}"
    cargo build --release --bin vyakti
fi

# Convert comma-separated values to arrays
IFS=',' read -ra GD_ARRAY <<< "$GRAPH_DEGREES"
IFS=',' read -ra SC_ARRAY <<< "$SEARCH_COMPLEXITIES"
IFS=',' read -ra CS_ARRAY <<< "$CHUNK_SIZES"

# Calculate total combinations
TOTAL_COMBINATIONS=$((${#GD_ARRAY[@]} * ${#SC_ARRAY[@]} * ${#CS_ARRAY[@]}))

echo -e "${GREEN}Parameter Grid:${NC}"
echo -e "  Graph degrees: ${YELLOW}${GRAPH_DEGREES}${NC}"
echo -e "  Search complexities: ${YELLOW}${SEARCH_COMPLEXITIES}${NC}"
echo -e "  Chunk sizes: ${YELLOW}${CHUNK_SIZES}${NC}"
echo -e "  ${CYAN}Total combinations: ${TOTAL_COMBINATIONS}${NC}"
echo ""

# Results file
RESULTS_FILE="$OUTPUT_DIR/optimization_results.csv"
echo "graph_degree,search_complexity,chunk_size,metric,score,build_time_s,avg_search_time_ms" > "$RESULTS_FILE"

# Best result tracking
BEST_SCORE=-1
BEST_CONFIG=""

# Progress counter
CURRENT=0

# Grid search
for GRAPH_DEGREE in "${GD_ARRAY[@]}"; do
    GRAPH_DEGREE=$(echo "$GRAPH_DEGREE" | xargs) # trim whitespace
    for SEARCH_COMPLEXITY in "${SC_ARRAY[@]}"; do
        SEARCH_COMPLEXITY=$(echo "$SEARCH_COMPLEXITY" | xargs)
        for CHUNK_SIZE in "${CS_ARRAY[@]}"; do
            CHUNK_SIZE=$(echo "$CHUNK_SIZE" | xargs)

            CURRENT=$((CURRENT + 1))

            echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            echo -e "${CYAN}Configuration $CURRENT/$TOTAL_COMBINATIONS${NC}"
            echo -e "  graph_degree: ${YELLOW}$GRAPH_DEGREE${NC}"
            echo -e "  search_complexity: ${YELLOW}$SEARCH_COMPLEXITY${NC}"
            echo -e "  chunk_size: ${YELLOW}$CHUNK_SIZE${NC}"
            echo ""

            # Index name
            INDEX_NAME="optimize_gd${GRAPH_DEGREE}_sc${SEARCH_COMPLEXITY}_cs${CHUNK_SIZE}"

            # Build index
            echo -e "${GREEN}📦 Building index...${NC}"
            BUILD_START=$(date +%s)

            ./target/release/vyakti build "$INDEX_NAME" \
                --input "$INPUT_DOCS" \
                --graph-degree "$GRAPH_DEGREE" \
                --chunk-size "$CHUNK_SIZE" \
                --embedding-model "$EMBEDDING_MODEL" \
                --compact \
                > /dev/null 2>&1

            BUILD_END=$(date +%s)
            BUILD_TIME=$((BUILD_END - BUILD_START))

            echo -e "  ${GREEN}✓${NC} Index built in ${BUILD_TIME}s"

            # Run evaluation
            echo -e "${GREEN}🔍 Evaluating...${NC}"
            EVAL_OUTPUT="$OUTPUT_DIR/${INDEX_NAME}_eval.json"

            ./target/release/vyakti-evaluate \
                --index "$INDEX_NAME" \
                --dataset "$DATASET_FILE" \
                --index-dir "$INDEX_DIR" \
                --embedding-model "$EMBEDDING_MODEL" \
                --embedding-dimension "$EMBEDDING_DIM" \
                --output "$EVAL_OUTPUT" \
                > /dev/null 2>&1

            # Extract metric score from JSON
            METRIC_KEY=$(echo "$OPTIMIZE_METRIC" | sed 's/@/_at_k./g')

            # Parse JSON based on metric type
            if [[ "$OPTIMIZE_METRIC" == *"@"* ]]; then
                # Metrics with K values (e.g., ndcg@10)
                METRIC_BASE=$(echo "$OPTIMIZE_METRIC" | cut -d'@' -f1)
                K_VALUE=$(echo "$OPTIMIZE_METRIC" | cut -d'@' -f2)

                case "$METRIC_BASE" in
                    "precision")
                        SCORE=$(jq -r ".mean_precision_at_k.\"$K_VALUE\"" "$EVAL_OUTPUT")
                        ;;
                    "recall")
                        SCORE=$(jq -r ".mean_recall_at_k.\"$K_VALUE\"" "$EVAL_OUTPUT")
                        ;;
                    "f1")
                        SCORE=$(jq -r ".mean_f1_at_k.\"$K_VALUE\"" "$EVAL_OUTPUT")
                        ;;
                    "ndcg")
                        SCORE=$(jq -r ".mean_ndcg_at_k.\"$K_VALUE\"" "$EVAL_OUTPUT")
                        ;;
                    *)
                        echo -e "${RED}Error: Unknown metric $OPTIMIZE_METRIC${NC}"
                        exit 1
                        ;;
                esac
            else
                # Single-value metrics (map, mrr)
                case "$OPTIMIZE_METRIC" in
                    "map")
                        SCORE=$(jq -r ".mean_average_precision" "$EVAL_OUTPUT")
                        ;;
                    "mrr")
                        SCORE=$(jq -r ".mean_reciprocal_rank" "$EVAL_OUTPUT")
                        ;;
                    *)
                        echo -e "${RED}Error: Unknown metric $OPTIMIZE_METRIC${NC}"
                        exit 1
                        ;;
                esac
            fi

            # Get average search time
            AVG_SEARCH_TIME=$(jq -r ".mean_search_time_ms" "$EVAL_OUTPUT")

            # Save result
            echo "$GRAPH_DEGREE,$SEARCH_COMPLEXITY,$CHUNK_SIZE,$OPTIMIZE_METRIC,$SCORE,$BUILD_TIME,$AVG_SEARCH_TIME" >> "$RESULTS_FILE"

            echo -e "  ${CYAN}${OPTIMIZE_METRIC}:${NC} ${YELLOW}$SCORE${NC}"
            echo -e "  ${CYAN}Avg search time:${NC} ${YELLOW}${AVG_SEARCH_TIME}ms${NC}"

            # Update best result
            IS_BETTER=$(echo "$SCORE > $BEST_SCORE" | bc -l)
            if [ "$IS_BETTER" -eq 1 ] || [ "$BEST_SCORE" = "-1" ]; then
                BEST_SCORE="$SCORE"
                BEST_CONFIG="graph_degree=$GRAPH_DEGREE, search_complexity=$SEARCH_COMPLEXITY, chunk_size=$CHUNK_SIZE"
                echo -e "  ${GREEN}🏆 New best configuration!${NC}"
            fi

            # Clean up index to save space
            ./target/release/vyakti remove "$INDEX_NAME" --index-dir "$INDEX_DIR" > /dev/null 2>&1 || true

            echo ""
        done
    done
done

echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              OPTIMIZATION COMPLETE                       ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Best Configuration:${NC}"
echo -e "  ${YELLOW}$BEST_CONFIG${NC}"
echo -e "  ${CYAN}${OPTIMIZE_METRIC}:${NC} ${YELLOW}$BEST_SCORE${NC}"
echo ""
echo -e "${CYAN}Full results saved to:${NC} ${YELLOW}$RESULTS_FILE${NC}"
echo ""

# Sort results by score
SORTED_RESULTS="$OUTPUT_DIR/optimization_results_sorted.csv"
head -n 1 "$RESULTS_FILE" > "$SORTED_RESULTS"
tail -n +2 "$RESULTS_FILE" | sort -t',' -k5 -rn >> "$SORTED_RESULTS"

echo -e "${CYAN}Top 5 configurations:${NC}"
echo ""
head -n 6 "$SORTED_RESULTS" | column -t -s','

echo ""
echo -e "${GREEN}✓ Optimization complete${NC}"
