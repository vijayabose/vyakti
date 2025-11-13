#!/bin/bash
# Vyakti Index Comparison Script
#
# Compare two indexes side-by-side using the same evaluation dataset.

set -e

# Default values
INDEX_A=""
INDEX_B=""
DATASET_FILE=""
OUTPUT_DIR="./evaluation/comparison"
INDEX_DIR=".vyakti"
K_VALUES="1,3,5,10,20"
EMBEDDING_MODEL="mxbai-embed-large"
EMBEDDING_DIM=1024

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
        --index-a)
            INDEX_A="$2"
            shift 2
            ;;
        --index-b)
            INDEX_B="$2"
            shift 2
            ;;
        --dataset)
            DATASET_FILE="$2"
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
        --k-values)
            K_VALUES="$2"
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
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --index-a <name>            First index name (required)"
            echo "  --index-b <name>            Second index name (required)"
            echo "  --dataset <file>            Evaluation dataset JSON file (required)"
            echo "  --output <dir>              Output directory (default: ./evaluation/comparison)"
            echo "  --index-dir <dir>           Index directory (default: .vyakti)"
            echo "  --k-values <list>           Comma-separated K values (default: 1,3,5,10,20)"
            echo "  --embedding-model <model>   Embedding model (default: mxbai-embed-large)"
            echo "  --embedding-dimension <dim> Embedding dimension (default: 1024)"
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
if [ -z "$INDEX_A" ]; then
    echo -e "${RED}Error: --index-a is required${NC}"
    exit 1
fi

if [ -z "$INDEX_B" ]; then
    echo -e "${RED}Error: --index-b is required${NC}"
    exit 1
fi

if [ -z "$DATASET_FILE" ]; then
    echo -e "${RED}Error: --dataset is required${NC}"
    exit 1
fi

if [ ! -f "$DATASET_FILE" ]; then
    echo -e "${RED}Error: Dataset file not found: $DATASET_FILE${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║            VYAKTI INDEX COMPARISON                       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Index A: ${YELLOW}$INDEX_A${NC}"
echo -e "  Index B: ${YELLOW}$INDEX_B${NC}"
echo -e "  Dataset: ${YELLOW}$DATASET_FILE${NC}"
echo -e "  Output: ${YELLOW}$OUTPUT_DIR${NC}"
echo ""

# Check if vyakti-evaluate binary exists
if [ ! -f "./target/release/vyakti-evaluate" ]; then
    echo -e "${YELLOW}⚠️  Building vyakti-evaluate binary...${NC}"
    cargo build --release --bin vyakti-evaluate
fi

# Output files
EVAL_A="$OUTPUT_DIR/${INDEX_A}_eval.json"
EVAL_B="$OUTPUT_DIR/${INDEX_B}_eval.json"

# Evaluate Index A
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Evaluating Index A: ${YELLOW}$INDEX_A${NC}"
echo ""

./target/release/vyakti-evaluate \
    --index "$INDEX_A" \
    --dataset "$DATASET_FILE" \
    --index-dir "$INDEX_DIR" \
    --k-values "$K_VALUES" \
    --embedding-model "$EMBEDDING_MODEL" \
    --embedding-dimension "$EMBEDDING_DIM" \
    --output "$EVAL_A" \
    | grep -E "(Dataset:|Queries:|Index loaded|Evaluation complete|P@10|R@10|NDCG@10|MRR|MAP)" || true

echo ""

# Evaluate Index B
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}Evaluating Index B: ${YELLOW}$INDEX_B${NC}"
echo ""

./target/release/vyakti-evaluate \
    --index "$INDEX_B" \
    --dataset "$DATASET_FILE" \
    --index-dir "$INDEX_DIR" \
    --k-values "$K_VALUES" \
    --embedding-model "$EMBEDDING_MODEL" \
    --embedding-dimension "$EMBEDDING_DIM" \
    --output "$EVAL_B" \
    | grep -E "(Dataset:|Queries:|Index loaded|Evaluation complete|P@10|R@10|NDCG@10|MRR|MAP)" || true

echo ""

# Compare results
echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║              COMPARISON RESULTS                          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Extract metrics
P10_A=$(jq -r '.mean_precision_at_k."10"' "$EVAL_A")
P10_B=$(jq -r '.mean_precision_at_k."10"' "$EVAL_B")

R10_A=$(jq -r '.mean_recall_at_k."10"' "$EVAL_A")
R10_B=$(jq -r '.mean_recall_at_k."10"' "$EVAL_B")

F1_10_A=$(jq -r '.mean_f1_at_k."10"' "$EVAL_A")
F1_10_B=$(jq -r '.mean_f1_at_k."10"' "$EVAL_B")

MAP_A=$(jq -r '.mean_average_precision' "$EVAL_A")
MAP_B=$(jq -r '.mean_average_precision' "$EVAL_B")

MRR_A=$(jq -r '.mean_reciprocal_rank' "$EVAL_A")
MRR_B=$(jq -r '.mean_reciprocal_rank' "$EVAL_B")

NDCG10_A=$(jq -r '.mean_ndcg_at_k."10"' "$EVAL_A")
NDCG10_B=$(jq -r '.mean_ndcg_at_k."10"' "$EVAL_B")

SEARCH_TIME_A=$(jq -r '.mean_search_time_ms' "$EVAL_A")
SEARCH_TIME_B=$(jq -r '.mean_search_time_ms' "$EVAL_B")

# Helper function to compare and show winner
compare_metric() {
    local name=$1
    local a=$2
    local b=$3
    local higher_is_better=${4:-true}

    local diff=$(echo "scale=4; $b - $a" | bc)
    local pct_change=$(echo "scale=2; ($diff / $a) * 100" | bc)

    printf "%-20s" "$name"
    printf "%-15s" "$a"
    printf "%-15s" "$b"
    printf "%-15s" "$diff"

    if [ "$higher_is_better" = "true" ]; then
        if (( $(echo "$b > $a" | bc -l) )); then
            printf "${GREEN}↑ %.1f%% (B wins)${NC}\n" "$pct_change"
        elif (( $(echo "$b < $a" | bc -l) )); then
            printf "${RED}↓ %.1f%% (A wins)${NC}\n" "${pct_change#-}"
        else
            printf "${YELLOW}= (tie)${NC}\n"
        fi
    else
        # For metrics like search time where lower is better
        if (( $(echo "$b < $a" | bc -l) )); then
            printf "${GREEN}↓ %.1f%% (B wins)${NC}\n" "${pct_change#-}"
        elif (( $(echo "$b > $a" | bc -l) )); then
            printf "${RED}↑ %.1f%% (A wins)${NC}\n" "$pct_change"
        else
            printf "${YELLOW}= (tie)${NC}\n"
        fi
    fi
}

# Print comparison table
echo -e "${CYAN}Metric Comparison:${NC}"
echo ""
printf "%-20s %-15s %-15s %-15s %-20s\n" "Metric" "Index A" "Index B" "Difference" "Winner"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

compare_metric "Precision@10" "$P10_A" "$P10_B"
compare_metric "Recall@10" "$R10_A" "$R10_B"
compare_metric "F1@10" "$F1_10_A" "$F1_10_B"
compare_metric "MAP" "$MAP_A" "$MAP_B"
compare_metric "MRR" "$MRR_A" "$MRR_B"
compare_metric "NDCG@10" "$NDCG10_A" "$NDCG10_B"
compare_metric "Search Time (ms)" "$SEARCH_TIME_A" "$SEARCH_TIME_B" false

echo ""

# Calculate overall winner
WINS_A=0
WINS_B=0

# Count wins for each metric (higher is better)
for metric_pair in "P10_A:P10_B" "R10_A:R10_B" "F1_10_A:F1_10_B" "MAP_A:MAP_B" "MRR_A:MRR_B" "NDCG10_A:NDCG10_B"; do
    IFS=':' read -r var_a var_b <<< "$metric_pair"
    val_a="${!var_a}"
    val_b="${!var_b}"

    if (( $(echo "$val_a > $val_b" | bc -l) )); then
        WINS_A=$((WINS_A + 1))
    elif (( $(echo "$val_b > $val_a" | bc -l) )); then
        WINS_B=$((WINS_B + 1))
    fi
done

# For search time, lower is better
if (( $(echo "$SEARCH_TIME_A < $SEARCH_TIME_B" | bc -l) )); then
    WINS_A=$((WINS_A + 1))
elif (( $(echo "$SEARCH_TIME_B < $SEARCH_TIME_A" | bc -l) )); then
    WINS_B=$((WINS_B + 1))
fi

echo -e "${CYAN}Overall Results:${NC}"
echo -e "  Index A (${YELLOW}$INDEX_A${NC}) wins: ${YELLOW}$WINS_A${NC} metrics"
echo -e "  Index B (${YELLOW}$INDEX_B${NC}) wins: ${YELLOW}$WINS_B${NC} metrics"
echo ""

if [ $WINS_A -gt $WINS_B ]; then
    echo -e "${GREEN}🏆 Winner: Index A (${INDEX_A})${NC}"
elif [ $WINS_B -gt $WINS_A ]; then
    echo -e "${GREEN}🏆 Winner: Index B (${INDEX_B})${NC}"
else
    echo -e "${YELLOW}🤝 Tie - Both indexes perform equally${NC}"
fi

echo ""
echo -e "${CYAN}Detailed results saved to:${NC}"
echo -e "  ${YELLOW}$EVAL_A${NC}"
echo -e "  ${YELLOW}$EVAL_B${NC}"
echo ""
echo -e "${GREEN}✓ Comparison complete${NC}"
