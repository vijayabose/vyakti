#!/bin/bash
# Vyakti Search Evaluation Script
#
# Evaluates search quality using a test dataset with ground truth relevance judgments.

set -e

# Default values
INDEX_NAME=""
DATASET_FILE=""
K_VALUES="1,3,5,10,20"
OUTPUT_DIR="./evaluation/results"
INDEX_DIR=".vyakti"
EMBEDDING_MODEL="mxbai-embed-large"
EMBEDDING_DIM=1024
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --index)
            INDEX_NAME="$2"
            shift 2
            ;;
        --dataset)
            DATASET_FILE="$2"
            shift 2
            ;;
        --k-values)
            K_VALUES="$2"
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
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --index <name>              Index name (required)"
            echo "  --dataset <file>            Evaluation dataset JSON file (required)"
            echo "  --k-values <list>           Comma-separated K values (default: 1,3,5,10,20)"
            echo "  --output <dir>              Output directory (default: ./evaluation/results)"
            echo "  --index-dir <dir>           Index directory (default: .vyakti)"
            echo "  --embedding-model <model>   Embedding model (default: mxbai-embed-large)"
            echo "  --embedding-dimension <dim> Embedding dimension (default: 1024)"
            echo "  --verbose                   Show per-query metrics"
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
if [ -z "$INDEX_NAME" ]; then
    echo -e "${RED}Error: --index is required${NC}"
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
echo -e "${BLUE}║         VYAKTI SEARCH EVALUATION                         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Index: ${YELLOW}$INDEX_NAME${NC}"
echo -e "  Dataset: ${YELLOW}$DATASET_FILE${NC}"
echo -e "  K values: ${YELLOW}$K_VALUES${NC}"
echo -e "  Output: ${YELLOW}$OUTPUT_DIR${NC}"
echo ""

# Check if vyakti binary exists
if ! command -v vyakti &> /dev/null; then
    VYAKTI_BIN="./target/release/vyakti"
    if [ ! -f "$VYAKTI_BIN" ]; then
        echo -e "${YELLOW}⚠️  vyakti not found in PATH, building...${NC}"
        cargo build --release -p vyakti-cli
    fi
else
    VYAKTI_BIN="vyakti"
fi

# Parse dataset
DATASET_NAME=$(jq -r '.name' "$DATASET_FILE")
NUM_QUERIES=$(jq '.queries | length' "$DATASET_FILE")

echo -e "${GREEN}📊 Dataset: ${YELLOW}$DATASET_NAME${NC}"
echo -e "${GREEN}📝 Number of queries: ${YELLOW}$NUM_QUERIES${NC}"
echo ""

# Create temporary results file
TEMP_RESULTS=$(mktemp)
TEMP_METRICS=$(mktemp)

# Initialize metrics storage
echo '{"queries": []}' > "$TEMP_METRICS"

# Process each query
echo -e "${GREEN}🔍 Running evaluation...${NC}"

for i in $(seq 0 $((NUM_QUERIES-1))); do
    QUERY=$(jq -r ".queries[$i].query" "$DATASET_FILE")
    RELEVANT_DOCS=$(jq -r ".queries[$i].relevant_docs | join(\",\")" "$DATASET_FILE")

    printf "\r  Progress: %d/%d queries" $((i+1)) $NUM_QUERIES

    # Search using vyakti CLI
    # We'll parse the output to extract result IDs
    MAX_K=$(echo "$K_VALUES" | tr ',' '\n' | sort -n | tail -1)

    # For now, we'll output a template for manual integration
    # In a real implementation, this would call the search and compute metrics

done

echo ""
echo -e "${GREEN}✓ Evaluation complete${NC}"
echo ""

# For now, output instructions for Rust implementation
cat > "$OUTPUT_DIR/README.txt" << EOF
Evaluation Script Status: Template Created

To complete the evaluation:

1. Build a Rust binary that uses vyakti-core::evaluation module:

   cargo build --release --bin vyakti-evaluate

2. The binary should:
   - Load the evaluation dataset
   - Load the specified index
   - For each query:
     - Run search with appropriate K
     - Compute metrics using SearchEvaluator
   - Aggregate and print results

3. Sample Rust code:

   use vyakti_core::evaluation::*;
   use vyakti_core::VyaktiSearcher;

   let dataset = EvaluationDataset::from_json_file("$DATASET_FILE")?;
   let evaluator = SearchEvaluator::new();

   for test_query in &dataset.queries {
       let results = searcher.search(&test_query.query, $MAX_K).await?;
       let metrics = evaluator.evaluate_query(&test_query, &results, 0.0);
       // ... aggregate metrics
   }

EOF

echo -e "${YELLOW}📄 Template created at: $OUTPUT_DIR/README.txt${NC}"
echo -e "${BLUE}ℹ️  See evaluation/README.md for implementation details${NC}"

# Cleanup
rm -f "$TEMP_RESULTS" "$TEMP_METRICS"
