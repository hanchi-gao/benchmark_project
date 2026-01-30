#!/bin/bash
# AMD vLLM Benchmark Script
# Customizable GPU configuration benchmark runner

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default parameters
GPU_COUNT=1
GPU_IDS="0"
MODEL=""
GPU_MEM_UTIL=0.9
INPUT_LEN=1024
OUTPUT_LEN=128
NUM_PROMPTS_START=${NUM_PROMPTS_START:-1}
NUM_PROMPTS_END=${NUM_PROMPTS_END:-200}
EXPERIMENT_NAME=""
MAX_MODEL_LEN=""
# Default to internal container URL when running inside Docker
VLLM_SERVER_URL="${VLLM_SERVER_URL:-http://vllm-server:8000}"
RESULTS_DIR="/root/output"

# Show usage
show_usage() {
    cat << EOF
Usage: $0 --model <MODEL> --experiment-name <NAME> [OPTIONS]

Required Parameters:
  --model <name>              Model name or HuggingFace path
  --experiment-name <name>    Name for the experiment folder

GPU Configuration:
  --gpu-count <N>             Number of GPUs to use (default: 1)
  --gpu-ids <IDs>             Comma-separated GPU IDs (default: "0")
                              Examples: "0", "0,1", "2,3,4,5"

Benchmark Parameters:
  --gpu-memory-utilization <F>  GPU memory utilization (default: 0.9)
  --input-len <tokens>          Input length in tokens (default: 1024)
  --output-len <tokens>         Output length in tokens (default: 128)
  --num-prompts-start <N>       Start of prompts range (default: 1)
  --num-prompts-end <N>         End of prompts range (default: 100)
  --max-model-len <tokens>      Max model length (default: input+output+128)

Server Configuration:
  --server-url <URL>          vLLM server URL (default: http://localhost:8000)

Other:
  --help                      Show this help message

Examples:
  # Single GPU benchmark
  $0 --model meta-llama/Llama-3.1-8B --experiment-name test1 --gpu-count 1

  # Dual GPU with specific IDs
  $0 --model meta-llama/Llama-3.1-8B --experiment-name test2 \\
     --gpu-count 2 --gpu-ids "0,1"

  # Custom prompts range
  $0 --model meta-llama/Llama-3.1-8B --experiment-name test3 \\
     --num-prompts-start 1 --num-prompts-end 200

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --gpu-memory-utilization)
            GPU_MEM_UTIL="$2"
            shift 2
            ;;
        --input-len)
            INPUT_LEN="$2"
            shift 2
            ;;
        --output-len)
            OUTPUT_LEN="$2"
            shift 2
            ;;
        --num-prompts-start)
            NUM_PROMPTS_START="$2"
            shift 2
            ;;
        --num-prompts-end)
            NUM_PROMPTS_END="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --server-url)
            VLLM_SERVER_URL="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown parameter: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$MODEL" ]]; then
    echo -e "${RED}Error: --model is required${NC}"
    show_usage
    exit 1
fi

if [[ -z "$EXPERIMENT_NAME" ]]; then
    echo -e "${RED}Error: --experiment-name is required${NC}"
    show_usage
    exit 1
fi

# Calculate max model length if not specified
if [[ -z "$MAX_MODEL_LEN" ]]; then
    MAX_MODEL_LEN=$((INPUT_LEN + OUTPUT_LEN + 128))
fi

# Set GPU environment
export HIP_VISIBLE_DEVICES="$GPU_IDS"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

# Results directory for this experiment
EXPERIMENT_DIR="${RESULTS_DIR}/${EXPERIMENT_NAME}"
mkdir -p "$EXPERIMENT_DIR"

# Display configuration
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║            AMD vLLM Benchmark                              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo -e "  Experiment:           ${YELLOW}${EXPERIMENT_NAME}${NC}"
echo -e "  Model:                ${YELLOW}${MODEL}${NC}"
echo -e "  GPU Count:            ${YELLOW}${GPU_COUNT}${NC}"
echo -e "  GPU IDs:              ${YELLOW}${GPU_IDS}${NC}"
echo -e "  GPU Memory Util:      ${YELLOW}${GPU_MEM_UTIL}${NC}"
echo -e "  Input Length:         ${YELLOW}${INPUT_LEN} tokens${NC}"
echo -e "  Output Length:        ${YELLOW}${OUTPUT_LEN} tokens${NC}"
echo -e "  Max Model Length:     ${YELLOW}${MAX_MODEL_LEN} tokens${NC}"
echo -e "  Prompts Range:        ${YELLOW}${NUM_PROMPTS_START} - ${NUM_PROMPTS_END}${NC}"
echo -e "  Server URL:           ${YELLOW}${VLLM_SERVER_URL}${NC}"
echo -e "  Results Directory:    ${YELLOW}${EXPERIMENT_DIR}${NC}"
echo ""

# Check vLLM server
echo -e "${BLUE}[1/3] Checking vLLM server status...${NC}"
if ! curl -s "${VLLM_SERVER_URL}/health" > /dev/null 2>&1; then
    echo -e "${RED}Error: Cannot connect to vLLM server at ${VLLM_SERVER_URL}${NC}"
    echo ""
    echo -e "${YELLOW}Please start the vLLM server first:${NC}"
    echo ""
    echo "  vllm serve ${MODEL} \\"
    echo "    --tensor-parallel-size ${GPU_COUNT} \\"
    echo "    --gpu-memory-utilization ${GPU_MEM_UTIL} \\"
    echo "    --max-model-len ${MAX_MODEL_LEN} \\"
    echo "    --enforce-eager"
    echo ""
    exit 1
fi
echo -e "${GREEN}✓ vLLM server is running${NC}"
echo ""

# Verify model
echo -e "${BLUE}[2/3] Verifying model...${NC}"
LOADED_MODEL=$(curl -s "${VLLM_SERVER_URL}/v1/models" | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4 2>/dev/null || echo "unknown")
if [[ "$LOADED_MODEL" != "$MODEL" ]]; then
    echo -e "${YELLOW}Warning: Server model '${LOADED_MODEL}' differs from '${MODEL}'${NC}"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ Model verified: ${LOADED_MODEL}${NC}"
fi
echo ""

# Run benchmarks
echo -e "${BLUE}[3/3] Running benchmark suite...${NC}"
echo -e "${YELLOW}Testing num_prompts from ${NUM_PROMPTS_START} to ${NUM_PROMPTS_END}${NC}"
echo ""

TOTAL_TESTS=$((NUM_PROMPTS_END - NUM_PROMPTS_START + 1))
COMPLETED=0
FAILED=0

for NUM_PROMPTS in $(seq $NUM_PROMPTS_START $NUM_PROMPTS_END); do
    COMPLETED=$((COMPLETED + 1))
    PROGRESS=$((COMPLETED * 100 / TOTAL_TESTS))

    echo -ne "${CYAN}[${COMPLETED}/${TOTAL_TESTS}] (${PROGRESS}%) Testing num_prompts=${NUM_PROMPTS}...${NC}"

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RESULT_FILE="${EXPERIMENT_DIR}/np${NUM_PROMPTS}_${TIMESTAMP}.json"
    TEMP_FILE="temp_result_${NUM_PROMPTS}.json"

    # Run benchmark
    if vllm bench serve \
        --model "$MODEL" \
        --backend openai \
        --endpoint /v1/completions \
        --base-url "$VLLM_SERVER_URL" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --ignore-eos \
        --save-result \
        --result-filename "$TEMP_FILE" > /dev/null 2>&1; then

        if [[ -f "$TEMP_FILE" ]]; then
            mv "$TEMP_FILE" "$RESULT_FILE"
            echo -e " ${GREEN}OK${NC}"
        else
            echo -e " ${RED}FAIL (no output)${NC}"
            FAILED=$((FAILED + 1))
        fi
    else
        echo -e " ${RED}FAIL${NC}"
        FAILED=$((FAILED + 1))
        rm -f "$TEMP_FILE"
    fi
done

# Summary
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║            Benchmark Complete                              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Total:     ${BLUE}${TOTAL_TESTS}${NC}"
echo -e "  Completed: ${GREEN}$((TOTAL_TESTS - FAILED))${NC}"
echo -e "  Failed:    ${RED}${FAILED}${NC}"
echo ""
echo -e "  Results:   ${YELLOW}${EXPERIMENT_DIR}${NC}"
echo ""

exit 0
