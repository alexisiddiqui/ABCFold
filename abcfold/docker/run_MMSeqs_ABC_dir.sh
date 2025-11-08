#!/bin/bash

# Default values (can be overridden via CLI)
INPUT_DIR="/lus/lfs1aip2/projects/s5h/hussian-simulation-hdx/projects/AF3_Prediction/AF3-Pipeline/examples/ATLAS/_msa_jsons/1a62A/test"
OUTPUT_BASE_DIR="/lus/lfs1aip2/projects/s5h/hussian-simulation-hdx/projects/AF3_Prediction/ATLAS_outputs/_ABC_MMSEQsMSA_test"

# AlphaFold3 and Boltz/Chai paths
SIF_PATH="/lus/lfs1aip2/projects/s5h/public/hussain/alphafold3/docker/alphafold3_arm64.sif"
MODEL_PARAMS="/lus/lfs1aip2/projects/s5h/hussian-simulation-hdx/projects/ATLAS_MSA/AF3_weights"
BOLTZ_SIF_PATH="/lus/lfs1aip2/projects/s5h/hussian-simulation-hdx/projects/ABCFold/abcfold/docker/Boltz/boltz2_GH200.sif"
CHAI_SIF_PATH="/lus/lfs1aip2/projects/s5h/hussian-simulation-hdx/projects/ABCFold/abcfold/docker/Chai/chai2_GH200.sif"

# Usage/help
usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -i, --input-dir DIR      Input directory containing JSON files
                           (default: $INPUT_DIR)
  -o, --output-dir DIR     Base output directory
                           (default: $OUTPUT_BASE_DIR)
  -h, --help               Show this help
EOF
}

# Parse command-line arguments (overrides defaults)
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input-dir)
            if [[ -n "$2" ]]; then
                INPUT_DIR="$2"
                shift 2
            else
                echo "Error: --input-dir requires a value" >&2
                usage
                exit 1
            fi
            ;;
        -o|--output-dir|--output)
            if [[ -n "$2" ]]; then
                OUTPUT_BASE_DIR="$2"
                shift 2
            else
                echo "Error: --output-dir requires a value" >&2
                usage
                exit 1
            fi
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

# GPU logging: start a background logger that writes to $OUTPUT_BASE_DIR/<dir>_<time>.gpu.log
GPU_LOG_INTERVAL=1  # seconds (adjust as needed)
_logger_pid=""
# GPU_LOG_FILE will be set when logger starts to include dir name and start time

start_gpu_logger() {
    mkdir -p "$OUTPUT_BASE_DIR"
    # Use INPUT_DIR basename as prefix and include timestamp
    dir_name="$(basename "$INPUT_DIR")"
    gpu_ts="$(date -u +"%Y%m%d_%H%M%S")"
    GPU_LOG_FILE="${OUTPUT_BASE_DIR}/${dir_name}_${gpu_ts}.gpu.log"

    # Header for the log
    {
        echo "==== GPU log started: $(date -u +"%Y-%m-%dT%H:%M:%SZ") ===="
        echo "Log file: ${GPU_LOG_FILE}"
        echo "Columns: timestamp,gpu_index,gpu_name,mem_used_MB,mem_total_MB,power_draw_W"
    } >>"$GPU_LOG_FILE"

    # If nvidia-smi missing, write a note and return
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ"),nvidia-smi,NOT_FOUND" >>"$GPU_LOG_FILE"
        return
    fi

    # Background logger loop
    (
        while true; do
            timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
            nvidia-smi --query-gpu=index,name,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>/dev/null |
            while IFS=',' read -r idx name mem_used mem_total power; do
                idx="${idx//[[:space:]]/}"
                name="$(echo "$name" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
                mem_used="${mem_used//[[:space:]]/}"
                mem_total="${mem_total//[[:space:]]/}"
                power="${power//[[:space:]]/}"
                echo "${timestamp},${idx},${name},${mem_used},${mem_total},${power}"
            done >>"$GPU_LOG_FILE"
            sleep "$GPU_LOG_INTERVAL"
        done
    ) &
    _logger_pid=$!
}

stop_gpu_logger() {
    if [[ -n "$_logger_pid" ]]; then
        kill "$_logger_pid" 2>/dev/null || true
        wait "$_logger_pid" 2>/dev/null || true
        _logger_pid=""
        # Append end marker to the same GPU_LOG_FILE set at start
        if [[ -n "${GPU_LOG_FILE:-}" ]]; then
            echo "==== GPU log ended: $(date -u +"%Y-%m-%dT%H:%M:%SZ") ====" >>"$GPU_LOG_FILE"
        fi
    fi
}

# Ensure logger is stopped on any exit
trap stop_gpu_logger EXIT

# Start GPU logging for the whole script
start_gpu_logger

# Find all JSON files recursively
find "$INPUT_DIR" -type f -name "*.json" | while read -r json_file; do
    # Get relative path from INPUT_DIR
    rel_path=$(realpath --relative-to="$INPUT_DIR" "$json_file")
    rel_dir=$(dirname "$rel_path")
    json_basename=$(basename "$json_file" .json)

    # Generate timestamp with milliseconds
    timestamp=$(date +%Y%m%d_%H%M%S)
    timestamp_ms="${timestamp}_$(date +%N | cut -c1-3)"

    # Create output directory path
    output_dir="${OUTPUT_BASE_DIR}/${rel_dir}/${json_basename}_${timestamp_ms}"

    echo "============================================"
    echo "Processing: $json_file"
    echo "Output to: $output_dir"
    echo "============================================"

    # Run abcfold
    if abcfold "$json_file" "$output_dir" \
        -abc \
        --sif_path "$SIF_PATH" \
        --model_params "$MODEL_PARAMS" \
        --override \
        --no_visuals \
        --number_of_models 25 \
        --boltz_sif_path "$BOLTZ_SIF_PATH" \
        --chai_sif_path "$CHAI_SIF_PATH"; then

        echo "ABCFold completed successfully for $json_basename"

        # Compress the output directory
        echo "Compressing output directory..."
        if tar -cf - "$output_dir" | xz -T0 > "${output_dir}.tar.xz"; then
            echo "Compression successful: ${output_dir}.tar.xz"

            # Remove the original directory
            echo "Removing uncompressed directory..."
            rm -rf "$output_dir"
            echo "Cleanup complete"
        else
            echo "ERROR: Compression failed for $output_dir"
        fi
    else
        echo "ERROR: ABCFold failed for $json_file"
    fi

    echo ""
done

echo "============================================"
echo "All JSON files processed!"
echo "============================================"
