#!/bin/bash
# Author: Alexander G. Hasson
# University of Oxford
# MMseqs2 Pipeline for AlphaFold3 with ABCFold integration
# Outputs MSAs in AF3 JSON submission format
#
# Usage: ./run_mmseqs2_af3_pipeline.sh <BASE_DIR> <BATCH_DIR> <HPC_ENV> <MODE>
# MODE: "msa" for MSA generation, "templates" for MSA + templates

set -e  # Exit on error

# =============================================================================
# Parse arguments
# =============================================================================
BASE_DIR=$1
BATCH_DIR=$2
HPC_ENV=$3
MODE=${4:-"msa"}  # Default to MSA only

# =============================================================================
# Validate input parameters
# =============================================================================
if [ -z "$BASE_DIR" ] || [ -z "$BATCH_DIR" ] || [ -z "$HPC_ENV" ]; then
    echo "Usage: $0 <BASE_DIR> <BATCH_DIR> <HPC_ENV> [MODE]"
    echo "  BASE_DIR: Base working directory"
    echo "  BATCH_DIR: Batch directory name (e.g., batch_0)"
    echo "  HPC_ENV: 'ARC', 'STATS', or 'AI'"
    echo "  MODE: 'msa' (default) or 'templates' (MSA + templates)"
    exit 1
fi

if [ "$MODE" != "msa" ] && [ "$MODE" != "templates" ]; then
    echo "Invalid MODE: ${MODE}. Must be 'msa' or 'templates'. Exiting." >&2
    exit 1
fi

# =============================================================================
# Set HPC-specific paths
# =============================================================================
if [ "$HPC_ENV" == "ARC" ]; then
    AF3_DATABASE_DIR="/apps/datasets/alphafold3/public_databases/"
    ABCFOLD_PATH="/path/to/ABCFold"  # Update this
elif [ "$HPC_ENV" == "STATS" ]; then
    echo "STATS environment not yet implemented. Exiting" >&2
    exit 1
elif [ "$HPC_ENV" == "AI" ]; then
    AF3_DATABASE_DIR="/lus/lfs1aip2/projects/s5h/public/hussain/AF3_MSA_DB/"
    ABCFOLD_PATH="/lus/lfs1aip2/projects/s5h/hussian-simulation-hdx/projects/ABCFold"
else
    echo "Invalid HPC environment: ${HPC_ENV}. Must be 'ARC', 'STATS', or 'AI'. Exiting." >&2
    exit 1
fi

# =============================================================================
# Set up directories following the original script structure
# =============================================================================
INPUT_BASE_DIR="$BASE_DIR/examples/ATLAS/_empty_jsons"
INPUT_DIR="$INPUT_BASE_DIR/$BATCH_DIR"

OUTPUT_BASE_DIR="$BASE_DIR/examples/ATLAS/_msa_jsons"
OUTPUT_DIR="$OUTPUT_BASE_DIR/$BATCH_DIR"

MMSEQS_WORKDIR="$BASE_DIR/mmseqs2_workdir/$BATCH_DIR"
mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$MMSEQS_WORKDIR"

# =============================================================================
# Validate directories and files
# =============================================================================
if [ ! -d "$BASE_DIR" ]; then
    echo "Base directory ${BASE_DIR} does not exist. Exiting." >&2
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "Batch input directory ${INPUT_DIR} does not exist. Exiting." >&2
    exit 1
fi

if [ ! -d "$AF3_DATABASE_DIR" ]; then
    echo "AlphaFold3 databases directory ${AF3_DATABASE_DIR} does not exist. Exiting." >&2
    exit 1
fi

if [ ! -d "$ABCFOLD_PATH" ]; then
    echo "ABCFold directory ${ABCFOLD_PATH} does not exist. Please update ABCFOLD_PATH. Exiting." >&2
    exit 1
fi

# Count JSON files
JSON_COUNT=$(ls -1 "$INPUT_DIR"/*.json 2>/dev/null | wc -l)
if [ "$JSON_COUNT" -eq 0 ]; then
    echo "No JSON files found in ${INPUT_DIR}. Exiting." >&2
    exit 1
fi

echo "=========================================="
echo "MMseqs2 Pipeline for AlphaFold3"
echo "=========================================="
echo "Processing ${JSON_COUNT} JSON files from batch: ${BATCH_DIR}"
echo "Mode: ${MODE}"
echo "Input directory: ${INPUT_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Working directory: ${MMSEQS_WORKDIR}"
echo ""

# =============================================================================
# Activate conda environment
# =============================================================================
echo "Activating ABCFold conda environment..."
eval "$(conda shell.bash hook)"
conda activate abcfold

if [ $? -ne 0 ]; then
    echo "Failed to activate abcfold conda environment. Exiting." >&2
    exit 1
fi

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# =============================================================================
# Set MMseqs2 parameters matching the Python code
# =============================================================================
# These match the parameters in run_local_mmseqs() from add_mmseqs_msa.py
THREADS=$(nproc)
NUM_ITERATIONS=3
DB_LOAD_MODE=2
EVALUE=0.1
MAX_SEQS=10000
SENSITIVITY=8.0
PREFILTER_MODE=0
EXPAND_EVAL="inf"
ALIGN_EVAL=10
DIFF=3000
QSC=-20.0
MAX_ACCEPT=1000000
FILTER=0  # 0 means no filtering, use 1 for filtered mode
NUM_TEMPLATES=20

# Set template flag
if [ "$MODE" == "templates" ]; then
    USE_TEMPLATES="--templates"
else
    USE_TEMPLATES=""
fi

echo "=========================================="
echo "MMseqs2 Parameters (matching ABCFold)"
echo "=========================================="
echo "Threads: ${THREADS}"
echo "Iterations: ${NUM_ITERATIONS}"
echo "Sensitivity: ${SENSITIVITY}"
echo "E-value: ${EVALUE}"
echo "Max sequences: ${MAX_SEQS}"
echo "Use templates: ${MODE}"
if [ "$MODE" == "templates" ]; then
    echo "Number of templates: ${NUM_TEMPLATES}"
fi
echo ""

# =============================================================================
# Process each JSON file
# =============================================================================
echo "=========================================="
echo "Processing JSON files"
echo "=========================================="

PROCESSED=0
FAILED=0

for input_json in "$INPUT_DIR"/*.json; do
    if [ ! -f "$input_json" ]; then
        continue
    fi

    BASENAME=$(basename "$input_json" .json)
    OUTPUT_JSON="$OUTPUT_DIR/${BASENAME}_mmseqs.json"

    # Skip if output already exists
    if [ -f "$OUTPUT_JSON" ]; then
        echo "Skipping ${BASENAME} - output already exists"
        ((PROCESSED++))
        continue
    fi

    echo "----------------------------------------"
    echo "Processing: ${BASENAME}"
    echo "Input: ${input_json}"
    echo "Output: ${OUTPUT_JSON}"

    # Create temporary working directory for this job
    JOB_WORKDIR="$MMSEQS_WORKDIR/${BASENAME}"
    mkdir -p "$JOB_WORKDIR"

    # Run ABCFold's add_mmseqs_msa.py script
    # This script handles MSA generation and adds results to JSON
    python "$ABCFOLD_PATH/abcfold/scripts/add_mmseqs_msa.py" \
        --input_json "$input_json" \
        --output_json "$OUTPUT_JSON" \
        --mmseqs_database "$AF3_DATABASE_DIR" \
        ${USE_TEMPLATES} \
        --num_templates "$NUM_TEMPLATES" 2>&1 | tee "$JOB_WORKDIR/mmseqs.log"

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "✓ Successfully processed ${BASENAME}"
        ((PROCESSED++))

        # Optionally clean up working directory to save space
        # Uncomment the line below if you want to remove intermediate files
        # rm -rf "$JOB_WORKDIR"
    else
        echo "✗ Failed to process ${BASENAME}"
        ((FAILED++))
        echo "Check log: $JOB_WORKDIR/mmseqs.log"
    fi

    echo ""
done

# =============================================================================
# Summary
# =============================================================================
echo "=========================================="
echo "Processing Summary"
echo "=========================================="
echo "Total JSON files: ${JSON_COUNT}"
echo "Successfully processed: ${PROCESSED}"
echo "Failed: ${FAILED}"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo "Working directory: ${MMSEQS_WORKDIR}"
echo ""

if [ $FAILED -gt 0 ]; then
    echo "⚠ Some files failed to process. Check logs in ${MMSEQS_WORKDIR}"
    exit 1
else
    echo "✓ All files processed successfully!"
fi

# =============================================================================
# Optional: Validate output JSONs
# =============================================================================
echo "=========================================="
echo "Validating Output JSONs"
echo "=========================================="

VALID=0
INVALID=0

for output_json in "$OUTPUT_DIR"/*_mmseqs.json; do
    if [ ! -f "$output_json" ]; then
        continue
    fi

    # Check if JSON is valid and contains MSA data
    if python -c "
import json
import sys
try:
    with open('$output_json', 'r') as f:
        data = json.load(f)
    # Check if sequences have MSA data
    has_msa = False
    for seq in data.get('sequences', []):
        if 'protein' in seq:
            if 'unpairedMsa' in seq['protein'] and seq['protein']['unpairedMsa']:
                has_msa = True
                break
    if has_msa:
        sys.exit(0)
    else:
        print('No MSA data found')
        sys.exit(1)
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
" 2>/dev/null; then
        ((VALID++))
    else
        echo "⚠ Invalid or incomplete JSON: $(basename $output_json)"
        ((INVALID++))
    fi
done

echo "Valid JSONs with MSA data: ${VALID}"
echo "Invalid or incomplete JSONs: ${INVALID}"
echo ""

# =============================================================================
# Generate summary report
# =============================================================================
REPORT_FILE="$OUTPUT_DIR/../mmseqs_report_${BATCH_DIR}.txt"
cat > "$REPORT_FILE" << EOF
MMseqs2 Pipeline Report
=======================
Date: $(date)
Batch: ${BATCH_DIR}
HPC Environment: ${HPC_ENV}
Mode: ${MODE}

Directories:
-----------
Input: ${INPUT_DIR}
Output: ${OUTPUT_DIR}
Working: ${MMSEQS_WORKDIR}
Database: ${AF3_DATABASE_DIR}

Processing Summary:
------------------
Total JSON files: ${JSON_COUNT}
Successfully processed: ${PROCESSED}
Failed: ${FAILED}
Valid outputs: ${VALID}
Invalid outputs: ${INVALID}

MMseqs2 Parameters:
------------------
Threads: ${THREADS}
Iterations: ${NUM_ITERATIONS}
Sensitivity: ${SENSITIVITY}
E-value: ${EVALUE}
Max sequences: ${MAX_SEQS}
Templates: ${MODE}
$([ "$MODE" == "templates" ] && echo "Number of templates: ${NUM_TEMPLATES}")

Output Files:
------------
$(ls -lh "$OUTPUT_DIR"/*_mmseqs.json 2>/dev/null | awk '{print $9, $5}' || echo "No output files found")
EOF

echo "Report saved to: ${REPORT_FILE}"
echo ""
echo "=========================================="
echo "Pipeline Complete"
echo "=========================================="
