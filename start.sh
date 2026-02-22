#!/bin/bash
set -x # Print commands and their arguments as they are executed

cd ${AGENT_DIR}

eval "$(conda shell.bash hook)" # make conda available to the shell
conda activate agent

# determine hardware available
if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
  HARDWARE=$(nvidia-smi --query-gpu=name --format=csv,noheader \
    | sed 's/^[ \t]*//' \
    | sed 's/[ \t]*$//' \
    | sort \
    | uniq -c \
    | sed 's/^ *\([0-9]*\) *\(.*\)$/\1 \2/' \
    | paste -sd ', ' -)
else
  HARDWARE="a CPU"
fi
export HARDWARE

# check that we can use the GPU in PyTorch
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'WARNING: No GPU')"

# convert $TIME_LIMIT_SECS to more readable format for prompt
format_time() {
  local time_in_sec=$1
  local hours=$((time_in_sec / 3600))
  local minutes=$(((time_in_sec % 3600) / 60))
  local seconds=$((time_in_sec % 60))
  echo "${hours}hrs ${minutes}mins ${seconds}secs"
}
export TIME_LIMIT=$(format_time $TIME_LIMIT_SECS)
export MAX_ITERATIONS=${MAX_ITERATIONS:-50}
export PER_VARIANT_TIMEOUT=${PER_VARIANT_TIMEOUT:-600}

# build full instructions: general instructions + additional notes + competition description
cp /home/instructions.txt ${AGENT_DIR}/full_instructions.txt
sed -i 's|/home/||g' ${AGENT_DIR}/full_instructions.txt

echo "" >> ${AGENT_DIR}/full_instructions.txt
envsubst < ${AGENT_DIR}/additional_notes.txt >> ${AGENT_DIR}/full_instructions.txt

printf "\nCOMPETITION INSTRUCTIONS\n------\n\n" >> ${AGENT_DIR}/full_instructions.txt
cat /home/data/description.md >> ${AGENT_DIR}/full_instructions.txt

# run with timeout
timeout $TIME_LIMIT_SECS python ${AGENT_DIR}/main.py \
  --data-dir /home/data \
  --submission-dir ${SUBMISSION_DIR} \
  --code-dir ${CODE_DIR} \
  --logs-dir ${LOGS_DIR} \
  --instructions "${AGENT_DIR}/full_instructions.txt" \
  "$@" # forward the bash arguments to main.py

EXIT_CODE=$?
if [ $EXIT_CODE -eq 124 ]; then
  echo "Timed out after $TIME_LIMIT"
fi

# validate the final submission if it exists
if [ -f "${SUBMISSION_DIR}/submission.csv" ]; then
  echo "Validating final submission..."
  bash /home/validate_submission.sh "${SUBMISSION_DIR}/submission.csv"
else
  echo "WARNING: No submission.csv produced!"
fi
