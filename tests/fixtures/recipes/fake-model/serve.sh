#!/bin/bash
# Test recipe: the sserve job contract (listen on $VLLM_PORT, require the key in $SSERVE_KEY_FILE),
# served by the fake vLLM. Per-job behaviour comes from $SSERVE_RUN_DIR/ctl.json.
set -eo pipefail
[ -f "$SSERVE_RUN_DIR/ctl.json" ] || echo '{}' > "$SSERVE_RUN_DIR/ctl.json"
[ -n "${FAKE_CRASH:-}" ] && { echo "fake vLLM: CUDA out of memory during graph capture" >&2; exit 1; }
exec "$FAKE_PY" -I "$FAKE_VLLM" --port "$VLLM_PORT" --model "$SSERVE_SERVED_MODEL" \
    --key-file "$SSERVE_KEY_FILE" --ctl "$SSERVE_RUN_DIR/ctl.json" --name "job$SLURM_JOB_ID" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" --load-s "${FAKE_LOAD_S:-0}"
