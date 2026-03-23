#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d-%H%M%S)}
RUN_ROOT=${RUN_ROOT:-${SCRIPT_DIR}/runs/${RUN_TAG}}

MODE=${MODE:-both}

SGLANG_REPO=${SGLANG_REPO:-/home/mxl/sglang}
SGLANG_PYTHON=${SGLANG_PYTHON:-${SGLANG_REPO}/python/.venv/bin/python}
SGLANG_ROUTER_MODULE=${SGLANG_ROUTER_MODULE:-sglang_router.launch_router}
SGLANG_BENCH_MODULE=${SGLANG_BENCH_MODULE:-sglang.bench_serving}

MODEL_PATH=${MODEL_PATH:-/mnt/data/models/Qwen3-8B}
TENT_CONF=${TENT_CONF:-/home/mxl/tent-sglang-pd.json}

BIND_HOST=${BIND_HOST:-127.0.0.1}
PREFILL_PORT=${PREFILL_PORT:-31000}
DECODE_PORT=${DECODE_PORT:-31001}
ROUTER_PORT=${ROUTER_PORT:-31080}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-18998}
PD_IB_DEVICE=${PD_IB_DEVICE:-ibp12s0}
PREFILL_IB_DEVICE=${PREFILL_IB_DEVICE:-${PD_IB_DEVICE}}
DECODE_IB_DEVICE=${DECODE_IB_DEVICE:-${PD_IB_DEVICE}}

PREFILL_GPU=${PREFILL_GPU:-0}
DECODE_GPU=${DECODE_GPU:-1}

READY_TIMEOUT=${READY_TIMEOUT:-600}
STARTUP_GRACE_SEC=${STARTUP_GRACE_SEC:-2}

COMMON_SERVER_ARGS=${COMMON_SERVER_ARGS:---mem-fraction-static 0.6 --tp-size 1 --disable-cuda-graph}
PREFILL_SERVER_ARGS=${PREFILL_SERVER_ARGS:-}
DECODE_SERVER_ARGS=${DECODE_SERVER_ARGS:-}
ROUTER_ARGS=${ROUTER_ARGS:---mini-lb}

COMMON_SERVER_ENV=${COMMON_SERVER_ENV:-SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=60}
ROUTER_ENV=${ROUTER_ENV:-}
BENCH_ENV=${BENCH_ENV:-}
CLASSIC_ENV=${CLASSIC_ENV:-}
TENT_ENV=${TENT_ENV:-}

BENCH_BACKEND=${BENCH_BACKEND:-sglang-oai-chat}
BENCH_DATASET_NAME=${BENCH_DATASET_NAME:-random}
BENCH_DATASET_PATH=${BENCH_DATASET_PATH:-}
BENCH_MODEL_NAME=${BENCH_MODEL_NAME:-}
NUM_PROMPTS=${NUM_PROMPTS:-100}
INPUT_LENS=${INPUT_LENS:-1024 4096}
OUTPUT_LENS=${OUTPUT_LENS:-128}
REQUEST_RATES=${REQUEST_RATES:-inf}
MAX_CONCURRENCIES=${MAX_CONCURRENCIES:-1 4 8}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO:-0.0}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-3}
BENCH_SEED=${BENCH_SEED:-42}
FLUSH_CACHE=${FLUSH_CACHE:-1}
BENCH_ARGS=${BENCH_ARGS:-}

PREFILL_TENT_METRICS_PORT=${PREFILL_TENT_METRICS_PORT:-}
DECODE_TENT_METRICS_PORT=${DECODE_TENT_METRICS_PORT:-}

KEEP_STACK_ON_EXIT=${KEEP_STACK_ON_EXIT:-0}

CURRENT_BACKEND=""
CURRENT_BACKEND_DIR=""

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

usage() {
  cat <<'EOF'
Run a same-host 1P1D SGLang PD benchmark and compare classic vs TENT.

The script is configured through environment variables instead of CLI flags.

Common usage:
  MODE=both \
  MODEL_PATH=/mnt/data/models/Qwen3-8B \
  COMMON_SERVER_ARGS="--mem-fraction-static 0.7 --tp-size 1" \
  INPUT_LENS="1024 4096 8192" \
  OUTPUT_LENS="128" \
  MAX_CONCURRENCIES="1 4 8" \
  NUM_PROMPTS=50 \
  ./run_pd_classic_vs_tent.sh

Important env vars:
  MODE                       classic | tent | both
  SGLANG_REPO                SGLang repo root
  SGLANG_PYTHON              Python executable used for server and benchmark
  MODEL_PATH                 Model path
  TENT_CONF                  TENT config file path
  COMMON_SERVER_ARGS         Extra args shared by prefill/decode
  PD_IB_DEVICE               Shared disaggregation NIC used by both sides
  PREFILL_IB_DEVICE          Optional override for prefill NIC
  DECODE_IB_DEVICE           Optional override for decode NIC
  PREFILL_SERVER_ARGS        Extra args only for prefill
  DECODE_SERVER_ARGS         Extra args only for decode
  ROUTER_ARGS                Extra args for sglang_router.launch_router
  BENCH_DATASET_NAME         Benchmark dataset name, default: random
  BENCH_DATASET_PATH         Optional dataset path. For random mode, a local fallback file is auto-generated.
  INPUT_LENS                 Space-separated input lengths
  OUTPUT_LENS                Space-separated output lengths
  REQUEST_RATES              Space-separated request rates, e.g. "inf 2 4"
  MAX_CONCURRENCIES          Space-separated max concurrency values
  NUM_PROMPTS                Number of prompts per case
  PREFILL_TENT_METRICS_PORT  Optional metrics HTTP port for prefill TENT
  DECODE_TENT_METRICS_PORT   Optional metrics HTTP port for decode TENT
  KEEP_STACK_ON_EXIT         1 to keep processes alive after the script exits

Outputs:
  ${RUN_ROOT}/classic
  ${RUN_ROOT}/tent
  ${RUN_ROOT}/all_results.csv
  ${RUN_ROOT}/comparison.csv
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

require_file() {
  [[ -f "$1" ]] || die "file not found: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "directory not found: $1"
}

append_shell_words() {
  local array_name=$1
  shift
  local shell_words="$*"
  if [[ -n "${shell_words}" ]]; then
    eval "${array_name}+=( ${shell_words} )"
  fi
}

apply_env_exports() {
  local env_spec=$1
  if [[ -n "${env_spec}" ]]; then
    eval "export ${env_spec}"
  fi
}

sanitize_token() {
  local token=$1
  token=${token//\//_}
  token=${token//:/_}
  token=${token// /_}
  token=${token//./p}
  echo "${token}"
}

cleanup_pid_file() {
  local pid_file=$1
  if [[ ! -f "${pid_file}" ]]; then
    return 0
  fi

  local pgid
  pgid=$(<"${pid_file}")
  if [[ -n "${pgid}" ]] && pgrep -g "${pgid}" >/dev/null 2>&1; then
    kill -TERM -- "-${pgid}" >/dev/null 2>&1 || true
    for _ in $(seq 1 20); do
      if ! pgrep -g "${pgid}" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
    if pgrep -g "${pgid}" >/dev/null 2>&1; then
      kill -KILL -- "-${pgid}" >/dev/null 2>&1 || true
    fi
  fi
  rm -f "${pid_file}"
}

cleanup_listen_port() {
  local port=$1
  local pids
  pids=$(lsof -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null | sort -u || true)
  if [[ -z "${pids}" ]]; then
    return 0
  fi

  local pid
  for pid in ${pids}; do
    kill -TERM "${pid}" >/dev/null 2>&1 || true
  done
  sleep 2
  for pid in ${pids}; do
    kill -KILL "${pid}" >/dev/null 2>&1 || true
  done
}

cleanup_backend() {
  local backend_dir=$1
  [[ -d "${backend_dir}" ]] || return 0
  cleanup_pid_file "${backend_dir}/pids/router.pid"
  cleanup_pid_file "${backend_dir}/pids/decode.pid"
  cleanup_pid_file "${backend_dir}/pids/prefill.pid"
  cleanup_listen_port "${ROUTER_PORT}"
  cleanup_listen_port "${DECODE_PORT}"
  cleanup_listen_port "${PREFILL_PORT}"
}

cleanup_all() {
  if [[ "${KEEP_STACK_ON_EXIT}" == "1" ]]; then
    log "KEEP_STACK_ON_EXIT=1, skip cleanup"
    return 0
  fi
  if [[ -n "${CURRENT_BACKEND_DIR}" ]]; then
    cleanup_backend "${CURRENT_BACKEND_DIR}"
  fi
}

print_log_tail() {
  local log_file=$1
  if [[ -f "${log_file}" ]]; then
    log "Last 80 lines of ${log_file}:"
    tail -n 80 "${log_file}" || true
  fi
}

wait_for_http() {
  local name=$1
  local url=$2
  local pid_file=$3
  local log_file=$4
  local timeout_sec=$5

  local start_ts
  start_ts=$(date +%s)

  while true; do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      log "${name} is ready: ${url}"
      return 0
    fi

    local now_ts
    now_ts=$(date +%s)
    if (( now_ts - start_ts >= timeout_sec )); then
      print_log_tail "${log_file}"
      die "${name} did not become ready within ${timeout_sec}s"
    fi

    if [[ -f "${pid_file}" ]]; then
      local pid
      pid=$(<"${pid_file}")
      if [[ -n "${pid}" ]] && ! kill -0 "${pid}" >/dev/null 2>&1; then
        print_log_tail "${log_file}"
        die "${name} exited before becoming ready"
      fi
    fi
    sleep 2
  done
}

prepare_local_random_dataset() {
  local dataset_path=$1
  local num_samples=$2
  if [[ -f "${dataset_path}" ]]; then
    return 0
  fi

  mkdir -p "$(dirname "${dataset_path}")"
  if [[ "${num_samples}" -lt 8 ]]; then
    num_samples=8
  fi

  DATASET_PATH="${dataset_path}" DATASET_SAMPLES="${num_samples}" python3 - <<'PY'
import json
import os

dataset_path = os.environ["DATASET_PATH"]
num_samples = int(os.environ["DATASET_SAMPLES"])

prompts = [
    "Explain why Paris is important in one paragraph with factual details and some historical context.",
    "Write a concise background note about distributed systems, data movement, and network bottlenecks in model serving.",
    "Describe one practical tradeoff between throughput optimization and tail latency optimization in inference infrastructure.",
    "Give a short explanation of why KV cache transfer matters for prefill and decode disaggregation.",
    "Summarize the difference between prefill bottlenecks and decode bottlenecks in large language model serving.",
    "List a few reasons why routing and transport stability matter in a disaggregated inference pipeline.",
    "Explain how request concurrency can affect TTFT and end-to-end latency in model serving.",
    "Describe what engineers usually monitor when comparing two transport backends in production inference."
]

answers = [
    "Paris is the capital of France and an important European center for government, finance, art, and science.",
    "Distributed model serving often separates compute and memory bottlenecks, making transport efficiency a first-order concern.",
    "Throughput-oriented scheduling can raise utilization, but often increases head-of-line blocking and hurts tail latency.",
    "KV cache transfer lets decode reuse prefill results remotely instead of recomputing attention over the entire prompt.",
    "Prefill is dominated by prompt processing compute, while decode is dominated by iterative KV cache access and token generation latency.",
    "Unstable routing or transport paths can inflate tail latency, trigger retries, and reduce effective throughput under load.",
    "Higher concurrency can improve utilization, but it may also increase queueing delay and stretch first-token latency.",
    "Engineers usually compare TTFT, ITL, TPOT, throughput, error rate, and long-tail latency percentiles."
]

records = []
for i in range(num_samples):
    prompt = prompts[i % len(prompts)] + f" Include concrete details for sample {i}."
    answer = answers[i % len(answers)] + f" Reference sample {i} for traceability."
    records.append(
        {
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": answer},
            ]
        }
    )

with open(dataset_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=True, indent=2)
PY
}

start_sglang_server() {
  local backend=$1
  local role=$2
  local gpu=$3
  local port=$4
  local log_file=$5
  local pid_file=$6
  local role_args=$7

  local -a cmd
  cmd=(
    "${SGLANG_PYTHON}" -m sglang.launch_server
    --model-path "${MODEL_PATH}"
    --host "${BIND_HOST}"
    --port "${port}"
    --disaggregation-mode "${role}"
    --disaggregation-transfer-backend mooncake
    --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}"
  )
  local ib_device=""
  if [[ "${role}" == "prefill" ]]; then
    ib_device="${PREFILL_IB_DEVICE}"
  else
    ib_device="${DECODE_IB_DEVICE}"
  fi
  if [[ -n "${ib_device}" ]]; then
    cmd+=(--disaggregation-ib-device "${ib_device}")
  fi
  append_shell_words cmd "${COMMON_SERVER_ARGS}"
  append_shell_words cmd "${role_args}"

  (
    export PYTHONPATH="${SGLANG_REPO}/python"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    apply_env_exports "${COMMON_SERVER_ENV}"
    if [[ "${backend}" == "tent" ]]; then
      export MC_USE_TENT=1
      export MC_TENT_CONF="${TENT_CONF}"
      apply_env_exports "${TENT_ENV}"
    else
      unset MC_USE_TENT MC_USE_TEV1 MC_TENT_CONF
      apply_env_exports "${CLASSIC_ENV}"
    fi
    setsid "${cmd[@]}" >"${log_file}" 2>&1 &
    echo $! >"${pid_file}"
  )

  sleep "${STARTUP_GRACE_SEC}"
  wait_for_http "${backend}/${role}" "http://${BIND_HOST}:${port}/health" "${pid_file}" "${log_file}" "${READY_TIMEOUT}"

  if [[ "${backend}" == "tent" ]]; then
    if grep -q "Loaded tent config from file" "${log_file}" 2>/dev/null; then
      log "Detected TENT config load in ${log_file}"
    else
      log "WARN: TENT startup log not found in ${log_file}; continue with health-checked process"
    fi
  fi
}

start_router() {
  local backend=$1
  local log_file=$2
  local pid_file=$3

  local -a cmd
  cmd=(
    "${SGLANG_PYTHON}" -m "${SGLANG_ROUTER_MODULE}"
    --pd-disaggregation
    --prefill "http://${BIND_HOST}:${PREFILL_PORT}"
    --decode "http://${BIND_HOST}:${DECODE_PORT}"
    --host "${BIND_HOST}"
    --port "${ROUTER_PORT}"
  )
  append_shell_words cmd "${ROUTER_ARGS}"

  (
    export PYTHONPATH="${SGLANG_REPO}/python"
    apply_env_exports "${ROUTER_ENV}"
    setsid "${cmd[@]}" >"${log_file}" 2>&1 &
    echo $! >"${pid_file}"
  )

  sleep "${STARTUP_GRACE_SEC}"
  wait_for_http "${backend}/router" "http://${BIND_HOST}:${ROUTER_PORT}/health" "${pid_file}" "${log_file}" "${READY_TIMEOUT}"
}

snapshot_tent_metrics() {
  local component=$1
  local port=$2
  local output_file=$3

  if [[ -z "${port}" ]]; then
    return 0
  fi
  if curl -fsS "http://${BIND_HOST}:${port}/metrics" >"${output_file}" 2>/dev/null; then
    log "Saved ${component} TENT metrics to ${output_file}"
  else
    log "WARN: failed to fetch ${component} TENT metrics from port ${port}"
    rm -f "${output_file}"
  fi
}

run_benchmark_case() {
  local backend=$1
  local backend_dir=$2
  local input_len=$3
  local output_len=$4
  local request_rate=$5
  local max_concurrency=$6

  local case_id
  case_id="in$(sanitize_token "${input_len}")_out$(sanitize_token "${output_len}")_rate$(sanitize_token "${request_rate}")_conc$(sanitize_token "${max_concurrency}")"
  local bench_log="${backend_dir}/logs/${case_id}.log"
  local bench_jsonl="${backend_dir}/results/${case_id}.jsonl"
  local bench_dataset_path="${BENCH_DATASET_PATH}"
  if [[ "${BENCH_DATASET_NAME}" == "random" && -z "${bench_dataset_path}" ]]; then
    bench_dataset_path="${backend_dir}/datasets/local_sharegpt.json"
    prepare_local_random_dataset "${bench_dataset_path}" "${NUM_PROMPTS}"
  fi

  local -a cmd
  cmd=(
    "${SGLANG_PYTHON}" -m "${SGLANG_BENCH_MODULE}"
    --backend "${BENCH_BACKEND}"
    --base-url "http://${BIND_HOST}:${ROUTER_PORT}"
    --dataset-name "${BENCH_DATASET_NAME}"
    --num-prompts "${NUM_PROMPTS}"
    --random-input-len "${input_len}"
    --random-output-len "${output_len}"
    --random-range-ratio "${RANDOM_RANGE_RATIO}"
    --request-rate "${request_rate}"
    --max-concurrency "${max_concurrency}"
    --seed "${BENCH_SEED}"
    --warmup-requests "${WARMUP_REQUESTS}"
    --output-file "${bench_jsonl}"
    --tag "${backend}:${case_id}"
    --disable-tqdm
  )
  if [[ -n "${bench_dataset_path}" ]]; then
    cmd+=(--dataset-path "${bench_dataset_path}")
  fi

  if [[ -n "${BENCH_MODEL_NAME}" ]]; then
    cmd+=(--model "${BENCH_MODEL_NAME}")
  fi
  if [[ "${FLUSH_CACHE}" == "1" ]]; then
    cmd+=(--flush-cache)
  fi
  append_shell_words cmd "${BENCH_ARGS}"

  log "Running ${backend} benchmark case ${case_id}"
  (
    export PYTHONPATH="${SGLANG_REPO}/python"
    apply_env_exports "${BENCH_ENV}"
    "${cmd[@]}"
  ) >"${bench_log}" 2>&1

  if [[ "${backend}" == "tent" ]]; then
    snapshot_tent_metrics "prefill" "${PREFILL_TENT_METRICS_PORT}" "${backend_dir}/metrics/${case_id}_prefill.prom"
    snapshot_tent_metrics "decode" "${DECODE_TENT_METRICS_PORT}" "${backend_dir}/metrics/${case_id}_decode.prom"
  fi
}

write_manifest() {
  local output_file=$1
  cat >"${output_file}" <<EOF
RUN_TAG=${RUN_TAG}
MODE=${MODE}
SGLANG_REPO=${SGLANG_REPO}
SGLANG_PYTHON=${SGLANG_PYTHON}
MODEL_PATH=${MODEL_PATH}
TENT_CONF=${TENT_CONF}
BIND_HOST=${BIND_HOST}
PREFILL_PORT=${PREFILL_PORT}
DECODE_PORT=${DECODE_PORT}
ROUTER_PORT=${ROUTER_PORT}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT}
PREFILL_GPU=${PREFILL_GPU}
DECODE_GPU=${DECODE_GPU}
COMMON_SERVER_ARGS=${COMMON_SERVER_ARGS}
PD_IB_DEVICE=${PD_IB_DEVICE}
PREFILL_IB_DEVICE=${PREFILL_IB_DEVICE}
DECODE_IB_DEVICE=${DECODE_IB_DEVICE}
PREFILL_SERVER_ARGS=${PREFILL_SERVER_ARGS}
DECODE_SERVER_ARGS=${DECODE_SERVER_ARGS}
ROUTER_ARGS=${ROUTER_ARGS}
COMMON_SERVER_ENV=${COMMON_SERVER_ENV}
ROUTER_ENV=${ROUTER_ENV}
BENCH_ENV=${BENCH_ENV}
CLASSIC_ENV=${CLASSIC_ENV}
TENT_ENV=${TENT_ENV}
BENCH_BACKEND=${BENCH_BACKEND}
BENCH_DATASET_NAME=${BENCH_DATASET_NAME}
BENCH_DATASET_PATH=${BENCH_DATASET_PATH}
BENCH_MODEL_NAME=${BENCH_MODEL_NAME}
NUM_PROMPTS=${NUM_PROMPTS}
INPUT_LENS=${INPUT_LENS}
OUTPUT_LENS=${OUTPUT_LENS}
REQUEST_RATES=${REQUEST_RATES}
MAX_CONCURRENCIES=${MAX_CONCURRENCIES}
RANDOM_RANGE_RATIO=${RANDOM_RANGE_RATIO}
WARMUP_REQUESTS=${WARMUP_REQUESTS}
BENCH_SEED=${BENCH_SEED}
FLUSH_CACHE=${FLUSH_CACHE}
BENCH_ARGS=${BENCH_ARGS}
PREFILL_TENT_METRICS_PORT=${PREFILL_TENT_METRICS_PORT}
DECODE_TENT_METRICS_PORT=${DECODE_TENT_METRICS_PORT}
EOF
}

run_backend_suite() {
  local backend=$1
  local backend_dir="${RUN_ROOT}/${backend}"

  mkdir -p "${backend_dir}/logs" "${backend_dir}/pids" "${backend_dir}/results" "${backend_dir}/metrics"
  write_manifest "${backend_dir}/manifest.env"

  CURRENT_BACKEND="${backend}"
  CURRENT_BACKEND_DIR="${backend_dir}"

  cleanup_backend "${backend_dir}"

  start_sglang_server "${backend}" prefill "${PREFILL_GPU}" "${PREFILL_PORT}" "${backend_dir}/logs/prefill.log" "${backend_dir}/pids/prefill.pid" "${PREFILL_SERVER_ARGS}"
  start_sglang_server "${backend}" decode "${DECODE_GPU}" "${DECODE_PORT}" "${backend_dir}/logs/decode.log" "${backend_dir}/pids/decode.pid" "${DECODE_SERVER_ARGS}"
  start_router "${backend}" "${backend_dir}/logs/router.log" "${backend_dir}/pids/router.pid"

  local input_len
  local output_len
  local request_rate
  local max_concurrency

  for input_len in ${INPUT_LENS}; do
    for output_len in ${OUTPUT_LENS}; do
      for request_rate in ${REQUEST_RATES}; do
        for max_concurrency in ${MAX_CONCURRENCIES}; do
          run_benchmark_case "${backend}" "${backend_dir}" "${input_len}" "${output_len}" "${request_rate}" "${max_concurrency}"
        done
      done
    done
  done

  cleanup_backend "${backend_dir}"
  CURRENT_BACKEND=""
  CURRENT_BACKEND_DIR=""
}

resolve_backends() {
  case "${MODE}" in
    classic)
      echo "classic"
      ;;
    tent)
      echo "tent"
      ;;
    both)
      echo "classic tent"
      ;;
    *)
      die "unsupported MODE=${MODE}, expected classic|tent|both"
      ;;
  esac
}

main() {
  if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
  fi

  require_command curl
  require_command python3
  require_file "${SGLANG_PYTHON}"
  require_dir "${SGLANG_REPO}/python"

  if [[ "${MODE}" == "tent" || "${MODE}" == "both" ]]; then
    require_file "${TENT_CONF}"
  fi

  PYTHONPATH="${SGLANG_REPO}/python" "${SGLANG_PYTHON}" -c "import sglang_router" >/dev/null 2>&1 || die "sglang_router is not importable from ${SGLANG_PYTHON}"

  mkdir -p "${RUN_ROOT}"
  write_manifest "${RUN_ROOT}/manifest.env"

  local backends
  backends=$(resolve_backends)
  for backend in ${backends}; do
    log "=== Running backend ${backend} ==="
    run_backend_suite "${backend}"
  done

  python3 "${SCRIPT_DIR}/summarize_results.py" "${RUN_ROOT}"
  log "All benchmark runs are complete. Results are under ${RUN_ROOT}"
}

trap cleanup_all EXIT INT TERM

main "$@"
