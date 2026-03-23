# SGLang PD Classic vs TENT Benchmark

This directory contains a small same-host 1P1D benchmark harness for comparing:

- `classic` Mooncake Transfer Engine
- `tent` backend enabled through `MC_USE_TENT=1`

The harness assumes:

- SGLang repo already exists on the benchmark machine
- model files are already present
- prefill and decode run on the same machine with different GPUs

The current default baseline is intentionally conservative because it was validated on the target server:

- shared NIC: `ibp12s0`
- `--mem-fraction-static 0.6`
- `--disable-cuda-graph`
- `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=60`

## Files

- `run_pd_classic_vs_tent.sh`: main benchmark driver
- `summarize_results.py`: converts JSONL outputs into `all_results.csv` and `comparison.csv`

## What It Collects

For each `(input_len, output_len, request_rate, max_concurrency)` case, it:

1. starts `prefill`
2. starts `decode`
3. starts `sglang_router`
4. runs `python -m sglang.bench_serving`
5. stores server logs and benchmark JSONL output

The benchmark output includes:

- request throughput
- input/output/total token throughput
- mean and tail TTFT
- mean and tail TPOT
- mean and tail ITL
- mean and tail E2E latency

If TENT metrics ports are exposed, the script can also snapshot `/metrics` after each TENT case.

## Typical Usage

Run on the Linux benchmark server:

```bash
cd /path/to/Mooncake/benchmarks/sglang_pd

MODE=both \
SGLANG_REPO=/home/mxl/sglang \
MODEL_PATH=/mnt/data/models/Qwen3-8B \
TENT_CONF=/home/mxl/tent-sglang-pd.json \
INPUT_LENS="1024 4096 8192" \
OUTPUT_LENS="128" \
REQUEST_RATES="inf" \
MAX_CONCURRENCIES="1 4 8" \
NUM_PROMPTS=50 \
./run_pd_classic_vs_tent.sh
```

The run directory is created under:

```text
benchmarks/sglang_pd/runs/<timestamp>/
```

It contains:

```text
classic/
tent/
all_results.csv
comparison.csv
```

## Useful Environment Variables

- `MODE`: `classic`, `tent`, or `both`
- `SGLANG_REPO`: SGLang repo root
- `SGLANG_PYTHON`: Python used to launch SGLang
- `MODEL_PATH`: model path
- `TENT_CONF`: TENT config path
- `PD_IB_DEVICE`: shared RDMA NIC for both prefill and decode
- `PREFILL_IB_DEVICE`, `DECODE_IB_DEVICE`: optional per-role NIC overrides
- `PREFILL_GPU`: GPU index for prefill
- `DECODE_GPU`: GPU index for decode
- `PREFILL_PORT`, `DECODE_PORT`, `ROUTER_PORT`, `BOOTSTRAP_PORT`
- `COMMON_SERVER_ARGS`: args shared by prefill/decode
- `PREFILL_SERVER_ARGS`, `DECODE_SERVER_ARGS`, `ROUTER_ARGS`
- `BENCH_DATASET_NAME`: benchmark dataset, default `random`
- `BENCH_DATASET_PATH`: optional dataset path. If omitted in `random` mode, the script creates a local ShareGPT-like JSON file so it can run fully offline.
- `INPUT_LENS`, `OUTPUT_LENS`, `REQUEST_RATES`, `MAX_CONCURRENCIES`
- `NUM_PROMPTS`, `WARMUP_REQUESTS`, `BENCH_SEED`
- `PREFILL_TENT_METRICS_PORT`, `DECODE_TENT_METRICS_PORT`

## Notes

- The first version focuses on the same-host 1P1D benchmark flow because that matches the currently verified TENT bring-up path.
- The script now starts each service in its own process group and cleans up by process group plus port listeners to avoid leaving `sglang::scheduler` processes behind.
- If you want to keep the stack alive after benchmarking for manual inspection, set `KEEP_STACK_ON_EXIT=1`.
