"""Deploy an OpenAI-compatible vLLM server on Modal.

Eval/ablation backend per spec §2.6 — throughput-bound (~4.7M token budget), no rate
limit, deterministic (fixed seed). Deploy with:

    poetry run modal deploy scripts/deploy_modal_vllm.py

Then read the printed URL into `MODAL_VLLM_BASE_URL` in `.env` (append `/v1`).

This provisions a billed GPU container image build + endpoint — confirm with the user
before running `modal deploy` for real; this file only needs to exist to be reviewed.
"""

import modal

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
GPU = "L4"
MINUTES = 60

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.6.3",
        "huggingface_hub[hf_transfer]==0.25.2",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("codeatlas-vllm", image=vllm_image)


@app.function(
    gpu=GPU,
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    min_containers=0,
    max_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    cmd = (
        f"vllm serve {MODEL_NAME} "
        "--host 0.0.0.0 --port 8000 "
        "--quantization awq "
        "--max-model-len 8192 "
        "--seed 0"
    )
    subprocess.Popen(cmd, shell=True)
