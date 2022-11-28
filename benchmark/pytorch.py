import torch
from torch.profiler import profile, record_function, ProfilerActivity
from pytorch_workloads import WORKLOADS

import argparse
import os


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "-w",
        "--workloads",
        nargs="+",
        type=str,
        choices=[
            "C1D",
            "C2D",
            "C3D",
            "GMM-1024",
            "GMM-4096",
            "DIL",
            "DEP",
            "GRP",
            "T2D",
            "CBR",
            "mobilenet_v2",
            "bert_large",
        ],
        required=True,
    )
    args.add_argument("-n", "--batch-size", nargs="+", type=int, default=[1, 16])
    args.add_argument("--log-dir", type=str, default="logs/pytorch/")
    parsed = args.parse_args()
    os.makedirs(parsed.log_dir, exist_ok=True)
    return parsed


ARGS = parse_args()

DEVICE = "cuda"
DTYPE = torch.float16


def _profile(name, model, inputs, warmup=50, repeat=100):
    model = model.to(DEVICE)

    if DEVICE == "cuda":
        model = model.half()
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        inputs = [x.to(DEVICE) for x in inputs]
    else:
        inputs = inputs.to(DEVICE)
    model.eval()

    activities = [ProfilerActivity.CPU] if DEVICE == "cpu" else [ProfilerActivity.CUDA]

    for _ in range(warmup):
        model(inputs)
    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("model_inference"):
            for _ in range(repeat):
                model(inputs)

    print("workloads", name)
    print(
        "Total CUDA time %.3f us"
        % (prof.key_averages().total_average().cuda_time_total / repeat)
    )
    print(prof.key_averages().table(sort_by=f"{DEVICE}_time_total", row_limit=10))

    log_file = f"{ARGS.log_dir}/{name}.log"
    with open(log_file, "w") as f:
        f.write(
            "Total CUDA time %.3f us"
            % (prof.key_averages().total_average().cuda_time_total / repeat)
        )
        f.write("\n")
        f.write(prof.key_averages().table(sort_by=f"{DEVICE}_time_total", row_limit=10))


if __name__ == "__main__":
    print(f"Running benchmarks for {ARGS.workloads}")
    print(f"Batch size: {ARGS.batch_size}")

    for workload in ARGS.workloads:
        for batch_size in ARGS.batch_size:
            model, inputs = WORKLOADS[workload](batch_size, DTYPE)
            _profile(f"{workload}-{batch_size}", model, inputs)
