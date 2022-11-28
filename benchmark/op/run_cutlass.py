import argparse
from functools import partial
import os
import subprocess
from typing import Tuple, Union

from benchmark.utils import load_config

CONFIGS = load_config()


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "-w",
        "--workloads",
        nargs="+",
        type=str,
        choices=["C1D", "C2D", "C3D", "GMM-1024", "GMM-4096", "DIL"],
        required=True,
    )
    args.add_argument("-n", "--batch-size", nargs="+", type=int, default=[1, 16])
    args.add_argument("--acc-dtype", type=str, choices=["f16", "f32"], default="f16")
    args.add_argument("--out-dtype", type=str, choices=["f16", "f32"], default="f16")
    args.add_argument("--cutlass-home", type=str)
    args.add_argument("--log-dir", type=str, default="logs/cutlass/")
    parsed = args.parse_args()
    parsed.cutlass_home = parsed.cutlass_home or os.getenv("CUTLASS_HOME")
    assert (
        parsed.cutlass_home
    ), "Please specify 'CUTLASS_HOME', by either setting the environment variable or using --cutlass-home"
    parsed.profiler = f"{parsed.cutlass_home}/build/tools/profiler/cutlass_profiler"
    os.makedirs(parsed.log_dir, exist_ok=True)
    return parsed


ARGS = parse_args()


def _run_cutlass(instruction: str, workload: str):
    print("Running:", workload)
    logs = subprocess.check_output(instruction, shell=True)
    logs = logs.decode("utf-8")
    logs = logs.split("\n")
    csv_index = logs.index("CSV Results:")

    csv_file = os.path.join(ARGS.log_dir, f"{workload}.csv")
    with open(csv_file, "w") as f:
        f.write("\n".join(logs[csv_index + 2 :]))

    max_gflops = max(
        [float(log.split(",")[-1]) for log in logs[csv_index + 3 :] if log]
    )
    print(f"{workload}: {max_gflops} GFLOPS")
    print(f"Full results have been written to {csv_file}")


def _run_gemm(
    workload: str,
    b: int,
    n: int,
    m: int,
    k: int,
    acc_dtype: str,
    out_dtype: str,
):
    _run_cutlass(
        f"{ARGS.profiler} --operation=gemm"
        f" --batch_count={b} --n={n} --m={m} --k={k}"
        f" --A=f16:row --B=f16:column --C={out_dtype}"
        f" --accumulator-type={acc_dtype}",
        workload=f"{workload}-{b}-{acc_dtype}-{out_dtype}",
    )


def _run_conv(
    workload: str,
    n: int,
    d: int,
    h: int,
    w: int,
    ci: int,
    co: int,
    kernel_size: Union[int, Tuple[int]],
    stride: Union[int, Tuple[int]],
    padding: Union[int, Tuple[int]],
    dilation: Union[int, Tuple[int]],
    acc_dtype: str,
    out_dtype: str,
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 2
    if isinstance(stride, int):
        stride = (stride,) * 2
    if isinstance(padding, int):
        padding = (padding,) * 2
    if isinstance(dilation, int):
        dilation = (dilation,) * 2
    operation = "Conv2d" if d == 0 else "Conv3d"
    _run_cutlass(
        f"{ARGS.profiler} --operation={operation} --Activation=f16:nhwc --Filter=f16:nhwc"
        f" --n={n} --h={h} --w={w} --c={ci} --k={co} {f'--d={d}' if d != 0 else ''}"
        f" --r={kernel_size[0]} --s={kernel_size[0]} --pad_h={padding[0]} --pad_w={padding[1]}"
        f" --stride_h={stride[0]} --stride_w={stride[1]}"
        f" --dilation_h={dilation[0]} --dilation_w={dilation[1]}"
        f" --accumulator-type={acc_dtype} --Output={out_dtype}",
        workload=f"{workload}-{n}-{acc_dtype}-{out_dtype}",
    )


def C1D(
    batch: int,
    acc_dtype: str,
    out_dtype: str,
):
    _, l, ci, co, kernel, stride, padding = CONFIGS["C1D"]
    return _run_conv(
        "C1D",
        batch,
        0,  # d
        1,
        l,
        ci,
        co,
        (1, kernel),
        (1, stride),
        (0, padding),
        (1, 1),
        acc_dtype,
        out_dtype,
    )


def C2D(
    batch: int,
    acc_dtype: str,
    out_dtype: str,
):
    _, h, w, ci, co, kernel, stride, padding = CONFIGS["C2D"]
    return _run_conv(
        "C2D",
        batch,
        0,  # d
        h,
        w,
        ci,
        co,
        kernel,
        stride,
        padding,
        1,  # dilation
        acc_dtype,
        out_dtype,
    )


def C3D(
    batch: int,
    acc_dtype: str,
    out_dtype: str,
):
    _, d, h, w, ci, co, kernel, stride, padding = CONFIGS["C3D"]
    return _run_conv(
        "C3D",
        batch,
        d,
        h,
        w,
        ci,
        co,
        kernel,
        stride,
        padding,
        1,  # dilation
        acc_dtype,
        out_dtype,
    )


def DIL(batch: int, acc_dtype: str, out_dtype: str):
    _, h, w, ci, co, kernel, stride, padding, dilation = CONFIGS["DIL"]
    return _run_conv(
        "DIL",
        batch,
        0,  # d
        h,
        w,
        ci,
        co,
        kernel,
        stride,
        padding,
        dilation,  # dilation
        acc_dtype,
        out_dtype,
    )


def GMM(
    workload: str,
    batch: int,
    n: int,
    m: int,
    k: int,
    acc_dtype: str,
    out_dtype: str,
):
    return _run_gemm(
        workload,
        batch,
        n,
        m,
        k,
        acc_dtype,
        out_dtype,
    )


WORKLOADS = {
    "C1D": C1D,
    "C2D": C2D,
    "C3D": C3D,
    "GMM-1024": partial(GMM, workload="GMM-1024", n=1024, m=1024, k=1024),
    "GMM-4096": partial(GMM, workload="GMM-4096", n=4096, m=4096, k=4096),
    "DIL": DIL,
}


def main():
    print(f"Running benchmarks for {ARGS.workloads}")
    print(f"Batch size: {ARGS.batch_size}")
    print(f"Accumulator type: {ARGS.acc_dtype}")
    print(f"Output type: {ARGS.out_dtype}")

    for workload in ARGS.workloads:
        for batch_size in ARGS.batch_size:
            WORKLOADS.get(workload)(
                batch=batch_size,
                acc_dtype=ARGS.acc_dtype,
                out_dtype=ARGS.out_dtype,
            )


if __name__ == "__main__":
    main()
