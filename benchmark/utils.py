import argparse
import logging
import os
from functools import partial

import numpy as np
import tvm
from tvm import meta_schedule as ms
from tvm import relax, relay, runtime
from tvm.meta_schedule import postproc as M
from tvm.meta_schedule.tune import TuneConfig
from tvm.target import Target

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("tvm.meta_schedule").setLevel(logging.INFO)


def parse_args(workload_candidates, default_trials=20000):
    args = argparse.ArgumentParser()
    args.add_argument(
        "-w",
        "--workload",
        nargs="+",
        type=str,
        choices=workload_candidates,
        required=True,
    )
    args.add_argument("-t", "--target", type=str)
    args.add_argument("-n", "--num-trials", type=int, default=default_trials)
    args.add_argument("--work-dir", type=str)
    args.add_argument("--rpc-host", type=str)
    args.add_argument("--rpc-port", type=int)
    args.add_argument("--rpc-key", type=str)
    args.add_argument("--workers", type=int)
    args.add_argument("--alloc-repeat", type=int, default=1)
    args.add_argument("--out_dtype", type=str, default="float16")

    parsed = args.parse_args()
    parsed.target = (
        parsed.target
        or os.environ.get("TVM_TARGET")
        or "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod -num-cores 32"
    )
    parsed.target = Target(parsed.target)
    parsed.work_dir = parsed.work_dir or f"logs/"
    parsed.rpc_host = parsed.rpc_host or os.environ.get("TVM_RPC_HOST")
    parsed.rpc_port = parsed.rpc_port or int(os.environ.get("TVM_RPC_PORT"))
    parsed.rpc_key = parsed.rpc_key or os.environ.get("TVM_RPC_KEY")
    assert (
        parsed.rpc_host and parsed.rpc_port and parsed.rpc_key
    ), "RPC host, port, and key must be provided"
    rpc_config = ms.runner.RPCConfig(
        tracker_host=parsed.rpc_host,
        tracker_port=parsed.rpc_port,
        tracker_key=parsed.rpc_key,
        session_timeout_sec=60,
    )
    workers = parsed.workers or rpc_config.count_num_servers(allow_missing=False)
    parsed.runner = partial(ms.runner.RPCRunner, rpc_config=rpc_config, max_workers=workers)
    parsed.runner = parsed.runner(
        evaluator_config=ms.runner.EvaluatorConfig(
            number=3,
            repeat=1,
            min_repeat_ms=100,
            enable_cpu_cache_flush=True,
        )
    )
    return parsed


def load_config():
    return {
        "GMM-1024": [[1024, 1024], [1024, 1024], 1024],
        "GMM-2048": [[2048, 2048], [2048, 2048], 2048],
        "GMM-4096": [[4096, 4096], [4096, 4096], 4096],
        "C2D-1": [7, 7, [3, 3], [2, 2], [1, 1], 64, [1, 224, 224, 8]],
        "C2D-2": [3, 3, [1, 1], [1, 1], [1, 1], 64, [1, 56, 56, 64]],
        "resnet_18": [1, 3, 224, 224],
        "resnet_50": [1, 3, 224, 224],
        "mobilenet_v2": [1, 3, 224, 224],
    }


def get_search_config(n_trails, trails_per_task=2000):
    return TuneConfig(
        num_trials_per_iter=64,
        max_trials_per_task=trails_per_task,
        max_trials_global=n_trails,
        search_strategy_config={
            "population_size": 2048,
            "init_measured_ratio": 0.2,
            "init_min_unmeasured": 50,
            "genetic_num_iters": 3,
            "genetic_mutate_prob": 0.85,
            "genetic_max_fail_count": 10,
            "eps_greedy": 0.05,
        },
    )


def _get_qnn_gemm_params(input_zp, input_sc, kernel_zp, kernel_sc, kernel_h, kernel_w):
    """Get output qnn parameters given input and kernel parameters."""
    input_max = input_sc * (255 - input_zp)
    input_min = -input_sc * input_zp
    kernel_max = kernel_sc * (255 - kernel_zp)
    kernel_min = -kernel_sc * kernel_zp
    output_limits = [
        kernel_max * kernel_h * kernel_w * input_max,
        kernel_min * kernel_h * kernel_w * input_max,
        kernel_min * kernel_h * kernel_w * input_min,
        kernel_max * kernel_h * kernel_w * input_min,
    ]
    output_max = max(output_limits)
    output_min = min(output_limits)
    output_sc = (output_max - output_min) / 255
    output_zp = -int(output_min / output_sc)
    return output_zp, output_sc


def get_qnn_gemm_model(
    shape,
    weight_shape,
    units,
    dtype="int8",
    input_zp=100,
    input_sc=0.5,
    kernel_zp=50,
    kernel_sc=0.03,
    has_bias=False,
):
    output_zp, output_sc = _get_qnn_gemm_params(
        input_zp, input_sc, kernel_zp, kernel_sc, weight_shape[0], weight_shape[1]
    )
    a = relay.var("data", shape=shape, dtype=dtype)
    w = tvm.nd.array(np.random.uniform(-128, 127, weight_shape).astype(dtype))
    weights = relay.const(w, dtype)
    out = relay.qnn.op.dense(
        a,
        weights,
        units=units,
        input_zero_point=relay.const(input_zp, "int32"),
        kernel_zero_point=relay.const(kernel_zp, "int32"),
        input_scale=relay.const(input_sc, "float32"),
        kernel_scale=relay.const(kernel_sc, "float32"),
        out_dtype="int32",
    )
    params = {"w": w}
    if has_bias:
        b = tvm.nd.array(np.random.randint(0, 255, weight_shape[0]).astype("int32"))
        biasc = relay.const(b, "int32")
        out = relay.nn.bias_add(out, biasc)
        params["b"] = b
    out = relay.qnn.op.requantize(
        out,
        relay.const(input_sc * kernel_sc, "float32"),  # input scale
        relay.const(0, "int32"),  # input zero point
        relay.const(output_sc, "float32"),  # output scale
        relay.const(output_zp, "int32"),  # output zero point
        out_dtype=dtype,
    )
    return out, params


def _get_qnn_conv2d_params(input_zp, input_sc, kernel_zp, kernel_sc, kernel_h, kernel_w, channels):
    """Get output qnn parameters given input and kernel parameters."""
    input_max = input_sc * (255 - input_zp)
    input_min = -input_sc * input_zp
    kernel_max = kernel_sc * (255 - kernel_zp)
    kernel_min = -kernel_sc * kernel_zp
    output_limits = [
        kernel_max * kernel_h * kernel_w * channels * input_max,
        kernel_min * kernel_h * kernel_w * channels * input_max,
        kernel_min * kernel_h * kernel_w * channels * input_min,
        kernel_max * kernel_h * kernel_w * channels * input_min,
    ]
    output_max = max(output_limits)
    output_min = min(output_limits)
    output_sc = (output_max - output_min) / 255
    output_zp = -int(output_min / output_sc)
    return output_zp, output_sc


def get_qnn_conv2d_model(
    shape,
    kernel_h,
    kernel_w,
    padding,
    strides,
    dilation,
    channels,
    groups=1,
    dtype="uint8",
    input_zp=100,
    input_sc=0.5,
    kernel_zp=50,
    kernel_sc=0.03,
    has_bias=True,
    has_activation=False,
    has_pad=False,
):
    """Return a model and any parameters it may have."""
    output_zp, output_sc = _get_qnn_conv2d_params(
        input_zp, input_sc, kernel_zp, kernel_sc, kernel_h, kernel_w, shape[3]
    )
    a = relay.var("data", shape=shape, dtype=dtype)
    if has_pad:
        p = ((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0))
        a = relay.nn.pad(a, pad_width=p, pad_value=input_zp, pad_mode="constant")
        padding = (0, 0, 0, 0)
    else:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
        shape = (shape[0], shape[1] + padding[0] * 2, shape[2] + padding[1] * 2, shape[3])
    is_depthwise = shape[3] == channels == groups
    weight_format = "HWOI" if is_depthwise else "HWIO"
    if weight_format == "HWIO":
        weight_shape = (kernel_h, kernel_w, shape[3] // groups, channels)
    else:
        weight_shape = (kernel_h, kernel_w, channels, shape[3] // groups)
    w = tvm.nd.array(np.random.uniform(0, 255, weight_shape).astype(dtype))
    weights = relay.const(w, dtype)
    out = relay.qnn.op.conv2d(
        a,
        weights,
        input_zero_point=relay.const(input_zp, "int32"),
        kernel_zero_point=relay.const(kernel_zp, "int32"),
        input_scale=relay.const(input_sc, "float32"),
        kernel_scale=relay.const(kernel_sc, "float32"),
        kernel_size=(kernel_h, kernel_w),
        data_layout="NHWC",
        kernel_layout=weight_format,
        dilation=dilation,
        strides=strides,
        padding=padding,
        groups=groups,
        channels=channels,
        out_dtype="int32",
    )
    params = {"w": w}
    if has_bias:
        bias_shape = weight_shape[2] if is_depthwise else weight_shape[3]
        b = tvm.nd.array(np.random.uniform(-128, 127, bias_shape).astype("int32"))
        biasc = relay.const(b, "int32")
        out = relay.nn.bias_add(out, biasc, axis=3)
        params["b"] = b
    if has_activation:
        out = relay.nn.relu(out)
    req = relay.qnn.op.requantize(
        out,
        relay.const(input_sc * kernel_sc, "float32"),  # input scale
        relay.const(0, "int32"),  # input zero point
        relay.const(output_sc, "float32"),  # output scale
        relay.const(output_zp, "int32"),  # output zero point
        out_dtype=dtype,
    )
    return req, params


def postprocs():
    def code():
        ll_code = open("benchmark/op/kernel.ll", mode="r").read()
        return ll_code

    return [
        M.DisallowDynamicLoop(),
        M.RewriteParallelVectorizeUnroll(),
        M.RewriteReductionBlock(),
        M.RewriteTensorize(),
        M.InjectKernelCode(code()),
    ]


def f_relax_measurement(rt_mod: runtime.Module, device: runtime.ndarray.Device, input_data):
    vm = relax.vm.VirtualMachine(exec=rt_mod, device=device)
    evaluator = vm.module.time_evaluator(
        func_name="main",
        dev=device,
        repeat=5,
        min_repeat_ms=500,
    )
    print(evaluator(input_data["data"]))
