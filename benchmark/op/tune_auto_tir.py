import numpy as np
import tvm
from tvm import meta_schedule as ms
from tvm import relax
from tvm.meta_schedule.default_config import _ARM_MicroKernel
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.relax.testing import relay_translator

from benchmark import utils

WORKLOADS = ["C2D-1", "C2D-2", "GMM-1024", "GMM-2048", "GMM-4096"]
ARGS = utils.parse_args(WORKLOADS, default_trials=1000)
configs = utils.load_config()


if __name__ == "__main__":
    for workload in ARGS.workload:
        config = configs.get(workload)
        assert config is not None
        print(f"Running {workload}")

        if "GMM" in workload:
            shape, weight_shape, units = config
            inputs = {"data": tvm.nd.array(np.random.uniform(-128, 127, shape).astype("int8"))}
            func, params = utils.get_qnn_gemm_model(shape, weight_shape, units)
        elif "C2D" in workload:
            kernel_h, kernel_w, pad, stride, dilation, out_channels, shape = config
            inputs = {"data": tvm.nd.array(np.random.uniform(0, 255, shape).astype("uint8"))}
            func, params = utils.get_qnn_conv2d_model(
                shape,
                kernel_h,
                kernel_w,
                pad,
                stride,
                dilation,
                out_channels,
            )

        with tvm.target.Target(ARGS.target):
            relax_mod = relay_translator.from_relay(func, target=ARGS.target)
            relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)

            relax_mod = relax.transform.FuseOps()(relax_mod)
            relax_mod = relax.transform.FuseTIR()(relax_mod)

            executable = ms.tune_relax(
                mod=relax_mod,
                target=ARGS.target,
                config=utils.get_search_config(ARGS.num_trials, ARGS.num_trials),
                work_dir=f"{ARGS.work_dir}/TIR/{workload}",
                builder=ms.builder.LocalBuilder(),
                runner=ARGS.runner,  # type: ignore
                sch_rules=_ARM_MicroKernel.schedule_rules,
                postprocs=utils.postprocs,
            )
            run_module_via_rpc(
                rpc_config=ARGS.runner.rpc_config,
                lib=executable.mod,
                dev_type=ARGS.target.kind.name,
                args=inputs,
                continuation=utils.f_relax_measurement,
            )
