# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from typing import Dict
import numpy as np  # type: ignore

import tvm
from tvm import relay, relax, runtime, transform
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.target.target import Target

from tvm.meta_schedule import extract_task_from_relax
from tvm.meta_schedule.tune import tune_extracted_tasks
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc

from tvm.relax.transform import MetaScheduleApplyHistoryBest
from tvm.relax.testing import relay_translator
from tvm.relax.vm import build as relax_build


from benchmark.utils import *
from benchmark.end2end.utils import *

# We use fp16-16-32 intrinsic for e2e workloads
from tvm.meta_schedule.testing import tir_tensor_intrin_fp16


WORKLOADS = ["resnet_50", "bert_large", "vit"]
ARGS = parse_args(WORKLOADS)


def apply_opt_before_tuning(
    relay_mod: IRModule, params: Dict[str, runtime.NDArray], target: Target
):
    with transform.PassContext(opt_level=3):
        main_func = relay_mod["main"]
        bind_main_func = relay.build_module.bind_params_by_name(main_func, params)
        relay_mod = IRModule.from_expr(bind_main_func)
        relay_mod = relay.transform.SimplifyInference()(relay_mod)
        relay_mod = relay.transform.EliminateCommonSubexpr()(relay_mod)
        relay_mod = relay.transform.FoldConstant()(relay_mod)
        relay_mod = relay.transform.FoldScaleAxis()(relay_mod)
        relay_mod = relay.transform.SimplifyExpr()(relay_mod)
        relay_mod = relay.transform.CanonicalizeCast()(relay_mod)
        relay_mod = relay.transform.CanonicalizeOps()(relay_mod)
        relay_mod = relay.transform.AlterOpLayout()(relay_mod)
        relay_mod = relay.transform.FoldConstant()(relay_mod)

        relax_mod = relay_translator.from_relay(relay_mod["main"], target=target)
        relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)
        relax_mod = relax.transform.FuseOps()(relax_mod)
        relax_mod = relax.transform.FuseTIR()(relax_mod)
    return relax_mod


def f_measurement(rt_mod: runtime.Module, device: runtime.ndarray.Device, input_data):
    vm = relax.vm.VirtualMachine(exec=rt_mod, device=device)
    evaluator = vm.module.time_evaluator(
        func_name="main",
        dev=device,
        repeat=5,
        min_repeat_ms=500,
    )
    print(evaluator(*(input_data.values())))


def tune(workload, input_shape):
    os.makedirs("benchmark/caches/relay")
    relay_mod, params, (input_name, input_shape, input_dtype) = get_network(
        workload,
        input_shape,
        cache_dir="benchmark/caches/relay",
    )
    relay_mod = convert_conv2d_layout(relay_mod, {"nn.conv2d": ["NHWC", "OHWI"]})
    relay_mod = relay.transform.ToMixedPrecision("float16")(relay_mod)
    relay_mod = rewrite_reshape_gelu(relay_mod)

    # translate the ResNet model from Relay to Relax
    relax_mod = apply_opt_before_tuning(relay_mod, params, target=ARGS.target)
    assert isinstance(relax_mod, tvm.IRModule)
    tasks = extract_task_from_relax(relax_mod, target=ARGS.target, params=params)

    # run tuning tasks
    print("Tuning...")
    memhammer_tasks = []
    other_tasks = []
    for tsk in tasks:
        if should_use_memhammer(tsk):
            print(tsk.task_name, "memhammer")
            memhammer_tasks.append(tsk)
        else:
            print(tsk.task_name, "non-memhammer")
            other_tasks.append(tsk)

    search_config = get_search_config(ARGS.num_trials, 1000)
    database = tune_extracted_tasks(
        memhammer_tasks,
        config=search_config,
        sch_rules=sch_rules_tensor_core,
        postprocs=postprocs_tensor_core,
        work_dir=f"{ARGS.work_dir}/TIR/{workload}-{input_shape}",
        runner=ARGS.runner,
    )

    database = tune_extracted_tasks(
        other_tasks,
        config=search_config,
        # use default CUDA rules
        work_dir=f"{ARGS.work_dir}/TIR/{workload}-{input_shape}",
        database=database,
        runner=ARGS.runner,
    )

    with PassContext(opt_level=3):
        relax_mod = MetaScheduleApplyHistoryBest(database, ARGS.target)(relax_mod)
        executable = relax_build(relax_mod, target=ARGS.target)

    if input_dtype.startswith("float"):
        input_data = {
            input_name: np.random.uniform(size=input_shape).astype(input_dtype)
        }
    else:
        input_data = {
            input_name: np.random.randint(
                low=0, high=10000, size=input_shape, dtype=input_dtype
            )
        }

    if ARGS.use_rpc:
        run_module_via_rpc(
            rpc_config=ARGS.runner.rpc_config,
            lib=executable.mod,
            dev_type=ARGS.target.kind.name,
            args=input_data,
            continuation=f_measurement,
        )
    else:
        dev = tvm.device(ARGS.target.kind.name)
        input_data = {
            key: tvm.runtime.ndarray.array(value, dev)
            for key, value in input_data.items()
        }
        f_measurement(executable.mod, dev, input_data)


if __name__ == "__main__":
    configs = load_config()
    for workload in ARGS.workload:
        shape = configs[workload]
        for batch in ARGS.batch_size:
            shape = [batch] + shape[1:]
            tune(workload, shape)
