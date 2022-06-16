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
import os
import json
import argparse
import logging
from typing import Dict
import numpy as np  # type: ignore

import tvm
from tvm import relay, relax, runtime, transform, tir
from tvm.ir.module import IRModule
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.relax.testing import relay_translator
from tvm.target.target import Target
from bert_rewrite import rewrite_reshape_gelu
from tvm.meta_schedule import schedule_rule as M
from tvm.meta_schedule import postproc, extract_task_from_relax
from tvm.relax.transform import MetaScheduleApplyHistoryBest
from tvm.ir.transform import PassContext
from tvm.relax.vm import build as relax_build

from tvm.meta_schedule.tune import (
    tune_extracted_tasks,
    TuneConfig,
)
import tir_tensor_intrin


def _parse_args():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--workload",
        type=str,
        required=True,
    )
    args.add_argument(
        "--input-shape",
        type=str,
        required=True,
    )
    args.add_argument(
        "--target",
        type=str,
        required=True,
    )
    args.add_argument(
        "--num-trials",
        type=int,
        required=True,
    )
    args.add_argument(
        "--rpc-host",
        type=str,
        default=None,
    )
    args.add_argument(
        "--rpc-port",
        type=int,
        default=None,
    )
    args.add_argument(
        "--rpc-key",
        type=str,
        default=None,
    )
    args.add_argument(
        "--work-dir",
        type=str,
        required=True,
    )
    args.add_argument(
        "--cache-dir",
        type=str,
        default=None,
    )
    parsed = args.parse_args()
    parsed.target = tvm.target.Target(parsed.target)
    parsed.input_shape = json.loads(parsed.input_shape)
    if parsed.target.attrs.get("mtriple", None) == "aarch64-linux-gnu":
        parsed.alloc_repeat = 3
    else:
        parsed.alloc_repeat = 1
    if parsed.rpc_host and parsed.rpc_port and parsed.rpc_key:
        parsed.rpc_config = ms.runner.RPCConfig(
            tracker_host=parsed.rpc_host,
            tracker_port=parsed.rpc_port,
            tracker_key=parsed.rpc_key,
            session_timeout_sec=180,
        )
        parsed.workers = parsed.rpc_config.count_num_servers(allow_missing=False)
    else:
        # check all rpc configs are None
        assert (
                (parsed.rpc_host is None) and (parsed.rpc_port is None) and (parsed.rpc_key is None)
        ), "Please set all 'rpc_host', 'rpc_port' and 'rpc_key' to use PRC server"
    return parsed


logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)
ARGS = _parse_args()


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


def f_measurement(rt_mod: runtime.Module, device: runtime.ndarray.Device, *input_data):
    vm = relax.vm.VirtualMachine(exec=rt_mod, device=device)
    evaluator = vm.module.time_evaluator(
        func_name="main",
        dev=device,
        repeat=5,
        min_repeat_ms=500,
    )
    print(evaluator(*input_data))


def get_runner():
    runner_config = {
        "evaluator_config": ms.runner.EvaluatorConfig(
            number=3,
            repeat=1,
            min_repeat_ms=100,
            enable_cpu_cache_flush=False,
        ),
        "alloc_repeat": ARGS.alloc_repeat,
    }
    if ARGS.rpc_config:
        runner = ms.runner.RPCRunner(
            rpc_config=ARGS.rpc_config, max_workers=ARGS.workers, **runner_config
        )
    else:
        runner = ms.runner.LocalRunner(**runner_config)

    return runner

def check_params_tensorcore_compatible(prim_func):
    params = prim_func.params
    buffer_map = prim_func.buffer_map
    buffers = [buffer_map[param] for param in params[:2]]
    for buffer in buffers:
        if buffer.shape[-1] % 16 != 0 or buffer.shape[-2] % 16 != 0:
            return False
    return True

def should_use_memhammer(task):
    mod = task.dispatched[0]
    global_var = mod.get_global_vars()[0]
    task_name = global_var.name_hint
    if "dense" in task_name or "batch_matmul" in task_name:
        prim_func = mod[global_var]
        return check_params_tensorcore_compatible(prim_func)


def main():

    with open("models/bert_large.json", "r") as fi:
        relay_mod = tvm.ir.load_json(fi.read())
    with open("models/bert_large.params", "rb") as fi:
        params = relay.load_param_dict(fi.read())
    relay_mod = rewrite_reshape_gelu(relay_mod)

    input_dtype = "int64"

    # translate the ResNet model from Relay to Relax
    relax_mod = apply_opt_before_tuning(relay_mod, params, target=ARGS.target)
    assert isinstance(relax_mod, tvm.IRModule)


    def sch_rules_tensor_core():
        return [
            M.MultiLevelTiling(
                structure="SSSRRSRS",
                tile_binds=["blockIdx.x", "blockIdx.y", "threadIdx.y"],
                use_tensor_core=True,
                max_innermost_factor=4,
                vector_load_lens=[1, 2, 4, 8],
                reuse_read=M.ReuseType(
                    req="must",
                    levels=[4],
                    scope="shared.dyn",
                ),
                reuse_write=M.ReuseType(
                    req="no",
                    levels=[3],
                    scope="shared.dyn",
                ),
            ),
            M.AutoInline(
                into_producer=True,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=False,
                require_injective=False,
                require_ordered=False,
                disallow_op=None,
            ),
            M.AutoInline(
                into_producer=True,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=False,
                require_injective=False,
                require_ordered=False,
                disallow_op=None,
            ),
            M.CrossThreadReduction(thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]),
            M.ParallelizeVectorizeUnroll(
                max_jobs_per_core=-1,  # disable parallelize
                max_vectorize_extent=-1,  # disable vectorize
                unroll_max_steps=[0, 16, 64, 512, 1024],
                unroll_explicit=True,
            ),
        ]


    def postprocs_tensor_core():
        return [
            postproc.RewriteCooperativeFetch(),
            postproc.RewriteUnboundBlock(),
            postproc.RewriteParallelVectorizeUnroll(),
            postproc.RewriteReductionBlock(),
            postproc.RewriteTensorCore(),
            postproc.VerifyGPUCode(),
        ]

    search_config = TuneConfig(
        num_trials_per_iter=64,
        max_trials_per_task=500,
        max_trials_global=ARGS.num_trials,
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

    tasks = extract_task_from_relax(relax_mod, target=ARGS.target, params=params)


    # run tuning tasks
    print("Tuning...")
    memhammer_tasks = []
    other_tasks = []
    for tsk in tasks:
        if "fused_dense_reshape1_add4_reshape2_transpose_reshape3" in tsk.task_name:
            print(tsk.dispatched[0].script())
        if should_use_memhammer(tsk):
            print(tsk.task_name, "memhammer")
            memhammer_tasks.append(tsk)
        else:
            print(tsk.task_name, "non-memhammer")
            other_tasks.append(tsk)

    database = tune_extracted_tasks(
        other_tasks,
        config=search_config,
        # use default CUDA rules
        work_dir=ARGS.work_dir,
        runner=get_runner(),
    )

    database = tune_extracted_tasks(
        memhammer_tasks,
        config=search_config,
        sch_rules=sch_rules_tensor_core,
        postprocs=postprocs_tensor_core,
        work_dir=ARGS.work_dir,
        database=database,
        runner=get_runner(),
    )


    with PassContext(opt_level=3):
        relax_mod = MetaScheduleApplyHistoryBest(database, ARGS.target)(relax_mod)
        executable = relax_build(relax_mod, target=ARGS.target)

    inputs = [
        np.random.randint(100, size=ARGS.input_shape, dtype=input_dtype),
        np.random.randint(100, size=ARGS.input_shape, dtype=input_dtype),
        np.random.randint(100, size=ARGS.input_shape, dtype=input_dtype),
    ]

    if ARGS.rpc_config:
        run_module_via_rpc(
            rpc_config=ARGS.rpc_config,
            lib=executable.mod,
            dev_type=ARGS.target.kind.name,
            args=inputs,
            continuation=f_measurement,
        )
    else:
        dev = tvm.device(ARGS.target.kind.name)
        input_data = [runtime.ndarray.array(arg, dev) for arg in inputs]
        f_measurement(executable.mod, dev, *input_data)


if __name__ == "__main__":
    main()
