import pickle
import os
import sys
import torch
import tvm
from tvm import relay, autotvm
from tvm import meta_schedule as ms
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.contrib.graph_executor as runtime


import logging
import tempfile
import pytest
import numpy as np
from typing import Tuple, List

import tvm
from tvm import relay
from tvm.ir import IRModule
from tvm.runtime.ndarray import cpu, cuda
from tvm.target.target import Target
from tvm.contrib import graph_executor
from tvm.contrib.debugger import debug_executor
from tvm.meta_schedule.database import PyDatabase, Workload, TuningRecord, JSONDatabase
from tvm.meta_schedule.tune import (
    tune_relay,
    tune_extracted_tasks,
    tune_tir,
    TuneConfig,
)
from tvm.meta_schedule import ApplyHistoryBest, extract_task_from_relay
from tvm.meta_schedule import schedule_rule as M
from tvm.meta_schedule import postproc
from tvm.meta_schedule.utils import derived_object
from tvm.meta_schedule.testing.schedule_rule import (
    multi_level_tiling_memhammer,
    multi_level_tiling_memhammer_tensor_core,
)
from tvm.script import tir as T
from tvm import tir

import pickle
import tir_tensor_intrin
# from tvm.meta_schedule.testing import tir_tensor_intrin

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)

with open("models/resnet50.json", "r") as fi:
    mod = tvm.ir.load_json(fi.read())
with open("models/resnet50.params", "rb") as fi:
    params = relay.load_param_dict(fi.read())

print('RelayModule')
print(mod.astext(False))
target = tvm.target.Target("nvidia/geforce-rtx-3070")

def cuda_build(mod, target, _params):
    from tvm.driver import build as tvm_build

    with tvm.transform.PassContext(
        config={
            "tir.predicate_opt": True,
        }
    ):
        return tvm_build(mod, target=target)

builder = ms.builder.LocalBuilder(
    f_build=cuda_build
)

def build_relay(database):
    with ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True},
        ):
            return tvm.relay.build(mod, target=target, params=params)

def check_params_tensorcore_compatible(prim_func):
    params = prim_func.params
    buffer_map = prim_func.buffer_map
    buffers = [buffer_map[param] for param in params[:2]]
    for buffer in buffers:
        if buffer.shape[-1] % 16 != 0 or buffer.shape[-2] % 16 != 0:
            return False
    return True

def check_params_conv2d_tensorcore_compatible(prim_func):
    params = prim_func.params
    buffer_map = prim_func.buffer_map
    buffers = [buffer_map[param] for param in params[:2]]
    X = buffers[0]
    Weight = buffers[1]
    N,H,W,C = X.shape
    O,S,C,K = Weight.shape
    if (N*H*W) % 16 == 0 and (C*K*S) % 16 == 0 and O % 16 == 0:
        return True
    return False


def should_use_memhammer(task):
    mod = task.dispatched[0]
    global_var = mod.get_global_vars()[0]
    task_name = global_var.name_hint
    if "dense" in task_name or "batch_matmul" in task_name:
        prim_func = mod[global_var]
        return check_params_tensorcore_compatible(prim_func)
    if "conv" in task_name:
        prim_func = mod[global_var]
        return check_params_conv2d_tensorcore_compatible(prim_func)


def tune():
    # extract workloads from relay program
    print("Extract tasks...")
    # print(mod.astext(False))

    tasks = extract_task_from_relay(mod, target=target, params=params)

    #pickle.dump(tasks, open('task.pkl', 'wb'))
    #tasks = pickle.load(open("task.pkl", "rb"))

    # run tuning tasks
    print("#Tuning...")
    memhammer_tasks = []
    other_tasks = []
    for i, tsk in enumerate(tasks):
        print('# task {}'.format(i))
        print(tsk.dispatched[0].script())
        if should_use_memhammer(tsk):
            memhammer_tasks.append(tsk)
        else:
            other_tasks.append(tsk)
    print('#----------')

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

    def sch_rules_tensor_core_memhammer():
        return [
            multi_level_tiling_memhammer_tensor_core(target=target),
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

    def postprocs_no_tensor_core():
        return [
            postproc.RewriteCooperativeFetch(),
            postproc.RewriteUnboundBlock(),
            postproc.RewriteParallelVectorizeUnroll(),
            postproc.RewriteReductionBlock(),
            postproc.VerifyGPUCode(),
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
        max_trials_global=2000,
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

    with tempfile.TemporaryDirectory() as work_dir:
        database = JSONDatabase("db/workload.json", "db/record.json")
        # print(memhammer_tasks[4].dispatched[0].script())
        tune_extracted_tasks(
            # error: 2,4,6
            memhammer_tasks,
            config=search_config,
            sch_rules=sch_rules_tensor_core,
            postprocs=postprocs_tensor_core,
            work_dir=work_dir,
            database=database,
            builder=builder,
        )

        tune_extracted_tasks(
            other_tasks,
            config=search_config,
            # use default CUDA rules
            work_dir=work_dir,
            database=database,
            builder=builder,
        )

        return build_relay(database)


def evaluate(lib):
    dev = tvm.device(str(target), 0)
    module = runtime.GraphModule(lib["default"](dev))

    # debug_module = debug_executor.GraphModuleDebug(lib['debug_create']('default', dev), [dev], lib.get_graph_json(), "dump")
    # module = debug_module

    batch_size = 16
    torch.manual_seed(1001)
    inputs = (torch.rand(batch_size,3,224,224, dtype=torch.float32),)
    module.set_input("input0", inputs[0])
    module.run()

    print("Evaluate inference time cost...")
    print(module.benchmark(dev, number=1, repeat=50))




lib = tune()
evaluate(lib)
