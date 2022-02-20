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
# pylint: disable=missing-docstring

from typing import List

import tvm
from tvm import meta_schedule as ms
from tvm.ir import IRModule
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.postproc import Postproc
from tvm.meta_schedule.testing import create_te_workload
from tvm.meta_schedule.tune import DefaultCUDA, DefaultLLVM
from tvm.meta_schedule.utils import remove_build_dir
from tvm.target import Target
from tvm.tir import Schedule


RPC_HOST = "192.168.6.66"
RPC_PORT = 4445
RPC_KEY = "jetson-agx-xavier"
TARGET = Target("nvidia/jetson-agx-xavier")
WORKLOAD = "C1D"
POSTPROCS: List[Postproc] = DefaultCUDA._postproc()  # pylint: disable=protected-access

TARGET = tvm.target.Target("nvidia/jetson-agx-xavier")


@tvm.register_func
def tvm_callback_cuda_postproc(code):
    import os

    if not os.path.exists("/tmp/perf"):
        os.mkdir("/tmp/perf")
    with open("/tmp/perf/tir.cu", "w") as f:
        f.write(code)
    return code


def schedule_fn(sch: Schedule):
    # pylint: disable=invalid-name,line-too-long,unused-variable
    # fmt: off
    b0 = sch.get_block(name="PadInput", func_name="main")
    b1 = sch.get_block(name="conv1d_nlc", func_name="main")
    b2 = sch.get_block(name="root", func_name="main")
    b3 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    l4, l5, l6, l7, l8 = sch.get_loops(block=b1)
    v9, v10, v11, v12, v13 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64, decision=[1, 1, 1, 1, 1])
    l14, l15, l16, l17, l18 = sch.split(loop=l4, factors=[v9, v10, v11, v12, v13])
    v19, v20, v21, v22, v23 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64, decision=[4, 1, 8, 4, 1])
    l24, l25, l26, l27, l28 = sch.split(loop=l5, factors=[v19, v20, v21, v22, v23])
    v29, v30, v31, v32, v33 = sch.sample_perfect_tile(loop=l6, n=5, max_innermost_factor=64, decision=[4, 1, 16, 1, 2])
    l34, l35, l36, l37, l38 = sch.split(loop=l6, factors=[v29, v30, v31, v32, v33])
    v39, v40, v41 = sch.sample_perfect_tile(loop=l7, n=3, max_innermost_factor=64, decision=[1, 1, 3])
    l42, l43, l44 = sch.split(loop=l7, factors=[v39, v40, v41])
    v45, v46, v47 = sch.sample_perfect_tile(loop=l8, n=3, max_innermost_factor=64, decision=[4, 8, 2])
    l48, l49, l50 = sch.split(loop=l8, factors=[v45, v46, v47])
    sch.reorder(l14, l24, l34, l15, l25, l35, l16, l26, l36, l42, l48, l43, l49, l17, l27, l37, l44, l50, l18, l28, l38)
    l51 = sch.fuse(l14, l24, l34)
    sch.bind(loop=l51, thread_axis="blockIdx.x")
    l52 = sch.fuse(l15, l25, l35)
    sch.bind(loop=l52, thread_axis="vthread.x")
    l53 = sch.fuse(l16, l26, l36)
    sch.bind(loop=l53, thread_axis="threadIdx.x")

    b54 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="shared")
    sch.compute_at(block=b54, loop=l48, preserve_unit_loops=True)
    l55, l56, l57, l58, l59, l60, l61, l62 = sch.get_loops(block=b54)
    l63 = sch.fuse(l60, l61, l62)
    v64, v65 = sch.sample_perfect_tile(loop=l63, n=2, max_innermost_factor=4, decision=[1040, 1])
    sch.annotate(block_or_loop=b54, ann_key="meta_schedule.cooperative_fetch", ann_val=v65)

    b66 = sch.cache_read(block=b1, read_buffer_index=2, storage_scope="shared")
    sch.compute_at(block=b66, loop=l48, preserve_unit_loops=True)
    l67, l68, l69, l70, l71, l72, l73, l74 = sch.get_loops(block=b66)
    l75 = sch.fuse(l72, l73, l74)
    v76, v77 = sch.sample_perfect_tile(loop=l75, n=2, max_innermost_factor=4, decision=[1536, 1])
    sch.annotate(block_or_loop=b66, ann_key="meta_schedule.cooperative_fetch", ann_val=v77)

    sch.reverse_compute_at(block=b3, loop=l53, preserve_unit_loops=True)
    sch.compute_inline(block=b0)
    # v78 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.2, 0.2, 0.2, 0.2, 0.2], decision=4)
    # sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v78)
    # fmt: on
    return sch


def _make_sch() -> Schedule:
    prim_func = create_te_workload(WORKLOAD, 0)
    prim_func = prim_func.with_attr("global_symbol", "main")
    prim_func = prim_func.with_attr("tir.noalias", True)
    mod = IRModule({"main": prim_func})
    return Schedule(mod, debug_mask="all")


def _apply_postproc(sch: Schedule):
    sch.enter_postproc()
    ctx = TuneContext(target=TARGET)
    for p in POSTPROCS:
        p.initialize_with_tune_context(ctx)
        assert p.apply(sch)


def run_sch(sch: Schedule):
    print(sch.mod.script())
    print(sch.trace)
    print(tvm.lower(sch.mod).script())
    tvm.build(sch.mod, target=TARGET)
    builder = ms.builder.LocalBuilder()
    runner = ms.runner.RPCRunner(
        rpc_config=ms.runner.RPCConfig(
            tracker_host=RPC_HOST,
            tracker_port=RPC_PORT,
            tracker_key=RPC_KEY,
            session_timeout_sec=60,
        ),
        alloc_repeat=3,
        max_workers=5,
    )
    (builder_result,) = builder.build(  # pylint: disable=unbalanced-tuple-unpacking
        [ms.builder.BuilderInput(sch.mod, TARGET)]
    )
    if builder_result.error_msg is not None:
        print(builder_result.error_msg)
        return
    try:
        runner_input = ms.runner.RunnerInput(
            builder_result.artifact_path,
            device_type=TARGET.kind.name,
            args_info=ms.arg_info.ArgInfo.from_prim_func(sch.mod["main"]),
        )
        (runner_future,) = runner.run([runner_input])  # pylint: disable=unbalanced-tuple-unpacking
        runner_result = runner_future.result()
        if runner_result.error_msg is not None:
            print(runner_result.error_msg)
        else:
            print([float(x) * 1000.0 for x in runner_result.run_secs])
    finally:
        remove_build_dir(builder_result.artifact_path)


def main():
    sch = schedule_fn(_make_sch())
    _apply_postproc(sch)
    run_sch(sch)


if __name__ == "__main__":
    main()
