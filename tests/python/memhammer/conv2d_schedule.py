import logging
import tempfile

import tvm
import pytest
from tvm.meta_schedule import ReplayTraceConfig, tune_tir
from tvm.meta_schedule.tune_context import TuneContext
from tvm.meta_schedule import schedule_rule, postproc
from tvm.meta_schedule.space_generator import PostOrderApply
from tvm.target.target import Target
from tvm.te.operation import create_prim_func
from tvm.tir import Schedule
from tvm.meta_schedule.testing import te_workload, tir_tensor_intrin
from tvm import te, tir
from tvm.script import tir as T
import numpy as np
import tvm.testing
from tvm.meta_schedule.postproc import (
    RewriteParallelVectorizeUnroll,
    RewriteReductionBlock,
    RewriteTensorCore,
    DisallowDynamicLoop,
    VerifyGPUCode
)

postprocs=[
    RewriteParallelVectorizeUnroll(),
    RewriteReductionBlock(),
    RewriteTensorCore(),
    DisallowDynamicLoop(),
    # VerifyGPUCode()
]

@tvm.script.ir_module
class ImplicitGemm:
    @T.prim_func
    def main(placeholder_1: T.Buffer[(3, 3, 512, 512), "float16"],
             placeholder_2: T.Buffer[(32, 14, 14, 512), "float16"], Conv2dOutput: T.Buffer[(32, 7, 7, 512),
                                                                                           "float32"]) -> \
            None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        PaddedInput = T.alloc_buffer([32, 16, 16, 512], dtype="float16")
        ActivationTransformed = T.alloc_buffer([1568, 4608], dtype="float16")
        WeightTransformed = T.alloc_buffer([4608, 512], dtype="float16")
        OutputTransformed = T.alloc_buffer([1568, 512], dtype="float32")
        for i0, i1, i2, i3 in T.grid(32, 16, 16, 512):
            with T.block("PaddedInput"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(placeholder_2[i0_1, i1_1 - 1, i2_1 - 1, i3_1])
                T.writes(PaddedInput[i0_1, i1_1, i2_1, i3_1])
                PaddedInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i1_1 and i1_1 < 15 and 1 <= i2_1 and i2_1 <
                                                                     15, placeholder_2[i0_1, i1_1 - 1, i2_1 - 1,
                                                                                       i3_1], T.float32(0),
                                                                     dtype="float16")

        for i in T.serial(0, 1568):
            for k in T.serial(0, 4608):
                with T.block("load_activation"):
                    vi, vk = T.axis.remap("SS",[i,k])
                    ActivationTransformed[vi, vk] = PaddedInput[vi//49, (vi%49//7)*2+vk//1536, vi%7*2+vk%1536//512, vk%512]

        for k in T.serial(0, 4608):
            for j in T.serial(0, 512):
                with T.block("load_weight"):
                    vk, vj = T.axis.remap("SS",[k,j])
                    WeightTransformed[vk, vj] = placeholder_1[vk//1536, vk%1536//512, vk%512, vj]

        for i,j,k in T.grid(1568, 512, 4608):
            with T.block("gemm"):
                vi, vj, vk = T.axis.remap("SSR",[i,j,k])
                with T.init():
                    OutputTransformed[vi, vj] = T.float32(0)
                OutputTransformed[vi, vj] += T.cast(ActivationTransformed[vi, vk], "float32") * \
                                             T.cast(WeightTransformed[vk,vj],"float32")

        for i, j in T.grid(1568, 512):
            with T.block("epilogue"):
                vi, vj = T.axis.remap("SS",[i, j])
                Conv2dOutput[vi // 49, vi %49 //7, vi%7, vj] = OutputTransformed[vi, vj]

sch = tir.Schedule(ImplicitGemm["main"])
b0 = sch.get_block(name="PaddedInput", func_name="main")
b1 = sch.get_block(name="load_activation", func_name="main")
b2 = sch.get_block(name="load_weight", func_name="main")
b3 = sch.get_block(name="gemm", func_name="main")
b4 = sch.get_block(name="epilogue", func_name="main")
l5, l6, l7 = sch.get_loops(block=b3)
l8, l9 = sch.split(loop=l5, factors=[98, 16])
l10, l11 = sch.split(loop=l6, factors=[32, 16])
l12, l13 = sch.split(loop=l7, factors=[288, 16])
l14, l15, l16, l17, l18, l19 = sch.get_loops(block=b3)
sch.reorder(l16, l18, l9, l11, l13)
b20 = sch.blockize(loop=l9)
sch.annotate(block_or_loop=b3, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync")
sch.annotate(block_or_loop=b20, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill")
b21 = sch.get_block(name="root", func_name="main")
sch.annotate(block_or_loop=b21, ann_key="meta_schedule.tensor_core_enabled", ann_val="1")
b22 = sch.get_block(name="root", func_name="main")
sch.annotate(block_or_loop=b22, ann_key="warp_execution", ann_val=1)
l23, l24, l25 = sch.get_loops(block=b20)
v26, v27, v28, v29, v30 = sch.sample_perfect_tile(loop=l23, n=5, max_innermost_factor=64, decision=[7, 1, 14, 1, 1])
l31, l32, l33, l34, l35 = sch.split(loop=l23, factors=[v26, v27, v28, v29, v30])
v36, v37, v38, v39, v40 = sch.sample_perfect_tile(loop=l24, n=5, max_innermost_factor=64, decision=[2, 1, 8, 2, 1])
l41, l42, l43, l44, l45 = sch.split(loop=l24, factors=[v36, v37, v38, v39, v40])
v46, v47, v48 = sch.sample_perfect_tile(loop=l25, n=3, max_innermost_factor=64, decision=[48, 1, 6])
l49, l50, l51 = sch.split(loop=l25, factors=[v46, v47, v48])
sch.reorder(l31, l41, l32, l42, l33, l43, l49, l50, l34, l44, l51, l35, l45)
l52 = sch.fuse(l31, l41)
sch.bind(loop=l52, thread_axis="blockIdx.x")
l53 = sch.fuse(l32, l42)
sch.bind(loop=l53, thread_axis="blockIdx.y")
l54 = sch.fuse(l33, l43)
sch.bind(loop=l54, thread_axis="threadIdx.y")
b55 = sch.read_at(loop=l49, block=b20, read_buffer_index=1, storage_scope="shared.dyn")
v56 = sch.sample_categorical(candidates=[4, 8, 16], probs=[0.33333333333333331, 0.33333333333333331, 0.33333333333333331], decision=2)
sch.annotate(block_or_loop=b55, ann_key="vector_bytes", ann_val=v56)
sch.annotate(block_or_loop=b55, ann_key="local_stage", ann_val=1)
sch.annotate(block_or_loop=b55, ann_key="double_buffer_scope", ann_val=0)
b57 = sch.read_at(loop=l49, block=b20, read_buffer_index=2, storage_scope="shared.dyn")
v58 = sch.sample_categorical(candidates=[4, 8, 16], probs=[0.33333333333333331, 0.33333333333333331, 0.33333333333333331], decision=0)
sch.annotate(block_or_loop=b57, ann_key="vector_bytes", ann_val=v58)
sch.annotate(block_or_loop=b57, ann_key="local_stage", ann_val=1)
sch.annotate(block_or_loop=b57, ann_key="double_buffer_scope", ann_val=0)
b59 = sch.read_at(loop=l50, block=b20, read_buffer_index=1, storage_scope="wmma.matrix_a")
b60 = sch.read_at(loop=l50, block=b20, read_buffer_index=2, storage_scope="wmma.matrix_b")
sch.annotate(block_or_loop=l50, ann_key="software_pipeline_stage", ann_val=[0, 0, 1])
sch.annotate(block_or_loop=l50, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
sch.annotate(block_or_loop=l49, ann_key="software_pipeline_stage", ann_val=[0, 0, 0, 0, 0, 1, 1])
sch.annotate(block_or_loop=l49, ann_key="software_pipeline_order", ann_val=[0, 3, 1, 4, 5, 2, 6])
b61 = sch.write_at(loop=l54, block=b20, write_buffer_index=0, storage_scope="wmma.accumulator")
v62 = sch.sample_categorical(candidates=[4, 8, 16], probs=[0.33333333333333331, 0.33333333333333331, 0.33333333333333331], decision=2)
sch.annotate(block_or_loop=b61, ann_key="vector_bytes", ann_val=v62)
sch.reverse_compute_inline(block=b4)
sch.compute_inline(block=b2)
sch.compute_inline(block=b1)
sch.compute_inline(block=b0)
for p in postprocs:
    p.apply(sch)
print(sch.mod.script())
mod = sch.mod["main"]
print(tvm.lower(mod, None, simple_mode=True))

dev = tvm.device("cuda", 0)
a_np = np.random.uniform(size=(3, 3, 512, 512)).astype("float16")
b_np = np.random.uniform(size=(32, 14, 14, 512)).astype("float16")
# c_np = np.dot(a_np.astype("float32"), b_np.astype("float32"))
a = tvm.nd.array(a_np, dev)
b = tvm.nd.array(b_np, dev)
c = tvm.nd.array(np.zeros((32, 7, 7, 512), dtype="float32"), dev)
f = tvm.build(mod, target="cuda", name="dense")
# print(f.imported_modules[0].get_source())
f(a, b, c)
# tvm.testing.assert_allclose(c.numpy(), c_np, rtol=1e-3)

evaluator = f.time_evaluator(f.entry_name, dev, number=1000)
# gflops = (N*M*K) * 2 / 1e9
time_ms = evaluator(a, b, c).mean * 1e3
# print("matmul with tensor core: %f ms, %f GFLOPS" % (time_ms, gflops / (time_ms / 1e3)))
print(time_ms)