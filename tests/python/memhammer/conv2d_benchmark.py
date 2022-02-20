import logging
import tempfile

import tvm
from tvm import tir
import pytest
from tvm.meta_schedule import ReplayTraceConfig, tune_tir
from tvm.meta_schedule.tune_context import TuneContext
from tvm.meta_schedule import schedule_rule, postproc
from tvm.meta_schedule.space_generator import PostOrderApply
from tvm.script import tir as T
from tvm.target.target import Target
from tvm.te.operation import create_prim_func
from tvm.tir import Schedule
from tvm.meta_schedule.testing import te_workload, tir_tensor_intrin


#todo: original test has batch_size = 1. In that case we'll need to handle imperfect tiling
@tvm.script.ir_module
class ImplicitGemm:
    @T.prim_func
    def main(placeholder_1: T.Buffer[(3, 3, 512, 512), "float32"],
          placeholder_2: T.Buffer[(32, 14, 14, 512), "float32"], Conv2dOutput: T.Buffer[(32, 7, 7, 512), "float32"]) ->\
            None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        PaddedInput = T.alloc_buffer([32, 16, 16, 512], dtype="float32")
        ActivationTransformed = T.alloc_buffer([32*7*7, 3*3*512], dtype="float32")
        WeightTransformed = T.alloc_buffer([3*3*512, 512], dtype="float32")
        OutputTransformed = T.alloc_buffer([32*7*7, 512], dtype="float32")
        for i0, i1, i2, i3 in T.grid(32, 16, 16, 512):
            with T.block("PaddedInput"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(placeholder_2[i0_1, i1_1 - 1, i2_1 - 1, i3_1])
                T.writes(PaddedInput[i0_1, i1_1, i2_1, i3_1])
                PaddedInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i1_1 and i1_1 < 15 and 1 <= i2_1 and i2_1 < 15, placeholder_2[i0_1, i1_1 - 1, i2_1 - 1, i3_1], T.float32(0), dtype="float32")

        for i in T.range(32*7*7):
            for k in T.range(3*3*512):
                with T.block("load_activation"):
                    vi, vk = T.axis.remap("SS",[i,k])
                    ActivationTransformed[vi, vk] = PaddedInput[vi//49, (vi%49//7)*2+vk//1536, vi%7*2+vk%1536//512, vk%512]

        for k in T.range(3*3*512):
            for j in T.range(512):
                with T.block("load_weight"):
                    vk, vj = T.axis.remap("SS",[k,j])
                    WeightTransformed[vk, vj] = placeholder_1[vk//1536, vk%1536//512, vk%512, vj]

        for i,j,k in T.grid(32*7*7, 512, 3*3*512):
            with T.block("gemm"):
                vi, vj, vk = T.axis.remap("SSR",[i,j,k])
                with T.init():
                    OutputTransformed[vi, vj] = T.float32(0)
                OutputTransformed[vi, vj] = ActivationTransformed[vi, vk] * WeightTransformed[vk, vj]
        for n,y,x,f in T.grid(32, 7, 7, 512):
            with T.block("epilogue"):
                nn,yy,xx,ff = T.axis.remap("SSSS",[n,y,x,f])
                Conv2dOutput[nn,yy,xx,ff] = OutputTransformed[nn*49+yy*7+xx,ff]
@tvm.script.ir_module
class Original:
    @T.prim_func
    def main(placeholder_1: T.Buffer[(3, 3, 512, 512), "float32"],
             placeholder_2: T.Buffer[(32, 14, 14, 512), "float32"], Conv2dOutput: T.Buffer[(32, 7, 7, 512), "float32"]) -> \
            None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        PaddedInput = T.alloc_buffer([32, 16, 16, 512], dtype="float32")
        for i0, i1, i2, i3 in T.grid(32, 16, 16, 512):
            with T.block("PaddedInput"):
                i0_1, i1_1, i2_1, i3_1 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(placeholder_2[i0_1, i1_1 - 1, i2_1 - 1, i3_1])
                T.writes(PaddedInput[i0_1, i1_1, i2_1, i3_1])
                PaddedInput[i0_1, i1_1, i2_1, i3_1] = T.if_then_else(1 <= i1_1 and i1_1 < 15 and 1 <= i2_1 and i2_1 < 15, placeholder_2[i0_1, i1_1 - 1, i2_1 - 1, i3_1], T.float32(0), dtype="float32")
        for i0, i1, i2, i3, i4, i5, i6 in T.grid(32, 7, 7, 512, 3, 3, 512):
            with T.block("Conv2dOutput"):
                nn, yy, xx, ff, ry, rx, rc = T.axis.remap("SSSSRRR", [i0, i1, i2, i3, i4, i5, i6])
                T.reads(Conv2dOutput[nn, yy, xx, ff], PaddedInput[nn, yy * 2 + ry, xx * 2 + rx, rc], placeholder_1[ry, rx, rc, ff])
                T.writes(Conv2dOutput[nn, yy, xx, ff])
                T.block_attr({"layout_free_placeholders":[placeholder_1], "workload":["conv2d_nhwc.gpu", ["TENSOR", [1, 14, 14, 512], "float32"], ["TENSOR", [3, 3, 512, 512], "float32"], [2, 2], [1, 1, 1, 1], [1, 1], "float32"]})
                with T.init():
                    Conv2dOutput[nn, yy, xx, ff] = T.float32(0)
                Conv2dOutput[nn, yy, xx, ff] = Conv2dOutput[nn, yy, xx, ff] + PaddedInput[nn, yy * 2 + ry, xx * 2 + rx, rc] * placeholder_1[ry, rx, rc, ff]

# sch = tir.Schedule(ImplicitGemm["main"])
# block_conv = sch.get_block("Conv2dOutput")
# n,p,q,k,r,s,c = sch.get_loops(block_conv)
# i = sch.fuse(n,p,q)
# j = k
# k = sch.fuse(r,s,c)
# i_factors = [None, 1, 4, 1, 4,16]
# j_factors = [1, None, 2, 1, 4,16]
# i0, i1, i2, i3, i4, tc_x = sch.split(i, factors=i_factors)
# j0, j1, j2, j3, j4, tc_y = sch.split(j, factors=j_factors)
# k_factors = [144, 32, 1]
# k0, k1, k2 = sch.split(k, factors=k_factors)
# sch.reorder(
#     # fmt: off
#     i0, j0,   # S => blockIdx.x
#     i1, j1,   # S => blockIdx.y
#     i2, j2,   # S => threadIdx.y
#     # cache_write here
#     k0,       # R
#     # vectorized cooperative fetching here
#     k1,       # R
#     i3, j3,   # S
#     k2,       # R
#     i4, j4,
#     tc_x,tc_y
#     # S
#     # fmt: on
# )
# block_idx = sch.fuse(i0, j0)
# block_idy = sch.fuse(i1, j1)
# thread_idy = sch.fuse(i2, j2)
# sch.bind(block_idx, "blockIdx.x")
# sch.bind(block_idy, "blockIdx.y")
# sch.bind(thread_idy, "threadIdx.y")
# sch.read_at(k0, block_conv, 1, "shared.dyn")
# # sch.tensorize()
# print(sch.mod.script())