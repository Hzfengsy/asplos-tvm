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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring

import tvm
from tvm import tir
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.postproc import RewriteCooperativeFetch
from tvm.meta_schedule.testing import te_workload
from tvm.script import tir as T
from tvm.target import Target
from tvm.te import create_prim_func


def _target() -> Target:
    return Target("cuda", host="llvm")


def _create_context(mod, target) -> TuneContext:
    ctx = TuneContext(
        mod=mod,
        target=target,
        postprocs=[
            RewriteCooperativeFetch(),
        ],
        task_name="test",
    )
    ctx.initialize()
    return ctx


# fmt: off
# pylint: disable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks

@tvm.script.ir_module
class AfterRewrite0:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle, var_C: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(var_A, [512, 512], dtype="float32")
        B = T.match_buffer(var_B, [512, 512], dtype="float32")
        C = T.match_buffer(var_C, [512, 512], dtype="float32")
        # body
        # with T.block("root")
        C_local = T.alloc_buffer([512, 512], dtype="float32", scope="local")
        A_shared = T.alloc_buffer([512, 512], dtype="float32", scope="shared")
        B_shared = T.alloc_buffer([512, 512], dtype="float32", scope="shared")
        for i0_0_i1_0_fused in T.thread_binding(0, 16, thread="blockIdx.x"):
            for i0_1_i1_1_fused in T.thread_binding(0, 16, thread="vthread.x"):
                for i0_2_i1_2_fused in T.thread_binding(0, 8, thread="threadIdx.x"):
                    for i2_0 in T.serial(0, 1):
                        for ax0_ax1_fused_0 in T.serial(0, 32768):
                            for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.x"):
                                with T.block("A_shared"):
                                    v0 = T.axis.spatial(512, (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1) // 512)
                                    v1 = T.axis.spatial(512, (ax0_ax1_fused_0 * 8 + ax0_ax1_fused_1) % 512)
                                    T.reads([A[v0, v1]])
                                    T.writes([A_shared[v0, v1]])
                                    A_shared[v0, v1] = A[v0, v1]
                        for ax0_ax1_fused_0 in T.serial(0, 1024):
                            for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.x"):
                                for ax0_ax1_fused_2 in T.vectorized(0, 2):
                                    with T.block("B_shared"):
                                        v0 = T.axis.spatial(512, (ax0_ax1_fused_0 * 16 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) // 32)
                                        v1 = T.axis.spatial(512, i0_0_i1_0_fused * 32 + (ax0_ax1_fused_0 * 16 + ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2) % 32)
                                        T.reads([B[v0, v1]])
                                        T.writes([B_shared[v0, v1]])
                                        B_shared[v0, v1] = B[v0, v1]
                        for i2_1, i0_3, i1_3, i2_2, i0_4, i1_4 in T.grid(16, 2, 2, 32, 16, 2):
                            with T.block("C"):
                                i = T.axis.spatial(512, i0_1_i1_1_fused * 32 + i0_3 * 16 + i0_4)
                                j = T.axis.spatial(512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + i1_3 * 2 + i1_4)
                                k = T.axis.reduce(512, i2_0 * 512 + i2_1 * 32 + i2_2)
                                T.reads([A_shared[i, k], B_shared[k, j]])
                                T.writes([C_local[i, j]])
                                with T.init():
                                    C_local[i, j] = T.float32(0)
                                C_local[i, j] = C_local[i, j] + A_shared[i, k] * B_shared[k, j]
                    for ax0, ax1 in T.grid(32, 4):
                        with T.block("C_local"):
                            v0 = T.axis.spatial(512, i0_1_i1_1_fused * 32 + ax0)
                            v1 = T.axis.spatial(512, i0_0_i1_0_fused * 32 + i0_2_i1_2_fused * 4 + ax1)
                            T.reads([C_local[v0, v1]])
                            T.writes([C[v0, v1]])
                            C[v0, v1] = C_local[v0, v1]


@tvm.script.ir_module
class AfterRewrite1:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle, var_C: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(var_A, [512, 512], dtype="float16")
        B = T.match_buffer(var_B, [512, 512], dtype="float16")
        C = T.match_buffer(var_C, [512, 512], dtype="float32")
        # body
        with T.block("root"):
            T.reads([])
            T.writes([])
            T.block_attr({"meta_schedule.tensor_core_enabled":"1"})
            C_local = T.alloc_buffer([512, 512], dtype="float32", scope="local")
            C_local_wmma_accumulator = T.alloc_buffer([512, 512], dtype="float32", scope="wmma.accumulator")
            A_shared = T.alloc_buffer([512, 512], dtype="float16", scope="shared")
            B_shared = T.alloc_buffer([512, 512], dtype="float16", scope="shared")
            A_shared_wmma_matrix_a = T.alloc_buffer([512, 512], dtype="float16", scope="wmma.matrix_a")
            B_shared_wmma_matrix_b = T.alloc_buffer([512, 512], dtype="float16", scope="wmma.matrix_b")
            for i0_0_0_i1_0_0_fused in T.thread_binding(0, 1, thread="blockIdx.x"):
                for i0_0_1_i1_0_1_fused in T.thread_binding(0, 4, thread="blockIdx.y"):
                    for i0_0_2_i1_0_2_fused in T.thread_binding(0, 8, thread="threadIdx.y"):
                        for i2_0_0 in T.serial(0, 4):
                            for ax0_ax1_fused_0 in T.serial(0, 128):
                                for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(0, 32, thread="threadIdx.x"):
                                        with T.block("A_shared"):
                                            v0 = T.axis.spatial(512, i0_0_1_i1_0_1_fused // 2 * 256 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 32 + ax0_ax1_fused_2) // 128)
                                            v1 = T.axis.spatial(512, i2_0_0 * 128 + (ax0_ax1_fused_0 * 256 + ax0_ax1_fused_1 * 32 + ax0_ax1_fused_2) % 128)
                                            T.reads([A[v0, v1]])
                                            T.writes([A_shared[v0, v1]])
                                            T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]], "meta_schedule.cooperative_fetch":1})
                                            A_shared[v0, v1] = A[v0, v1]
                            for ax0_ax1_fused_0 in T.serial(0, 32):
                                for ax0_ax1_fused_1 in T.thread_binding(0, 8, thread="threadIdx.y"):
                                    for ax0_ax1_fused_2 in T.thread_binding(0, 32, thread="threadIdx.x"):
                                        for ax0_ax1_fused_3 in T.vectorized(0, 4):
                                            with T.block("B_shared"):
                                                v0 = T.axis.spatial(512, i2_0_0 * 128 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) // 256)
                                                v1 = T.axis.spatial(512, i0_0_1_i1_0_1_fused % 2 * 256 + (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 * 128 + ax0_ax1_fused_2 * 4 + ax0_ax1_fused_3) % 256)
                                                T.reads([B[v0, v1]])
                                                T.writes([B_shared[v0, v1]])
                                                T.block_attr({"buffer_dim_align":[[0, 0, 32, 8]], "meta_schedule.cooperative_fetch":4})
                                                B_shared[v0, v1] = B[v0, v1]
                            for i2_0_1, i0_0_3, i1_0_3, i2_0_2 in T.grid(8, 1, 1, 1):
                                for ax0, ax1 in T.grid(256, 16):
                                    with T.block("A_shared_wmma.matrix_a"):
                                        v0 = T.axis.spatial(512, i0_0_1_i1_0_1_fused // 2 * 256 + ax0)
                                        v1 = T.axis.spatial(512, i2_0_0 * 128 + i2_0_1 * 16 + ax1)
                                        T.reads([A_shared[v0, v1]])
                                        T.writes([A_shared_wmma_matrix_a[v0, v1]])
                                        T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_a"})
                                        A_shared_wmma_matrix_a[v0, v1] = A_shared[v0, v1]
                                for ax0, ax1 in T.grid(16, 32):
                                    with T.block("B_shared_wmma.matrix_b"):
                                        v0 = T.axis.spatial(512, i2_0_0 * 128 + i2_0_1 * 16 + ax0)
                                        v1 = T.axis.spatial(512, i0_0_1_i1_0_1_fused % 2 * 256 + i0_0_2_i1_0_2_fused * 32 + ax1)
                                        T.reads([B_shared[v0, v1]])
                                        T.writes([B_shared_wmma_matrix_b[v0, v1]])
                                        T.block_attr({"meta_schedule.auto_tensorize":"wmma_load_b"})
                                        B_shared_wmma_matrix_b[v0, v1] = B_shared[v0, v1]
                                for i0_0_4, i1_0_4 in T.grid(16, 2):
                                    with T.block("blockized_C"):
                                        io = T.axis.spatial(32, i0_0_1_i1_0_1_fused // 2 * 16 + i0_0_4)
                                        jo = T.axis.spatial(32, i0_0_1_i1_0_1_fused % 2 * 16 + i0_0_2_i1_0_2_fused * 2 + i1_0_4)
                                        ko = T.axis.reduce(32, i2_0_0 * 8 + i2_0_1)
                                        T.reads([C_local_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16], A_shared_wmma_matrix_a[io * 16 : io * 16 + 16, ko * 16 : ko * 16 + 16], B_shared_wmma_matrix_b[ko * 16 : ko * 16 + 16, jo * 16 : jo * 16 + 16]])
                                        T.writes([C_local_wmma_accumulator[io * 16 : io * 16 + 16, jo * 16 : jo * 16 + 16]])
                                        T.block_attr({"meta_schedule.auto_tensorize":"wmma_fill"})
                                        with T.init():
                                            for i0_1, i1_1 in T.grid(16, 16):
                                                with T.block("C_init"):
                                                    i_init, j_init = T.axis.remap("SS", [i0_1, i1_1])
                                                    T.reads()
                                                    T.writes(C_local_wmma_accumulator[io * 16 + i_init, jo * 16 + j_init])
                                                    C_local_wmma_accumulator[io * 16 + i_init, jo * 16 + j_init] = T.float32(0)
                                        for i0_1, i1_1, i2_1 in T.grid(16, 16, 16):
                                            with T.block("C"):
                                                i, j, k = T.axis.remap("SSR", [i0_1, i1_1, i2_1])
                                                T.reads(C_local_wmma_accumulator[io * 16 + i, jo * 16 + j], A_shared_wmma_matrix_a[io * 16 + i, ko * 16 + k], B_shared_wmma_matrix_b[ko * 16 + k, jo * 16 + j])
                                                T.writes(C_local_wmma_accumulator[io * 16 + i, jo * 16 + j])
                                                T.block_attr({"meta_schedule.auto_tensorize":"wmma_sync"})
                                                C_local_wmma_accumulator[io * 16 + i, jo * 16 + j] = C_local_wmma_accumulator[io * 16 + i, jo * 16 + j] + T.cast(A_shared_wmma_matrix_a[io * 16 + i, ko * 16 + k], "float32") * T.cast(B_shared_wmma_matrix_b[ko * 16 + k, jo * 16 + j], "float32")
                        for ax0, ax1 in T.grid(256, 32):
                            with T.block("C_local_wmma.accumulator"):
                                v0 = T.axis.spatial(512, i0_0_1_i1_0_1_fused // 2 * 256 + ax0)
                                v1 = T.axis.spatial(512, i0_0_1_i1_0_1_fused % 2 * 256 + i0_0_2_i1_0_2_fused * 32 + ax1)
                                T.reads([C_local_wmma_accumulator[v0, v1]])
                                T.writes([C_local[v0, v1]])
                                T.block_attr({"meta_schedule.auto_tensorize":"wmma_store"})
                                C_local[v0, v1] = C_local_wmma_accumulator[v0, v1]
                        for ax0, ax1 in T.grid(256, 32):
                            with T.block("C_local"):
                                v0 = T.axis.spatial(512, i0_0_1_i1_0_1_fused // 2 * 256 + ax0)
                                v1 = T.axis.spatial(512, i0_0_1_i1_0_1_fused % 2 * 256 + i0_0_2_i1_0_2_fused * 32 + ax1)
                                T.reads([C_local[v0, v1]])
                                T.writes([C[v0, v1]])
                                C[v0, v1] = C_local[v0, v1]


# pylint: enable=no-member,invalid-name,unused-variable,no-self-argument,line-too-long,chained-comparison,not-callable,too-many-nested-blocks
# fmt: on


def test_rewrite_cooperative_fetch():
    mod = create_prim_func(te_workload.matmul(n=512, m=512, k=512))
    target = _target()
    ctx = _create_context(mod, target)

    sch = tir.Schedule(mod, debug_mask="all")
    # fmt: off
    # pylint: disable=line-too-long,invalid-name
    b0 = sch.get_block(name="C", func_name="main")
    b1 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")
    l2, l3, l4 = sch.get_loops(block=b0)
    v5, v6, v7, v8, v9 = sch.sample_perfect_tile(loop=l2, n=5, max_innermost_factor=64, decision=[1, 16, 1, 2, 16])
    l10, l11, l12, l13, l14 = sch.split(loop=l2, factors=[v5, v6, v7, v8, v9])
    v15, v16, v17, v18, v19 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64, decision=[16, 1, 8, 2, 2])
    l20, l21, l22, l23, l24 = sch.split(loop=l3, factors=[v15, v16, v17, v18, v19])
    v25, v26, v27 = sch.sample_perfect_tile(loop=l4, n=3, max_innermost_factor=64, decision=[1, 16, 32])
    l28, l29, l30 = sch.split(loop=l4, factors=[v25, v26, v27])
    sch.reorder(l10, l20, l11, l21, l12, l22, l28, l29, l13, l23, l30, l14, l24)
    l31 = sch.fuse(l10, l20)
    sch.bind(loop=l31, thread_axis="blockIdx.x")
    l32 = sch.fuse(l11, l21)
    sch.bind(loop=l32, thread_axis="vthread.x")
    l33 = sch.fuse(l12, l22)
    sch.bind(loop=l33, thread_axis="threadIdx.x")
    b34 = sch.cache_read(block=b0, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b34, loop=l28, preserve_unit_loops=True)
    _, _, _, _, l39, l40 = sch.get_loops(block=b34)
    l41 = sch.fuse(l39, l40)
    _, v43 = sch.sample_perfect_tile(loop=l41, n=2, max_innermost_factor=4, decision=[262144, 1])
    sch.annotate(block_or_loop=b34, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)
    b44 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared")
    sch.compute_at(block=b44, loop=l28, preserve_unit_loops=True)
    _, _, _, _, l49, l50 = sch.get_loops(block=b44)
    l51 = sch.fuse(l49, l50)
    _, v53 = sch.sample_perfect_tile(loop=l51, n=2, max_innermost_factor=4, decision=[8192, 2])
    sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v53)
    sch.reverse_compute_at(block=b1, loop=l33, preserve_unit_loops=True)
    # pylint: enable=line-too-long,invalid-name
    # fmt: on
    sch.enter_postproc()
    assert ctx.postprocs[0].apply(sch)
    tvm.ir.assert_structural_equal(sch.mod, AfterRewrite0)


def test_rewrite_cooperative_fetch_tensor_core():
    mod = create_prim_func(te_workload.matmul_fp16(n=512, m=512, k=512))
    target = _target()
    ctx = _create_context(mod, target)

    sch = tir.Schedule(mod, debug_mask="all")
    # fmt: off
    # pylint: disable=line-too-long,invalid-name
    b0 = sch.get_block(name="C", func_name="main")
    l1, l2, l3 = sch.get_loops(block=b0)
    _, l5 = sch.split(loop=l1, factors=[32, 16])
    _, l7 = sch.split(loop=l2, factors=[32, 16])
    _, l9 = sch.split(loop=l3, factors=[32, 16])
    _, _, l12, _, l14, _ = sch.get_loops(block=b0)
    sch.reorder(l12, l14, l5, l7, l9)
    b16 = sch.blockize(loop=l5)
    sch.annotate(block_or_loop=b0, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync")
    sch.annotate(block_or_loop=b16, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill")
    b17 = sch.get_block(name="root", func_name="main")
    sch.annotate(block_or_loop=b17, ann_key="meta_schedule.tensor_core_enabled", ann_val="1")
    b18 = sch.cache_write(block=b16, write_buffer_index=0, storage_scope="local")
    b19 = sch.cache_write(block=b16, write_buffer_index=0, storage_scope="wmma.accumulator")
    sch.annotate(block_or_loop=b19, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_store")
    l20, l21, l22 = sch.get_loops(block=b16)
    v23, v24, v25, v26, v27 = sch.sample_perfect_tile(loop=l20, n=5, max_innermost_factor=64, decision=[1, 2, 1, 1, 16])
    l28, l29, l30, l31, l32 = sch.split(loop=l20, factors=[v23, v24, v25, v26, v27])
    v33, v34, v35, v36, v37 = sch.sample_perfect_tile(loop=l21, n=5, max_innermost_factor=64, decision=[1, 2, 8, 1, 2])
    l38, l39, l40, l41, l42 = sch.split(loop=l21, factors=[v33, v34, v35, v36, v37])
    v43, v44, v45 = sch.sample_perfect_tile(loop=l22, n=3, max_innermost_factor=64, decision=[4, 8, 1])
    l46, l47, l48 = sch.split(loop=l22, factors=[v43, v44, v45])
    sch.reorder(l28, l38, l29, l39, l30, l40, l46, l47, l31, l41, l48, l32, l42)
    l49 = sch.fuse(l28, l38)
    sch.bind(loop=l49, thread_axis="blockIdx.x")
    l50 = sch.fuse(l29, l39)
    sch.bind(loop=l50, thread_axis="blockIdx.y")
    l51 = sch.fuse(l30, l40)
    sch.bind(loop=l51, thread_axis="threadIdx.y")
    b52 = sch.cache_read(block=b16, read_buffer_index=1, storage_scope="shared")
    sch.compute_at(block=b52, loop=l46, preserve_unit_loops=True)
    _, _, _, _, l57, l58 = sch.get_loops(block=b52)
    l59 = sch.fuse(l57, l58)
    _, v61 = sch.sample_perfect_tile(loop=l59, n=2, max_innermost_factor=4, decision=[32768, 1])
    sch.annotate(block_or_loop=b52, ann_key="meta_schedule.cooperative_fetch", ann_val=v61)
    b62 = sch.cache_read(block=b16, read_buffer_index=2, storage_scope="shared")
    sch.compute_at(block=b62, loop=l46, preserve_unit_loops=True)
    _, _, _, _, l67, l68 = sch.get_loops(block=b62)
    l69 = sch.fuse(l67, l68)
    _, v71 = sch.sample_perfect_tile(loop=l69, n=2, max_innermost_factor=4, decision=[8192, 4])
    sch.annotate(block_or_loop=b62, ann_key="meta_schedule.cooperative_fetch", ann_val=v71)
    b72 = sch.cache_read(block=b16, read_buffer_index=1, storage_scope="wmma.matrix_a")
    b73 = sch.cache_read(block=b16, read_buffer_index=2, storage_scope="wmma.matrix_b")
    sch.compute_at(block=b72, loop=l48, preserve_unit_loops=True)
    sch.compute_at(block=b73, loop=l48, preserve_unit_loops=True)
    sch.annotate(block_or_loop=b72, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_a")
    sch.annotate(block_or_loop=b73, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_b")
    sch.reverse_compute_at(block=b19, loop=l51, preserve_unit_loops=True)
    sch.reverse_compute_at(block=b18, loop=l51, preserve_unit_loops=True)
    # pylint: enable=line-too-long,invalid-name
    # fmt: on
    sch.enter_postproc()
    assert ctx.postprocs[0].apply(sch)
    tvm.ir.assert_structural_equal(sch.mod, AfterRewrite1)


if __name__ == "__main__":
    test_rewrite_cooperative_fetch()
    test_rewrite_cooperative_fetch_tensor_core()
