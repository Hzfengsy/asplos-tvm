import numpy as np

import tvm
import tvm.testing
from tvm import te
from tvm.script import tir as T
from tvm import meta_schedule as ms
from tvm._ffi import register_func
from tvm import tir
from tvm.runtime import convert
from tvm.contrib import cublas


def matmul_fp16(  # pylint: disable=invalid-name,missing-docstring
    N: int,
    M: int,
    K: int,
):
    x = te.placeholder((N, K), name="X", dtype="float16")
    y = te.placeholder((K, M), name="Y", dtype="float16")
    k = te.reduce_axis((0, K), name="k")
    c = te.compute(  # pylint: disable=invalid-name
        (N, M),
        lambda i, j: te.sum(x[i][k] * y[k][j], axis=[k]),
        name="C",
    )
    return (x, y, c)

N = 4096
M = 4096
K = 4096

workload = matmul_fp16(N=N, M=M, K=K)
workload = te.create_prim_func(workload)

from tvm.script import tir as T
@T.prim_func
def func(X: T.Buffer[(4096, 4096), "float16"], Y: T.Buffer[(4096, 4096), "float16"], C: T.Buffer[(4096, 4096), "float16"]) -> None:
    # function attr dict
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    # var definition
    tx = T.env_thread("threadIdx.x")
    tx = T.env_thread("threadIdx.x")
    tx = T.env_thread("threadIdx.x")
    tx = T.env_thread("threadIdx.x")
    a0 = T.var("int32")
    a1 = T.var("int32")
    b0 = T.var("int32")
    b1 = T.var("int32")
    c0 = T.var("int32")
    c1 = T.var("int32")
    d0 = T.var("int32")
    d0_1 = T.var("int32")
    d1 = T.var("int32")
    d1_1 = T.var("int32")
    s0 = T.var("int32")
    s0_1 = T.var("int32")
    s0_2 = T.var("int32")
    s1 = T.var("int32")
    s1_1 = T.var("int32")
    s1_2 = T.var("int32")
    # body
    with T.block("root"):
        T.reads()
        T.writes()
        T.block_attr({"warp_execution":1})
        for by in T.thread_binding(4, thread="blockIdx.y"):
            for bx in T.thread_binding(256, thread="blockIdx.x"):
                for ty in T.thread_binding(4, thread="threadIdx.y"):
                    with T.block():
                        T.reads(X[bx // 8 * 128 : bx // 8 * 128 + 128, 0 : 4096], Y[0 : 4096, by * 1024 + bx % 8 * 128 : by * 1024 + bx % 8 * 128 + 128])
                        T.writes(C[bx // 8 * 128 + ty % 2 * 64 : bx // 8 * 128 + ty % 2 * 64 + 64, by * 1024 + bx % 8 * 128 + ty // 2 * 64 : by * 1024 + bx % 8 * 128 + ty // 2 * 64 + 64])
                        C_m16n8k8_matrixC = T.alloc_buffer([64, 64], dtype="float16", scope="m16n8k8.matrixC")
                        for i0_0_3_init, i1_0_3_init, i0_0_4_init, i1_0_4_init in T.grid(1, 1, 4, 8):
                            with T.block("C_o_init"):
                                T.reads()
                                T.writes(C_m16n8k8_matrixC[i0_0_4_init * 16 : i0_0_4_init * 16 + 16, i1_0_4_init * 8 : i1_0_4_init * 8 + 8])
                                for i0_1_0 in T.serial(2):
                                    with T.block("C_init_o"):
                                        T.reads()
                                        T.writes(C_m16n8k8_matrixC[i0_0_4_init * 16 + i0_1_0 * 8 : i0_0_4_init * 16 + i0_1_0 * 8 + 8, i1_0_4_init * 8 : i1_0_4_init * 8 + 8])
                                        dst = T.match_buffer(C_m16n8k8_matrixC[i0_0_4_init * 16 + i0_1_0 * 8 : i0_0_4_init * 16 + i0_1_0 * 8 + 8, i1_0_4_init * 8 : i1_0_4_init * 8 + 8], [8, 8], dtype="float16", scope="m16n8k8.matrixC", offset_factor=1)
                                        T.launch_thread(tx, 32)
                                        for i in T.vectorized(2):
                                            dst[tx // 4, tx % 4 * 2 + i] = T.float16(0)
                        for i2_0_0 in T.serial(128):
                            with T.block():
                                T.reads(X[bx // 8 * 128 : bx // 8 * 128 + 128, i2_0_0 * 32 : i2_0_0 * 32 + 32], Y[i2_0_0 * 32 : i2_0_0 * 32 + 32, by * 1024 + bx % 8 * 128 : by * 1024 + bx % 8 * 128 + 128], C_m16n8k8_matrixC[0 : 64, 0 : 64])
                                T.writes(C_m16n8k8_matrixC[0 : 64, 0 : 64])
                                X_shared_dyn = T.alloc_buffer([128, 32], dtype="float16", strides=[32, 1], scope="shared.dyn")
                                Y_shared_dyn = T.alloc_buffer([32, 128], dtype="float16", strides=[128, 1], scope="shared.dyn")
                                with T.block("X_shared.dyn"):
                                    T.reads(X[bx // 8 * 128 : bx // 8 * 128 + 128, i2_0_0 * 32 : i2_0_0 * 32 + 32])
                                    T.writes(X_shared_dyn[0 : 128, 0 : 32])
                                    T.block_attr({"auto_copy":1, "double_buffer_scope":0, "local_stage":1, "meta_schedule.cache_type":0, "vector_bytes":16})
                                    X_shared_dyn_local = T.alloc_buffer([4, 8], dtype="float16", scope="local")
                                    for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                        for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_ax1_fused_0_cache in T.serial(4):
                                                for ax0_ax1_fused_3_cache in T.vectorized(8):
                                                    X_shared_dyn_local[ax0_ax1_fused_0_cache, ax0_ax1_fused_3_cache] = X[bx // 8 * 128 + (ax0_ax1_fused_0_cache * 4 + ax0_ax1_fused_1) * 8 + ax0_ax1_fused_2 // 4, i2_0_0 * 32 + ax0_ax1_fused_2 % 4 * 8 + ax0_ax1_fused_3_cache]
                                            for ax0_ax1_fused_0 in T.serial(4):
                                                for ax0_ax1_fused_3 in T.vectorized(8):
                                                    # X_shared_dyn[(ax0_ax1_fused_0 * 4 + ax0_ax1_fused_1) * 8 + ax0_ax1_fused_2 // 4, (ax0_ax1_fused_2 % 4) * 8 + ax0_ax1_fused_3] = X_shared_dyn_local[ax0_ax1_fused_0, ax0_ax1_fused_3]
                                                    X_shared_dyn[(ax0_ax1_fused_0 * 4 + ax0_ax1_fused_1) * 8 + ax0_ax1_fused_2 // 4, ((ax0_ax1_fused_2 // 8) ^ (ax0_ax1_fused_2 % 4)) * 8 + ax0_ax1_fused_3] = X_shared_dyn_local[ax0_ax1_fused_0, ax0_ax1_fused_3]
                                with T.block("Y_shared.dyn"):
                                    T.reads(Y[i2_0_0 * 32 : i2_0_0 * 32 + 32, by * 1024 + bx % 8 * 128 : by * 1024 + bx % 8 * 128 + 128])
                                    T.writes(Y_shared_dyn[0 : 32, 0 : 128])
                                    T.block_attr({"auto_copy":1, "double_buffer_scope":0, "local_stage":1, "meta_schedule.cache_type":0, "vector_bytes":16})
                                    Y_shared_dyn_local = T.alloc_buffer([4, 8], dtype="float16", scope="local")
                                    for ax0_ax1_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                        for ax0_ax1_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_ax1_fused_0_cache in T.serial(4):
                                                for ax0_ax1_fused_3_cache in T.vectorized(8):
                                                    Y_shared_dyn_local[ax0_ax1_fused_0_cache, ax0_ax1_fused_3_cache] = Y[i2_0_0 * 32 + (ax0_ax1_fused_0_cache * 4 + ax0_ax1_fused_1) * 2 + ax0_ax1_fused_2 // 16, by * 1024 + bx % 8 * 128 + ax0_ax1_fused_2 % 16 * 8 + ax0_ax1_fused_3_cache]
                                            for ax0_ax1_fused_0 in T.serial(4):
                                                for ax0_ax1_fused_3 in T.vectorized(8):
                                                    # Y_shared_dyn[(ax0_ax1_fused_0 * 4 + ax0_ax1_fused_1) * 2 + ax0_ax1_fused_2 // 16, (ax0_ax1_fused_2 % 16) * 8 + ax0_ax1_fused_3] = Y_shared_dyn_local[ax0_ax1_fused_0, ax0_ax1_fused_3]
                                                    Y_shared_dyn[(ax0_ax1_fused_0 * 4 + ax0_ax1_fused_1) * 2 + ax0_ax1_fused_2 // 16, (((ax0_ax1_fused_2 % 16) // 8) * 8 + ((ax0_ax1_fused_2 % 8) ^ (ax0_ax1_fused_1 * 2 + ax0_ax1_fused_2 // 16))) * 8 + ax0_ax1_fused_3] = Y_shared_dyn_local[ax0_ax1_fused_0, ax0_ax1_fused_3]

                                for i2_0_1 in T.serial(4):
                                    with T.block():
                                        T.reads(X_shared_dyn[ty % 2 * 64 : ty % 2 * 64 + 64, i2_0_1 * 8 : i2_0_1 * 8 + 8], Y_shared_dyn[i2_0_1 * 8 : i2_0_1 * 8 + 8, ty // 2 * 64 : ty // 2 * 64 + 64], C_m16n8k8_matrixC[0 : 64, 0 : 64])
                                        T.writes(C_m16n8k8_matrixC[0 : 64, 0 : 64])
                                        X_shared_dyn_m16n8k8_matrixA = T.alloc_buffer([64, 8], dtype="float16", scope="m16n8k8.matrixA")
                                        Y_shared_dyn_m16n8k8_matrixB = T.alloc_buffer([8, 64], dtype="float16", scope="m16n8k8.matrixB")
                                        for ax0_0, ax1_0 in T.grid(2, 1):
                                            with T.block("X_shared.dyn_m16n8k8.matrixA_o"):
                                                T.reads(X_shared_dyn[ty % 2 * 64 + ax0_0 * 32 : ty % 2 * 64 + ax0_0 * 32 + 32, i2_0_1 * 8 : i2_0_1 * 8 + 8])
                                                T.writes(X_shared_dyn_m16n8k8_matrixA[ax0_0 * 32 : ax0_0 * 32 + 32, 0 : 8])
                                                src = T.match_buffer(X_shared_dyn[ty % 2 * 64 + ax0_0 * 32 : ty % 2 * 64 + ax0_0 * 32 + 32, i2_0_1 * 8 : i2_0_1 * 8 + 8], [32, 8], dtype="float16", strides=[s0, s1], scope="shared.dyn", offset_factor=1)
                                                dst_1 = T.match_buffer(X_shared_dyn_m16n8k8_matrixA[ax0_0 * 32 : ax0_0 * 32 + 32, 0 : 8], [32, 8], dtype="float16", strides=[d0, d1], scope="m16n8k8.matrixA", offset_factor=1)
                                                T.launch_thread(tx, 32)
                                                # T.evaluate(T.ptx_ldmatrix(False, 4, ".b16", dst_1.data, dst_1.elem_offset % d0 // 8 * 8 + dst_1.elem_offset // d0 // 32 * (d0 // 8) * 8 + dst_1.elem_offset // d0 % 32 // 16 * 4, T.tvm_access_ptr(T.type_annotation(dtype="float16"), src.data, src.elem_offset, s0 * 32, 1, dtype="handle"), tx * s0, dtype="float16"))
                                                T.evaluate(T.ptx_ldmatrix(False, 4, ".b16", dst_1.data, dst_1.elem_offset % d0 // 8 * 8 + dst_1.elem_offset // d0 // 32 * (d0 // 8) * 8 + dst_1.elem_offset // d0 % 32 // 16 * 4, T.tvm_access_ptr(T.type_annotation(dtype="float16"), src.data, (ty % 2) * 2048 + ax0_0 * 1024 + tx * 32, 8, 1, dtype="handle"), (((tx // 2) % 4) ^ i2_0_1) * 8, dtype="float16"))
                                        for ax0_0, ax1_0 in T.grid(1, 2):
                                            with T.block("Y_shared.dyn_m16n8k8.matrixB_o"):
                                                T.reads(Y_shared_dyn[i2_0_1 * 8 : i2_0_1 * 8 + 8, ty // 2 * 64 + ax1_0 * 32 : ty // 2 * 64 + ax1_0 * 32 + 32])
                                                T.writes(Y_shared_dyn_m16n8k8_matrixB[0 : 8, ax1_0 * 32 : ax1_0 * 32 + 32])
                                                src_1 = T.match_buffer(Y_shared_dyn[i2_0_1 * 8 : i2_0_1 * 8 + 8, ty // 2 * 64 + ax1_0 * 32 : ty // 2 * 64 + ax1_0 * 32 + 32], [8, 32], dtype="float16", strides=[s0_1, s1_1], scope="shared.dyn", offset_factor=1)
                                                dst_2 = T.match_buffer(Y_shared_dyn_m16n8k8_matrixB[0 : 8, ax1_0 * 32 : ax1_0 * 32 + 32], [8, 32], dtype="float16", strides=[d0_1, d1_1], scope="m16n8k8.matrixB", offset_factor=1)
                                                T.launch_thread(tx, 32)
                                                # T.evaluate(T.ptx_ldmatrix(True, 4, ".b16", dst_2.data, dst_2.elem_offset // d0_1 // 8 * (d0_1 // 32) * 8 + dst_2.elem_offset % d0_1 // 8 * 2, T.tvm_access_ptr(T.type_annotation(dtype="float16"), src_1.data, src_1.elem_offset, s0_1 * 8, 1, dtype="handle"), tx // 8 * 8 + s0_1 * (tx % 8), dtype="float16"))
                                                T.evaluate(T.ptx_ldmatrix(True, 4, ".b16", dst_2.data, dst_2.elem_offset // d0_1 // 8 * (d0_1 // 32) * 8 + dst_2.elem_offset % d0_1 // 8 * 2, T.tvm_access_ptr(T.type_annotation(dtype="float16"), src_1.data, (i2_0_1 * 8 + tx % 8) * 128, s0_1 * 8, 1, dtype="handle"), ((ty // 2) * 8 + ((ax1_0 * 4 + tx // 8) ^ (tx % 8))) * 8, dtype="float16"))
                                        for i0_0_3, i1_0_3, i2_0_2, i0_0_4, i1_0_4 in T.grid(1, 1, 1, 4, 8):
                                            with T.block("C_o_update"):
                                                T.reads(C_m16n8k8_matrixC[i0_0_4 * 16 : i0_0_4 * 16 + 16, i1_0_4 * 8 : i1_0_4 * 8 + 8], X_shared_dyn_m16n8k8_matrixA[i0_0_4 * 16 : i0_0_4 * 16 + 16, 0 : 8], Y_shared_dyn_m16n8k8_matrixB[0 : 8, i1_0_4 * 8 : i1_0_4 * 8 + 8])
                                                T.writes(C_m16n8k8_matrixC[i0_0_4 * 16 : i0_0_4 * 16 + 16, i1_0_4 * 8 : i1_0_4 * 8 + 8])
                                                with T.block("C_o"):
                                                    T.reads(C_m16n8k8_matrixC[i0_0_4 * 16 : i0_0_4 * 16 + 16, i1_0_4 * 8 : i1_0_4 * 8 + 8], X_shared_dyn_m16n8k8_matrixA[i0_0_4 * 16 : i0_0_4 * 16 + 16, 0 : 8], Y_shared_dyn_m16n8k8_matrixB[0 : 8, i1_0_4 * 8 : i1_0_4 * 8 + 8])
                                                    T.writes(C_m16n8k8_matrixC[i0_0_4 * 16 : i0_0_4 * 16 + 16, i1_0_4 * 8 : i1_0_4 * 8 + 8])
                                                    A = T.match_buffer(X_shared_dyn_m16n8k8_matrixA[i0_0_4 * 16 : i0_0_4 * 16 + 16, 0 : 8], [16, 8], dtype="float16", strides=[a0, a1], scope="m16n8k8.matrixA", offset_factor=1)
                                                    B = T.match_buffer(Y_shared_dyn_m16n8k8_matrixB[0 : 8, i1_0_4 * 8 : i1_0_4 * 8 + 8], [8, 8], dtype="float16", strides=[b0, b1], scope="m16n8k8.matrixB", offset_factor=1)
                                                    C_1 = T.match_buffer(C_m16n8k8_matrixC[i0_0_4 * 16 : i0_0_4 * 16 + 16, i1_0_4 * 8 : i1_0_4 * 8 + 8], [16, 8], dtype="float16", strides=[c0, c1], scope="m16n8k8.matrixC", offset_factor=1)
                                                    T.evaluate(T.ptx_mma("m16n8k8", "row", "col", "fp16", "fp16", "fp16", A.data, A.elem_offset % a0 // 8 * 8 + A.elem_offset // a0 // 32 * (a0 // 8) * 8 + A.elem_offset // a0 % 32 // 16 * 4, B.data, B.elem_offset // b0 // 8 * (b0 // 32) * 8 + B.elem_offset % b0 // 8 * 2, C_1.data, C_1.elem_offset % c0 // 8 * 2 + C_1.elem_offset // c0 % 16 // 8 + C_1.elem_offset // c0 // 16 * 2 * (c0 // 8), False, dtype="float16"))
                        with T.block("C_m16n8k8.matrixC"):
                            T.reads(C_m16n8k8_matrixC[0 : 64, 0 : 64])
                            T.writes(C[bx // 8 * 128 + ty % 2 * 64 : bx // 8 * 128 + ty % 2 * 64 + 64, by * 1024 + bx % 8 * 128 + ty // 2 * 64 : by * 1024 + bx % 8 * 128 + ty // 2 * 64 + 64])
                            T.block_attr({"auto_copy":1})
                            C_m16n8k8_matrixC_shared_dyn = T.alloc_buffer([4, 8, 8, 8], dtype="float16", strides=[512, 64, 8, 1], scope="shared.dyn")
                            for ax0_0 in T.serial(8):
                                for ax1_0 in T.serial(8):
                                    with T.block("mma_store"):
                                        T.reads(C_m16n8k8_matrixC[ax0_0 * 8 : ax0_0 * 8 + 8, ax1_0 * 8 : ax1_0 * 8 + 8])
                                        T.writes(C_m16n8k8_matrixC_shared_dyn[ty, ax1_0, 0 : 8, 0 : 8])
                                        src_2 = T.match_buffer(C_m16n8k8_matrixC[ax0_0 * 8 : ax0_0 * 8 + 8, ax1_0 * 8 : ax1_0 * 8 + 8], [8, 8], dtype="float16", scope="m16n8k8.matrixC", offset_factor=8)
                                        tgt = T.match_buffer(C_m16n8k8_matrixC_shared_dyn[ty, ax1_0, 0 : 8, 0 : 8], [8, 8], dtype="float16", strides=[s1_2, s0_2], scope="shared.dyn", offset_factor=8)
                                        T.launch_thread(tx, 32)
                                        for vec in T.vectorized(2):
                                            tgt[tx // 4, tx % 4 * 2 + vec] = src_2[tx // 4, tx % 4 * 2 + vec]
                                for ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 in T.serial(16):
                                    for ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1 in T.thread_binding(4, thread="threadIdx.y"):
                                        for ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_3 in T.vectorized(1):
                                                C[bx // 8 * 128 + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 % 8 // 4 * 64 + ax0_0 * 8 + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1 % 2 * 4 + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2 // 8, by * 1024 + bx % 8 * 128 + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 // 8 * 64 + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 % 4 * 16 + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1 // 2 * 8 + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2 % 8] = C_m16n8k8_matrixC_shared_dyn[ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 // 4, ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 % 4 * 2 + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1 // 2, ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_1 % 2 * 4 + ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2 // 8, ty_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_2 % 8]
                                                

@register_func("permuted_layout")
def permuted_layout():
    return func


@register_func("tir.index_map_m16n8k8.matrixC")
def index_map_m16n8k8_matrixC(ind):
    i, j = ind[0], ind[1]
    return convert([(i // 8) // 2, j // 8, (i // 8) % 2, (j % 8) % 2])

target = tvm.target.create("cuda")
workload = tvm.ir.IRModule({"main": workload})

with tvm.transform.PassContext(config={"tir.inject": True}):
    f = tvm.build(workload, target=target)

print(f.imported_modules[0].get_source())

np.random.seed(913)

m = 4096
n = 4096
k = 4096

dev = tvm.device("cuda", 0)
a_np = np.random.uniform(0, 1, size=(m, k)).astype("float32")
b_np = np.random.uniform(0, 1, size=(k, n)).astype("float32")

a = tvm.nd.array(a_np.astype("float16"), device=dev)
b = tvm.nd.array(b_np.astype("float16"), device=dev)
c = tvm.nd.array(np.zeros((m, n)).astype("float16"), device=dev)

f(a, b, c)

time = f.time_evaluator(f.entry_name, dev, number=10)
print("time: %f ms" % (time(a, b, c).mean * 1000))

# cublas

A_cublas = te.placeholder((m, k), name="A", dtype="float16")
B_cublas = te.placeholder((k, n), name="B", dtype="float16")
C_cublas = cublas.matmul(A_cublas, B_cublas, dtype="float16")
s = te.create_schedule(C_cublas.op)

dev = tvm.cuda(0)
f_cublas = tvm.build(s, [A_cublas, B_cublas, C_cublas], "cuda")

a_cublas = tvm.nd.array(a_np.astype("float16"), dev)
b_cublas = tvm.nd.array(b_np.astype("float16"), dev)
c_cublas = tvm.nd.array(np.zeros((m, n), dtype=C_cublas.dtype), dev)
f_cublas(a_cublas, b_cublas, c_cublas)

print(c.asnumpy())
print(c_cublas.numpy())
# print((a_np @ b_np).astype("float16"))

tvm.testing.assert_allclose(
    c.asnumpy(),
    c_cublas.numpy(),
    rtol=1e-5,
)

tvm.testing.assert_allclose(
    c.asnumpy(),
    (a_np @ b_np).astype("float16"),
    rtol=1e-2,
)