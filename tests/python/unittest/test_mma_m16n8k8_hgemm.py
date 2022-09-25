import numpy as np

import tvm
import tvm.testing
from tvm import te
from tvm.script import tir as T
from tvm import meta_schedule as ms
from tvm._ffi import register_func
from tvm import tir
from tvm.script import tir as T
from tvm.runtime import convert
from tvm.contrib import cublas

VEC_SIZE = 8


def get_index_A(elem_offset, stride):
    i = elem_offset // stride
    j = elem_offset % stride
    stride_b = stride // 8
    bi = i // 32
    bj = j // 8
    no = bi * stride_b + bj
    return no * 8 + (i % 32) // 16 * 4


def get_index_B(elem_offset, stride):
    i = elem_offset // stride
    j = elem_offset % stride
    stride_b = stride // 32
    bi = i // 8
    bj = j // 32
    no = bi * stride_b + bj
    return no * 8 + (j % 32) // 8 * 2


def get_index_C(elem_offset, stride):
    i = elem_offset // stride
    j = elem_offset % stride
    stride_b = stride // 8
    bi = i // 8
    bj = j // 8
    return ((bi // 2) * 2 * stride_b + bi % 2 + bj * 2)


@T.prim_func
def m16n8k8_load_A_row_major_desc(a: T.handle, c: T.handle) -> None:
    src = T.match_buffer(
        a, (32, 8), "float16", align=128, offset_factor=1, scope="shared.dyn"
    )
    dst = T.match_buffer(
        c, (32, 8), "float16", align=128, offset_factor=1, scope="m16n8k8.matrixA"
    )

    with T.block("root"):
        T.reads(src[0:32, 0:8])
        T.writes(dst[0:32, 0:8])
        for i, j in T.grid(32, 8):
            with T.block("m16n8k8_load_A"):
                vi, vj = T.axis.remap("SS", [i, j])
                dst[vi, vj] = src[vi, vj]


@T.prim_func
def m16n8k8_load_A_row_major_impl(a: T.handle, c: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    src = T.match_buffer(
        a, (32, 8), "float16", align=128, offset_factor=1, scope="shared.dyn",
        strides=[s0, s1]
    )

    d0 = T.var("int32")
    d1 = T.var("int32")
    dst = T.match_buffer(
        c, (32, 8), "float16", align=128, offset_factor=1, scope="m16n8k8.matrixA",
        strides=[d0, d1]
    )
 
    with T.block("root"):
        T.reads(src[0:32, 0:8])
        T.writes(dst[0:32, 0:8])

        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 32)

        T.evaluate(
                T.ptx_ldmatrix(
                    False, # trans 
                    4, # Always load 4 matrices
                    ".b16",
                    dst.data,
                    get_index_A(dst.elem_offset, d0),
                    src.access_ptr("r"),
                    tx * s0,
                    dtype="float16"
                )
        )


@T.prim_func
def m16n8k8_load_B_row_major_desc(a: T.handle, c: T.handle) -> None:
    src = T.match_buffer(
        a, (8, 32), "float16", align=128, offset_factor=1, scope="shared.dyn"
    )
    dst = T.match_buffer(
        c, (8, 32), "float16", align=128, offset_factor=1, scope="m16n8k8.matrixB"
    )

    with T.block("root"):
        T.reads(src[0:8, 0:32])
        T.writes(dst[0:8, 0:32])
        for i, j in T.grid(8, 32):
            with T.block("m16n8k8_load_B"):
                vi, vj = T.axis.remap("SS", [i, j])
                dst[vi, vj] = src[vi, vj]


@T.prim_func
def m16n8k8_load_B_row_major_impl(a: T.handle, c: T.handle) -> None:
    s0 = T.var("int32")
    s1 = T.var("int32")
    src = T.match_buffer(
        a, (8, 32), "float16", align=128, offset_factor=1, scope="shared.dyn",
        strides=[s0, s1]
    )
    d0 = T.var("int32")
    d1 = T.var("int32")
    dst = T.match_buffer(
        c, (8, 32), "float16", align=128, offset_factor=1, scope="m16n8k8.matrixB",
        strides=[d0, d1]
    )

    with T.block("root"):
        T.reads(src[0:8, 0:32])
        T.writes(dst[0:8, 0:32])

        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 32)

        T.evaluate(
                T.ptx_ldmatrix(
                    True, # trans 
                    4, # Always load 4 matrices
                    ".b16",
                    dst.data,
                    get_index_B(dst.elem_offset, d0),
                    src.access_ptr("r"),
                    s0 * (tx % 8) + 8 * (tx // 8),
                    dtype="float16"
                )
        )


@T.prim_func
def m16n8k8_store_C_row_major_desc(a: T.handle, c: T.handle) -> None:
    src = T.match_buffer(
        a, (8, 8), "float16", align=128, offset_factor=1, scope="m16n8k8.matrixC"
    )
    dst = T.match_buffer(
        c, (8, 8), "float16", align=128, offset_factor=1, scope="shared.dyn"
    )

    with T.block("root"):
        T.reads(src[0:8, 0:8])
        T.writes(dst[0:8, 0:8])
        for i, j in T.grid(8, 8):
            with T.block("m16n8k8_store"):
                vi, vj = T.axis.remap("SS", [i, j])
                dst[vi, vj] = src[vi, vj]


@T.prim_func
def m16n8k8_store_C_row_major_impl(a: T.handle, c: T.handle) -> None:
    src = T.match_buffer(
        a, (8, 8), "float16", align=128, offset_factor=1, scope="m16n8k8.matrixC"
    )
    dst = T.match_buffer(
        c, (8, 8), "float16", align=128, offset_factor=1, scope="shared.dyn"
    )

    with T.block("root"):
        T.reads(src[0:8, 0:8])
        T.writes(dst[0:8, 0:8])

        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 32)

        for i in T.vectorized(2):
            dst[tx // 4, tx % 4 * 2 + i] = src[tx // 4, tx % 4 * 2 + i]


@T.prim_func
def m16n8k8_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(
        a, (16, 8), "float16", align=128, offset_factor=1, scope="m16n8k8.matrixA"
    )
    B = T.match_buffer(
        b, (8, 8), "float16", align=128, offset_factor=1, scope="m16n8k8.matrixB"
    )
    C = T.match_buffer(
        c, (16, 8), "float16", align=128, offset_factor=1, scope="m16n8k8.matrixC"
    )

    with T.block("root"):
        T.reads(C[0:16, 0:8], A[0:16, 0:8], B[0:8, 0:8])
        T.writes(C[0:16, 0:8])
        for i, j, k in T.grid(16, 8, 8):
            with T.block("m16n8k8_sync"):
                vi, vj, vkk = T.axis.remap("SSR", [i, j, k])
                C[vi, vj] = C[vi, vj] + A[vi, vkk] * B[vkk, vj]


@T.prim_func
def m16n8k8_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    a0 = T.var("int32")
    a1 = T.var("int32")
    A = T.match_buffer(
        a, (16, 8), "float16", align=128, offset_factor=1, scope="m16n8k8.matrixA",
        strides=[a0, a1]
    )
    b0 = T.var("int32")
    b1 = T.var("int32")
    B = T.match_buffer(
        b, (8, 8), "float16", align=128, offset_factor=1, scope="m16n8k8.matrixB",
        strides=[b0, b1]
    )
    c0 = T.var("int32")
    c1 = T.var("int32")
    C = T.match_buffer(
        c, (16, 8), "float16", align=128, offset_factor=1, scope="m16n8k8.matrixC",
        strides=[c0, c1]
    )

    with T.block("root"):
        T.reads(C[0:16, 0:8], A[0:16, 0:8], B[0:8, 0:8])
        T.writes(C[0:16, 0:8])
        T.evaluate(
            T.ptx_mma(
                "m16n8k8",
                "row",
                "col",
                "fp16",
                "fp16",
                "fp16",
                A.data,
                get_index_A(A.elem_offset, a0),
                B.data,
                get_index_B(B.elem_offset, b0),
                C.data,
                get_index_C(C.elem_offset, c0),
                False,
                dtype="float16",
            )
        )


@T.prim_func
def m16n8k8_init_desc(c: T.handle) -> None:
    dst = T.match_buffer(
        c, (8, 8), "float16", align=128, offset_factor=1, scope="m16n8k8.matrixC"
    )

    with T.block("root"):
        T.reads()
        T.writes(dst[0:8, 0:8])
        for i, j in T.grid(8, 8):
            with T.block("m16n8k8_store"):
                vi, vj = T.axis.remap("SS", [i, j])
                dst[vi, vj] = T.float16(0)


@T.prim_func
def m16n8k8_init_impl(c: T.handle) -> None:
    dst = T.match_buffer(
        c, (8, 8), "float16", align=128, offset_factor=1, scope="m16n8k8.matrixC"
    )

    with T.block("root"):
        T.reads()
        T.writes(dst[0:8, 0:8])

        tx = T.env_thread("threadIdx.x")
        T.launch_thread(tx, 32)

        for i in T.vectorized(2):
            dst[tx // 4, tx % 4 * 2 + i] = T.float16(0)


tir.TensorIntrin.register("m16n8k8_load_A_row_major", m16n8k8_load_A_row_major_desc, m16n8k8_load_A_row_major_impl)
tir.TensorIntrin.register("m16n8k8_load_B_row_major", m16n8k8_load_B_row_major_desc, m16n8k8_load_B_row_major_impl)
tir.TensorIntrin.register("m16n8k8_store_C_row_major", m16n8k8_store_C_row_major_desc, m16n8k8_store_C_row_major_impl)
tir.TensorIntrin.register("m16n8k8_sync", m16n8k8_sync_desc, m16n8k8_sync_impl)
tir.TensorIntrin.register("m16n8k8_init", m16n8k8_init_desc, m16n8k8_init_impl)


@register_func("tir.index_map_m16n8k8.matrixC")
def index_map_m16n8k8_matrixC(ind):
    i, j = ind[0], ind[1]
    return convert([(i // 8) // 2, j // 8, (i // 8) % 2, (j % 8) % 2])


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
sch = tir.Schedule(workload)

block = sch.get_block("C")
i, j, k = sch.get_loops(block)

# Step 1. Rule-Auto-Tensorize
# pylint: disable=invalid-name
i, i_tc = sch.split(i, factors=[None, 16])
j, j_tc = sch.split(j, factors=[None, 8])
k, k_tc = sch.split(k, factors=[None, 8])
sch.reorder(
    # fmt: off
    i, j, k,
    # tensor core
    i_tc, j_tc, k_tc,
    # fmt: on
)
block_inner = sch.blockize(i_tc)
block_outer, block_inner = block_inner, block
del block


# Step 2. Rule-Multi-Level-Tiling
i_factors = [1, 32, 2, 1, 4]
j_factors = [4, 8, 2, 1, 8]  # swizzle: identity8
k_factors = [128, 4, 1]
i0, i1, i2, i3, i4 = sch.split(i, factors=i_factors)
j0, j1, j2, j3, j4 = sch.split(j, factors=j_factors)
k0, k1, k2 = sch.split(k, k_factors)
# pylint: enable=invalid-name
sch.reorder(
    # fmt: off
    i0, j0,   # S => blockIdx.x
    i1, j1,   # S => blockIdx.y
    j2, i2,   # S => threadIdx.y
    # cache_write here
    k0,       # R
    # vectorized cooperative fetching here
    k1,       # R
    i3, j3,   # S
    k2,       # R
    i4, j4,
    # S
    # fmt: on
)
block_idx = sch.fuse(i0, j0)
block_idy = sch.fuse(i1, j1)
thread_idy = sch.fuse(j2, i2)
sch.bind(block_idx, "blockIdx.y")
sch.bind(block_idy, "blockIdx.x")
sch.bind(thread_idy, "threadIdx.y")

b24 = sch.get_block(name="root", func_name="main")
sch.annotate(block_or_loop=b24, ann_key="warp_execution", ann_val=1)

num_ty = i_factors[2] * j_factors[2]
def fetch_to_shared(block, idx, ndim):
    vector_size = 16
    block_read = sch.read_at(k0, block, idx, "shared.dyn")
    sch.annotate(block_or_loop=block_read, ann_key="vector_bytes", ann_val=vector_size)
    sch.annotate(block_or_loop=block_read, ann_key="local_stage", ann_val=1)
    sch.annotate(block_or_loop=block_read, ann_key="double_buffer_scope", ann_val=0)
    sch.annotate(block_or_loop=block_read, ann_key="meta_schedule.cache_type", ann_val=0)

fetch_to_shared(block_outer, 0, 2)
fetch_to_shared(block_outer, 1, 2)

# Step 3. Postproc-Rewrite-Tensorize

# Step 3.1. Cache read
loop = sch.get_loops(block_outer)[-1]
block_read_a = sch.cache_read(block_outer, 0, "m16n8k8.matrixA")
block_read_b = sch.cache_read(block_outer, 1, "m16n8k8.matrixB")
sch.compute_at(block_read_a, k1)
sch.compute_at(block_read_b, k1)

l0, l1 = sch.get_loops(block_read_a)[-2:]
l00, l01 = sch.split(l0, [None, 32])
l10, l11 = sch.split(l1, [None, 8])
sch.reorder(l00, l10, l01, l11)
sch.tensorize(l01, "m16n8k8_load_A_row_major")

l0, l1 = sch.get_loops(block_read_b)[-2:]
l00, l01 = sch.split(l0, [None, 8])
l10, l11 = sch.split(l1, [None, 32])
sch.reorder(l00, l10, l01, l11)
sch.tensorize(l01, "m16n8k8_load_B_row_major")

# block_read_a = sch.read_at(k1, block_outer, 0, "m16n8k8.matrixA")
# block_read_b = sch.read_at(k1, block_outer, 1, "m16n8k8.matrixB")

# Step 3.2. Cache write

block_write_c = sch.write_at(thread_idy, block_outer, 0, "m16n8k8.matrixC")

sch.annotate(block_or_loop=k1, ann_key="software_pipeline_stage", ann_val=[0, 0, 1])
sch.annotate(block_or_loop=k1, ann_key="software_pipeline_order", ann_val=[0, 1, 2])
sch.annotate(block_or_loop=k0, ann_key="software_pipeline_stage", ann_val=[0, 0, 0, 0, 0, 1, 1])
sch.annotate(block_or_loop=k0, ann_key="software_pipeline_order", ann_val=[0, 3, 1, 4, 5, 2, 6])

# Step 3.3. Decompose
loop = sch.get_loops(block_outer)[3]
block_init_c = sch.decompose_reduction(block_outer, loop)
block_init_c_inner = sch.get_child_blocks(block_init_c)[0]
# Step 3.4. Tensorize
loop = sch.get_loops(block_inner)[-3]
sch.tensorize(loop, "m16n8k8_sync")

loop = sch.get_loops(block_init_c_inner)[-2]
l0, l1 = sch.split(loop, [None, 8])
sch.tensorize(l1, "m16n8k8_init")

# print(sch.mod.script())
# print(tvm.lower(sch.mod).script())

target = tvm.target.Target("nvidia/geforce-rtx-3090")
f = tvm.build(sch.mod, target=target)
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

tvm.testing.assert_allclose(
    c_cublas.numpy(),
    c.asnumpy(),
    rtol=1e-2,
)