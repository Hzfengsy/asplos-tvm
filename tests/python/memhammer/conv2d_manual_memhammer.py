import tvm
import tvm.topi
import tvm.topi.cuda
import os
import tvm.topi.testing
from tvm import te, tir, topi
from tvm.script import tir as T
import numpy as np
np.set_printoptions(threshold=(1000000))


TASK = "conv2d"
USE_MANUAL_CODE = False

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)


@tvm.register_func
def tvm_callback_cuda_postproc(code):
    # if not os.path.exists("perf"):
    #     os.mkdir("perf")
    # write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("tests/python/memhammer/conv2d.cu").read()
    return code

@T.prim_func
def wmma_load_a_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16,
                       scope="shared")
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16,
                       scope="wmma.matrix_a")

    with T.block("root"):
        T.reads(A[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_load_a_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="shared", strides=[s1, s0])
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a")

    with T.block("root"):
        T.reads(A[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        T.evaluate(T.tvm_load_matrix_sync(
            C.data, 16, 16, 16, C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16), A.access_ptr("r"), s1, "row_major",
            dtype="handle"))


@T.prim_func
def wmma_load_b_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16,
                       scope="shared")
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16,
                       scope="wmma.matrix_b")

    with T.block("root"):
        T.reads(A[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_load_b_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16, scope="shared", strides=[s1, s0])
    C = T.match_buffer(c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b")

    with T.block("root"):
        T.reads(A[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        T.evaluate(T.tvm_load_matrix_sync(
            C.data, 16, 16, 16, C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16), A.access_ptr("r"), s1, "col_major",
            dtype="handle"))

@T.prim_func
def wmma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16,
                       scope="wmma.matrix_a")
    B = T.match_buffer(b, (16, 16), "float16", align=128, offset_factor=16,
                       scope="wmma.matrix_b")
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16,
                       scope="wmma.accumulator")

    with T.block("root"):
        T.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0: 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        for i, j, k in T.grid(16, 16, 16):
            with T.block(""):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                C[vii, vjj] = C[vii, vjj] + T.cast(A[vii, vkk], 'float32') * T.cast(B[vjj, vkk], 'float32')


@T.prim_func
def wmma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float16", align=128, offset_factor=16,
                       scope="wmma.matrix_a")
    B = T.match_buffer(b, (16, 16), "float16", align=128, offset_factor=16,
                       scope="wmma.matrix_b")
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16,
                       scope="wmma.accumulator")

    with T.block("root"):
        T.reads(C[0 : 16, 0 : 16], A[0 : 16, 0 : 16], B[0: 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        T.evaluate(T.tvm_mma_sync(C.data, C.elem_offset // 256,
                                  A.data, A.elem_offset // 256,
                                  B.data, B.elem_offset // 256,
                                  C.data, C.elem_offset // 256, dtype='handle'))


@T.prim_func
def wmma_fill_desc(c: T.handle) -> None:
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator")

    with T.block("root"):
        T.reads()
        T.writes(C[0 : 16, 0 : 16])
        for i, j in T.grid(16, 16):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = T.float32(0)


@T.prim_func
def wmma_fill_impl(c: T.handle) -> None:
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator")
    with T.block("root"):
        T.reads()
        T.writes(C[0 : 16, 0 : 16])
        T.evaluate(T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16), T.float32(0), dtype="handle"))


@T.prim_func
def wmma_store_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator")
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="global")
    with T.block("root"):
        T.reads(A[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_store_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(a, (16, 16), "float32", align=128, offset_factor=16, scope="wmma.accumulator")
    C = T.match_buffer(c, (16, 16), "float32", align=128, offset_factor=16, scope="global", strides=[s1, s0])
    with T.block("root"):
        T.reads(A[0 : 16, 0 : 16])
        T.writes(C[0 : 16, 0 : 16])
        T.evaluate(T.tvm_store_matrix_sync(
            A.data, 16, 16, 16, A.elem_offset // 256 + T.floordiv(T.floormod(A.elem_offset, 256), 16), C.access_ptr("w"), s1, "row_major",
            dtype="handle"))



tir.TensorIntrin.register("wmma.load_matrix_a", wmma_load_a_desc, wmma_load_a_impl)
tir.TensorIntrin.register("wmma.load_matrix_b", wmma_load_b_desc, wmma_load_b_impl)
tir.TensorIntrin.register("wmma.mma_sync", wmma_sync_desc, wmma_sync_impl)
tir.TensorIntrin.register("wmma.fill", wmma_fill_desc, wmma_fill_impl)
tir.TensorIntrin.register("wmma.store", wmma_store_desc, wmma_store_impl)


N = 1
I = O = 64
# padding = 0
H = W = 16
KH = KW = 3

PADDED_H = H + KH // 2 * 2
PADDED_W = W + KW // 2 * 2

def te_workload():
    X = te.placeholder(shape=(N, H, W, I), dtype="float16", name='data')
    Weight = te.placeholder(shape=(O, KW, KW, I), dtype="float16", name='weight')
    pad_input = topi.nn.pad(X, [0, KH//2, KW//2, 0])

    im2col_data = te.compute(
        (N * H * W, KH * KW * I),
        lambda i, j: pad_input[i // (H * W), (i % (H * W)) // W + j // (KW * I), (i % (H * W)) % W + (j % (KW * I)) // I,
                       j % I],
        name='im2col_data',
    )
    im2col_weight = te.compute(
        (O, KH * KW * I),
        lambda i, j: Weight[i, j // (KW * I), (j % (KW * I)) // I, (j % (KW * I)) % I],
        name='im2col_weight',
    )
    k = te.reduce_axis((0, KH * KW * I))
    conv = te.compute((N, H, W, O), lambda n, h, w, o: tir.sum(im2col_data[n * (H * W) + h * W + w, k].astype('float32') * im2col_weight[o, k].astype('float32'), axis=k), name='conv2d')
    s = te.create_schedule(conv.op)
    return te.create_prim_func([X, Weight, conv]), [X, Weight, conv]

f, args_ = te_workload()
s = tir.Schedule(f)
print(s.mod.script())
conv = s.get_block('conv2d')
n, h, w, o, k = s.get_loops(conv)
w, w_tc = s.split(w, [None, 16])
o, o_tc = s.split(o, [None, 16])
k, k_tc = s.split(k, [None, 16])
s.reorder(w, o, k, w_tc, o_tc, k_tc)
gemm_inner = conv
gemm = s.blockize(w_tc)


n_factors = [1, 1, 1, 1, 1]
h_factors = [4, 1, 2, 2, 1]
w_factors = [1, 1, 1, 1, 1]
o_factors = [1, 2, 2, 1, 1]
k_factors = [18, 2, 1]

n0, n1, n2, n3, n4 = s.split(n, n_factors)
h0, h1, h2, h3, h4 = s.split(h, h_factors)
w0, w1, w2, w3, w4 = s.split(w, w_factors)
o0, o1, o2, o3, o4 = s.split(o, o_factors)
k0, k1, k2 = s.split(k, k_factors)
s.reorder(
    n0, h0, w0, o0,
    n1, h1, w1, o1,
    n2, h2, w2, o2,
    k0,
    k1,
    n3, h3, w3, o3,
    k2,
    n4, h4, w4, o4
)

bx = s.fuse(n0, h0, w0, o0)
s.bind(bx, 'blockIdx.x')
by = s.fuse(n1, h1, w1, o1)
s.bind(by, 'blockIdx.y')
ty = s.fuse(n2, h2, w2, o2)
s.bind(ty, 'threadIdx.y')
# s.annotate(k0, 'software_pipeline_stage', [0,0,0,0,0,1,1])
# s.annotate(k0, 'software_pipeline_order', [0,3,1,4,5,2,6])
# s.annotate(k1, 'software_pipeline_stage', [0, 0, 1])
# s.annotate(k1, 'software_pipeline_order', [0, 1, 2])
block_shared_a = s.read_at(k0, gemm, 1, "shared")
s.annotate(block_shared_a, "local_stage", True)
s.annotate(block_shared_a,"vector_bytes", 16)
# s.annotate(block_shared_a,"double_buffer_scope", 0)
block_shared_b = s.read_at(k0, gemm, 2, "shared")
s.annotate(block_shared_b, "local_stage", True)
s.annotate(block_shared_b,"vector_bytes", 16)
# s.annotate(block_shared_b,"double_buffer_scope", 0)
block_wmma_a = s.read_at(k1, gemm, 1, "wmma.matrix_a")
block_wmma_b = s.read_at(k1, gemm, 2, "wmma.matrix_b")
cache_write = s.write_at(ty, gemm, 0, 'wmma.accumulator')
s.annotate(cache_write,"vector_bytes", 16)
s.compute_inline(s.get_block('im2col_data'))
s.compute_inline(s.get_block('im2col_weight'))
s.compute_inline(s.get_block('PadInput'))


init_block = s.decompose_reduction(gemm, k0)
init_inner = s.get_child_blocks(init_block)[0]
s.tensorize(s.get_loops(init_inner)[-2], 'wmma.fill')
s.tensorize(gemm, 'wmma.mma_sync')
root_block = s.get_block("root")
s.annotate(root_block, "warp_execution", True)

dev = tvm.device("cuda", 0)

a_np = np.random.rand(N, H , W, I)
b_np = np.random.rand(O, KW, KW, I)
b_np_transpose = b_np.transpose(1, 2, 3, 0)  # OHWI -> HWIO
c_np = tvm.topi.testing.conv2d_nhwc_python(a_np, b_np_transpose, 1, KH //2).astype('float32')
a_tvm = tvm.nd.array(a_np.astype('float16'), device=dev)
b_tvm = tvm.nd.array(b_np.astype('float16'), device=dev)
c_tvm = tvm.nd.empty(c_np.shape, device=dev)

print(s.mod['main'].script())
print(tvm.lower(s.mod['main'], simple_mode=True))

with tvm.target.cuda():
    f = tvm.build(s.mod['main'])
    print(f.imported_modules[0].get_source())

    f(a_tvm, b_tvm, c_tvm)

    tvm.testing.assert_allclose(c_tvm.numpy(), c_np, atol=1e-3, rtol=1e-3)

    evaluator = f.time_evaluator(f.entry_name, dev, number=1000)
    gflops = (N*H*W*O*KH*KW*I) * 2 / 1e9
    time_ms = evaluator(a_tvm, b_tvm, c_tvm).mean * 1e3
    print("conv2d with tensor core: %f ms, %f GFLOPS" % (time_ms, gflops / (time_ms / 1e3)))


