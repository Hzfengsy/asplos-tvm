from tvm import tir
from tvm.script import tir as T


@T.prim_func
def wmma_load_a_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(
        a, (16, 16), "float16", align=128, offset_factor=16, scope="shared.dyn"
    )
    C = T.match_buffer(
        c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_load_a_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(
        a,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="shared.dyn",
        strides=[s1, s0],
    )
    C = T.match_buffer(
        c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.tvm_load_matrix_sync(
                C.data,
                16,
                16,
                16,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                A.access_ptr("r"),
                s1,
                "row_major",
                dtype="handle",
            )
        )


@T.prim_func
def wmma_load_b_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(
        a, (16, 16), "float16", align=128, offset_factor=16, scope="shared.dyn"
    )
    C = T.match_buffer(
        c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for i, j in T.grid(16, 16):
            with T.block("load"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_load_b_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(
        a,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="shared.dyn",
        strides=[s1, s0],
    )
    C = T.match_buffer(
        c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )

    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.tvm_load_matrix_sync(
                C.data,
                16,
                16,
                16,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                A.access_ptr("r"),
                s1,
                "col_major",
                dtype="handle",
            )
        )


@T.prim_func
def wmma_sync_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(
        a, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )
    B = T.match_buffer(
        b, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    C = T.match_buffer(
        c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )

    with T.block("root"):
        T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for i, j, k in T.grid(16, 16, 16):
            with T.block(""):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]


@T.prim_func
def wmma_sync_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(
        a, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_a"
    )
    B = T.match_buffer(
        b, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.matrix_b"
    )
    C = T.match_buffer(
        c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )

    with T.block("root"):
        T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.tvm_mma_sync(
                C.data,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                A.data,
                A.elem_offset // 256 + T.floordiv(T.floormod(A.elem_offset, 256), 16),
                B.data,
                B.elem_offset // 256 + T.floordiv(T.floormod(B.elem_offset, 256), 16),
                C.data,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                dtype="handle",
            )
        )


@T.prim_func
def wmma_fill_desc(c: T.handle) -> None:
    C = T.match_buffer(
        c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )

    with T.block("root"):
        T.reads()
        T.writes(C[0:16, 0:16])
        for i, j in T.grid(16, 16):
            with T.block("init"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = T.float16(0)


@T.prim_func
def wmma_fill_impl(c: T.handle) -> None:
    C = T.match_buffer(
        c, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    with T.block("root"):
        T.reads()
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.tvm_fill_fragment(
                C.data,
                16,
                16,
                16,
                C.elem_offset // 256 + T.floordiv(T.floormod(C.elem_offset, 256), 16),
                T.float16(0),
                dtype="handle",
            )
        )


@T.prim_func
def wmma_store_desc(a: T.handle, c: T.handle) -> None:
    A = T.match_buffer(
        a, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(
        c, (16, 16), "float16", align=128, offset_factor=16, scope="shared.dyn"
    )
    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for i, j in T.grid(16, 16):
            with T.block("store"):
                vii, vjj = T.axis.remap("SS", [i, j])
                C[vii, vjj] = A[vii, vjj]


@T.prim_func
def wmma_store_impl(a: T.handle, c: T.handle) -> None:
    s1 = T.var("int32")
    s0 = T.var("int32")
    A = T.match_buffer(
        a, (16, 16), "float16", align=128, offset_factor=16, scope="wmma.accumulator"
    )
    C = T.match_buffer(
        c,
        (16, 16),
        "float16",
        align=128,
        offset_factor=16,
        scope="shared.dyn",
        strides=[s1, s0],
    )
    with T.block("root"):
        T.reads(A[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.tvm_store_matrix_sync(
                A.data,
                16,
                16,
                16,
                A.elem_offset // 256 + T.floordiv(T.floormod(A.elem_offset, 256), 16),
                C.access_ptr("w"),
                s1,
                "row_major",
                dtype="handle",
            )
        )


tir.TensorIntrin.register("wmma_load_a", wmma_load_a_desc, wmma_load_a_impl)
tir.TensorIntrin.register("wmma_load_b", wmma_load_b_desc, wmma_load_b_impl)
tir.TensorIntrin.register("wmma_sync", wmma_sync_desc, wmma_sync_impl)
tir.TensorIntrin.register("wmma_fill", wmma_fill_desc, wmma_fill_impl)
tir.TensorIntrin.register("wmma_store", wmma_store_desc, wmma_store_impl)
