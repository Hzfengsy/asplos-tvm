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
# pylint: disable=invalid-name,missing-function-docstring
"""Intrinsics for ARM tensorization."""
from tvm.script import tir as T
from .. import TensorIntrin


# TODO(masahi): Parametrize the TVMScript description of dot product by
# shape and dtype, and share the common description with x86.


@T.prim_func
def dot_product_4x4_i8i8i32_desc(
    A: T.Buffer((4,), "int8", offset_factor=1),
    B: T.Buffer((4, 4), "int8", offset_factor=1),
    C: T.Buffer((4,), "int32", offset_factor=1),
) -> None:
    with T.block("root"):
        T.reads(C[0:4], A[0:4], B[0:4, 0:4])
        T.writes(C[0:4])
        for i in T.serial(0, 4):
            with T.init():
                C[i] = T.int32(0)
            for k in T.serial(0, 4):
                with T.block("update"):
                    vi, vk = T.axis.remap("SR", [i, k])
                    C[vi] = C[vi] + T.cast(A[vk], "int32") * T.cast(B[vi, vk], "int32")
                    
@T.prim_func
def dot_product_12x8_i8i8i32_desc(
    a: T.handle, b: T.handle, c: T.handle
) -> None:
    A = T.match_buffer(a, (12, 512), "int8", offset_factor=1, scope="global")
    B = T.match_buffer(b, (8, 512), "int8", offset_factor=1, scope="global")
    C = T.match_buffer(c, (12, 8), "int32", offset_factor=1, scope="global")
    with T.block("root"):
        T.reads(C[0:12, 0:8], A[0:12, 0:512], B[0:8, 0:512])
        T.writes(C[0:12, 0:8])
        for i in T.serial(0, 12):
            for j in T.serial(0, 8):
                for k in T.serial(0, 512):
                    with T.block("update"):
                        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                        C[vi, vj] = C[vi, vj] +T.cast(A[vi, vk],"int32") * T.cast(B[vj, vk],"int32")                   

@T.prim_func
def dot_product_12x8_f32f32f32_desc(
    a: T.handle, b: T.handle, c: T.handle
) -> None:
    A = T.match_buffer(a, (4, 12), "float32", offset_factor=1, scope="global")
    B = T.match_buffer(b, (4, 8), "float32", offset_factor=1, scope="global")
    C = T.match_buffer(c, (12, 8), "float32", offset_factor=1, scope="global")
    with T.block("root"):
        T.reads(C[0:12, 0:8], A[0:4, 0:12], B[0:4, 0:8])
        T.writes(C[0:12, 0:8])
        for i in T.serial(0, 12):
            for j in T.serial(0, 8):
                for k in T.serial(0, 4):
                    with T.block("update"):
                        vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                        C[vi, vj] = C[vi, vj] +A[vk, vi] * B[vk, vj]


@T.prim_func
def dot_product_4x4_i8i8i32_neon(
    A: T.Buffer((4,), "int8", offset_factor=1),
    B: T.Buffer((4, 4), "int8", offset_factor=1),
    C: T.Buffer((4,), "int32", offset_factor=1),
) -> None:
    with T.block("root"):
        T.reads(C[0:4], A[0:4], B[0:4, 0:4])
        T.writes(C[0:4])

        A_int8 = A.vload([0], "int8x4")
        re_int32 = T.reinterpret(A_int8, dtype="int32")
        vec_ai32 = T.broadcast(re_int32, 2)
        vec_a = T.reinterpret(vec_ai32, dtype="int8x8")

        vec_b = B.vload([0, 0], dtype="int8x16")

        # TODO(masahi): Remove duplication when inlined function call is supported
        vec_b_low = T.vectorlow(vec_b, dtype="int8x8")

        multiply_low = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.smull.v8i16"),
            T.uint32(2),
            vec_a,
            vec_b_low,
            dtype="int16x8",
        )

        pairwise_reduction_low = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.saddlp.v4i32.v8i16"),
            T.uint32(1),
            multiply_low,
            dtype="int32x4",
        )

        vec_b_high = T.vectorhigh(vec_b, dtype="int8x8")

        multiply_high = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.smull.v8i16"),
            T.uint32(2),
            vec_a,
            vec_b_high,
            dtype="int16x8",
        )

        pairwise_reduction_high = T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.saddlp.v4i32.v8i16"),
            T.uint32(1),
            multiply_high,
            dtype="int32x4",
        )

        C[T.ramp(T.int32(0), 1, 4)] += T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.addp.v4i32"),
            T.uint32(2),
            pairwise_reduction_low,
            pairwise_reduction_high,
            dtype="int32x4",
        )


@T.prim_func
def dot_product_4x4_i8i8i32_sdot(
    A: T.Buffer((4,), "int8", offset_factor=1),
    B: T.Buffer((4, 4), "int8", offset_factor=1),
    C: T.Buffer((4,), "int32", offset_factor=1),
) -> None:
    with T.block("root"):
        T.reads(C[0:4], A[0:4], B[0:4, 0:4])
        T.writes(C[0:4])

        A_i8x4 = A.vload([0], "int8x4")
        A_i32 = T.reinterpret(A_i8x4, dtype="int32")
        vec_ai32 = T.broadcast(A_i32, 4)
        vec_a = T.reinterpret(vec_ai32, dtype="int8x16")

        vec_b = B.vload([0, 0], dtype="int8x16")

        C[T.ramp(T.int32(0), 1, 4)] += T.call_llvm_pure_intrin(
            T.llvm_lookup_intrinsic_id("llvm.aarch64.neon.sdot.v4i32.v16i8"),
            T.uint32(3),
            T.int32x4(0),
            vec_a,
            vec_b,
            dtype="int32x4",
        )


@T.prim_func
def dot_product_12x8_f32f32f32_microkernel(
    a: T.handle, b: T.handle, c: T.handle
) -> None:
    A = T.match_buffer(a, (4, 12), "float32", offset_factor=1, scope="global")
    B = T.match_buffer(b, (4, 8), "float32", offset_factor=1, scope="global")
    C = T.match_buffer(c, (12, 8), "float32", offset_factor=1, scope="global")
    with T.block("root"):
        T.reads(C[0:12, 0:8], A[0:4, 0:12], B[0:4, 0:8])
        T.writes(C[0:12, 0:8])
        T.evaluate(
            T.call_extern(
                "Run",
                A.access_ptr("r"),
                B.access_ptr("r"),
                C.access_ptr("w"),
                4,
                dtype="handle"
            )
        )


@T.prim_func
def dot_product_8x12_i8i8i32_desc(
    a: T.handle, b: T.handle, c: T.handle
) -> None:
    A = T.match_buffer(a, (128, 8, 4), "int8", offset_factor=1, scope="global")
    B = T.match_buffer(b, (128, 12, 4), "int8", offset_factor=1, scope="global")
    C = T.match_buffer(c, (8, 12), "int32", offset_factor=1, scope="global")
    with T.block("root"):
        T.reads(C[0:8, 0:12], A[0:128, 0:8, 0:4], B[0:128, 0:12, 0:4])
        T.writes(C[0:8, 0:12])
        for ko in T.serial(0, 128):
            for i in T.serial(0, 8):
                for j in T.serial(0, 12):
                    for ki in T.serial(0, 4):
                        with T.block("update"):
                            vko, vi, vj, vki = T.axis.remap("RSSR", [ko, i, j, ki])
                            C[vi, vj] = C[vi, vj] + T.cast(A[vko, vi, vki], "int32") * T.cast(B[vko, vj, vki], "int32")

@T.prim_func
def dot_product_8x12_i8i8i32_microkernel(
    a: T.handle, b: T.handle, c: T.handle
) -> None:
    A = T.match_buffer(a, (128, 8, 4), "int8", offset_factor=1, scope="global")
    B = T.match_buffer(b, (128, 12, 4), "int8", offset_factor=1, scope="global")
    C = T.match_buffer(c, (8, 12), "int32", offset_factor=1, scope="global")
    with T.block("root"):
        T.reads(C[0:8, 0:12], A[0:128, 0:8, 0:4], B[0:128, 0:12, 0:4])
        T.writes(C[0:8, 0:12])
        T.evaluate(
            T.call_extern(
                "a64_gemm_s8_8x12",
                A.access_ptr("r"),
                B.access_ptr("r"),
                C.access_ptr("w"),
                dtype="handle"
            )
        )


@T.prim_func
def dot_product_8x12x4_i8i8i32_desc(
    a: T.handle, b: T.handle, c: T.handle
) -> None:
    A = T.match_buffer(a, (8, 4), "int8", offset_factor=1, scope="global")
    B = T.match_buffer(b, (12, 4), "int8", offset_factor=1, scope="global")
    C = T.match_buffer(c, (8, 12), "int32", offset_factor=1, scope="global")
    with T.block("root"):
        T.reads(C[0:8, 0:12], A[0:8, 0:4], B[0:12, 0:4])
        T.writes(C[0:8, 0:12])
        for i in T.serial(0, 8):
            for j in T.serial(0, 12):
                for ki in T.serial(0, 4):
                    with T.block("update"):
                        vi, vj, vki = T.axis.remap("SSR", [i, j, ki])
                        C[vi, vj] = C[vi, vj] + T.cast(A[vi, vki], "int32") * T.cast(B[vj, vki], "int32")

@T.prim_func
def dot_product_8x12x4_i8i8i32_microkernel(
    a: T.handle, b: T.handle, c: T.handle
) -> None:
    A = T.match_buffer(a, (8, 4), "int8", offset_factor=1, scope="global")
    B = T.match_buffer(b, (12, 4), "int8", offset_factor=1, scope="global")
    C = T.match_buffer(c, (8, 12), "int32", offset_factor=1, scope="global")
    with T.block("root"):
        T.reads(C[0:8, 0:12], A[0:8, 0:4], B[0:12, 0:4])
        T.writes(C[0:8, 0:12])
        T.evaluate(
            T.call_extern(
                "a64_gemm_s8_8x12",
                A.access_ptr("r"),
                B.access_ptr("r"),
                C.access_ptr("w"),
                dtype="handle"
            )
        )


@T.prim_func
def dot_product_8x12x16_i8i8i32_desc(
    a: T.handle, b: T.handle, c: T.handle, k: T.int32
) -> None:
    A = T.match_buffer(a, (k, 8, 4), "int8", offset_factor=1, scope="global")
    B = T.match_buffer(b, (k, 12, 4), "int8", offset_factor=1, scope="global")
    C = T.match_buffer(c, (8, 12), "int32", offset_factor=1, scope="global")
    with T.block("root"):
        T.reads(C[0:8, 0:12], A[0:k, 0:8, 0:4], B[0:k, 0:12, 0:4])
        T.writes(C[0:8, 0:12])
        for ko in T.serial(0, k):
            for i in T.serial(0, 8):
                for j in T.serial(0, 12):
                    for ki in T.serial(0, 4):
                        with T.block("update"):
                            vko, vi, vj, vki = T.axis.remap("RSSR", [ko, i, j, ki])
                            C[vi, vj] = C[vi, vj] + T.cast(A[vko, vi, vki], "int32") * T.cast(B[vko, vj, vki], "int32")


@T.prim_func
def dot_product_8x12x16_i8i8i32_microkernel(
    a: T.handle, b: T.handle, c: T.handle, k: T.int32
) -> None:
    A = T.match_buffer(a, (k, 8, 4), "int8", offset_factor=1, scope="global")
    B = T.match_buffer(b, (k, 12, 4), "int8", offset_factor=1, scope="global")
    C = T.match_buffer(c, (8, 12), "int32", offset_factor=1, scope="global")
    with T.block("root"):
        T.reads(C[0:8, 0:12], A[0:k, 0:8, 0:4], B[0:k, 0:12, 0:4])
        T.writes(C[0:8, 0:12])
        T.evaluate(
            T.call_extern(
                "a64_gemm_s8_8x12",
                A.access_ptr("r"),
                B.access_ptr("r"),
                C.access_ptr("w"),
                k*4,
                dtype="handle"
            )
        )
        

@T.prim_func
def dot_product_8x12x16_i8i8i32_fake_microkernel(
    a: T.handle, b: T.handle, c: T.handle
) -> None:
    A = T.match_buffer(a, (8, 4), "int8", offset_factor=1, scope="global")
    B = T.match_buffer(b, (12, 4), "int8", offset_factor=1, scope="global")
    C = T.match_buffer(c, (8, 12), "int32", offset_factor=1, scope="global")
    with T.block("root"):
        T.reads(C[0:8, 0:12], A[0:8, 0:4], B[0:12, 0:4])
        T.writes(C[0:8, 0:12])
        T.evaluate(
            T.call_extern(
                "a64_gemm_s8_8x12",
                A.access_ptr("r"),
                B.access_ptr("r"),
                C.access_ptr("w"),
                dtype="handle"
            )
        )


@T.prim_func
def dot_product_8x12x16_i8i8i32_fake_desc(
    a: T.handle, b: T.handle, c: T.handle
) -> None:
    A = T.match_buffer(a, (8, 4), "int8", offset_factor=1, scope="global")
    B = T.match_buffer(b, (12, 4), "int8", offset_factor=1, scope="global")
    C = T.match_buffer(c, (8, 12), "int32", offset_factor=1, scope="global")
    with T.block("root"):
        T.reads(C[0:8, 0:12], A[0:8, 0:4], B[ 0:12, 0:4])
        T.writes(C[0:8, 0:12])
        for i in T.serial(0, 8):
            for j in T.serial(0, 12):
                for ki in T.serial(0, 4):
                    with T.block("update"):
                        vi, vj, vki = T.axis.remap("SSR", [i, j, ki])
                        C[vi, vj] = C[vi, vj] + T.cast(A[vi, vki], "int32") * T.cast(B[vj, vki], "int32")


ARM_DOT_4x4_i8_NEON_INTRIN = "dot_4x4_i8i8s32_neon"
ARM_DOT_4x4_i8_SDOT_INTRIN = "dot_4x4_i8i8s32_sdot"
ARM_DOT_12x8_fp32_MICROKERNEL_INTRIN = "dot_12x8_f32f32f32_microkernel"
ARM_DOT_8x12_i8_MICROKERNEL_INTRIN = "dot_8x12_i8i8i32_microkernel"
ARM_DOT_8x12x4_i8_MICROKERNEL_INTRIN = "dot_8x12x4_i8i8i32_microkernel"
ARM_DOT_8x12x16_i8_MICROKERNEL_INTRIN = "dot_8x12x16_i8i8i32_microkernel"
ARM_DOT_8x12x16_i8_FAKE_MICROKERNEL_INTRIN = "dot_8x12x16_i8i8i32_fake_microkernel"

TensorIntrin.register(
    ARM_DOT_4x4_i8_NEON_INTRIN, dot_product_4x4_i8i8i32_desc, dot_product_4x4_i8i8i32_neon
)

TensorIntrin.register(
    ARM_DOT_4x4_i8_SDOT_INTRIN, dot_product_4x4_i8i8i32_desc, dot_product_4x4_i8i8i32_sdot
)

TensorIntrin.register(
    ARM_DOT_12x8_fp32_MICROKERNEL_INTRIN, dot_product_12x8_f32f32f32_desc, dot_product_12x8_f32f32f32_microkernel
)

TensorIntrin.register(
    ARM_DOT_8x12_i8_MICROKERNEL_INTRIN, dot_product_8x12_i8i8i32_desc, dot_product_8x12_i8i8i32_microkernel
)

TensorIntrin.register(
    ARM_DOT_8x12x4_i8_MICROKERNEL_INTRIN, dot_product_8x12x4_i8i8i32_desc, dot_product_8x12x4_i8i8i32_microkernel
)

TensorIntrin.register(
    ARM_DOT_8x12x16_i8_MICROKERNEL_INTRIN, dot_product_8x12x16_i8i8i32_desc, dot_product_8x12x16_i8i8i32_microkernel
)

TensorIntrin.register(
    ARM_DOT_8x12x16_i8_FAKE_MICROKERNEL_INTRIN, dot_product_8x12x16_i8i8i32_fake_desc, dot_product_8x12x16_i8i8i32_fake_microkernel
)