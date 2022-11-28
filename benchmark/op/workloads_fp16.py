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

from typing import Tuple

from tvm import te, tir, topi

from benchmark.utils import load_config


def _conv2d_nhwc_f16(  # pylint: disable=invalid-name
    Input: te.Tensor,
    Filter: te.Tensor,
    stride: int,
    padding: int,
    dilation: int,
    out_dtype="float32",
):
    batch, in_h, in_w, in_channel = Input.shape  # type: ignore
    out_channel, k_h, k_w, channel_per_group = Filter.shape  # type: ignore
    groups = in_channel // channel_per_group
    out_channel_per_group = out_channel // groups

    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1

    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")
    rc = te.reduce_axis((0, channel_per_group), name="rc")

    padded = topi.nn.pad(Input, [0, padding, padding, 0])

    output = te.compute(
        (batch, out_h, out_w, out_channel),
        lambda n, h, w, co: te.sum(
            (
                tir.Cast(
                    value=padded[
                        n,
                        h * stride + rh * dilation,
                        w * stride + rw * dilation,
                        co // out_channel_per_group * channel_per_group + rc,
                    ],
                    dtype=out_dtype,
                )
                * tir.Cast(value=Filter[co, rh, rw, rc], dtype=out_dtype)
            ),
            axis=[rh, rw, rc],
        ),
        name="conv2d_nhwc",
    )
    return output


def conv1d_nlc_f16(  # pylint: disable=invalid-name,missing-docstring
    N: int,
    L: int,
    CI: int,
    CO: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    out_dtype: str = "float32",
):
    inputs = te.placeholder((N, L, CI), name="inputs", dtype="float16")
    weight = te.placeholder(
        (CO, kernel_size, CI // groups), name="weight", dtype="float16"
    )

    batch_size, in_len, _ = inputs.shape
    out_channel, k_len, channel_per_group = weight.shape
    out_channel_per_group = out_channel // groups
    out_len = (in_len + 2 * padding - dilation * (k_len - 1) - 1) // stride + 1
    rc = te.reduce_axis((0, channel_per_group), name="rc")
    rl = te.reduce_axis((0, k_len), name="rl")

    padded = topi.nn.pad(inputs, [0, padding, 0])
    output = te.compute(
        (batch_size, out_len, out_channel),
        lambda n, l, co: te.sum(
            (
                tir.Cast(
                    value=padded[
                        n,
                        l * stride + rl * dilation,
                        co // out_channel_per_group * channel_per_group + rc,
                    ],
                    dtype=out_dtype,
                )
                * tir.Cast(value=weight[co, rl, rc], dtype=out_dtype)
            ),
            axis=[rl, rc],
        ),
        name="conv1d_nlc",
    )
    return (inputs, weight, output)


def conv2d_nhwc_f16(  # pylint: disable=invalid-name,missing-docstring
    N: int,
    H: int,
    W: int,
    CI: int,
    CO: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    out_dtype: str = "float32",
):
    inputs = te.placeholder((N, H, W, CI), name="inputs", dtype="float16")
    weight = te.placeholder(
        (CO, kernel_size, kernel_size, CI // groups), name="weight", dtype="float16"
    )
    output = _conv2d_nhwc_f16(inputs, weight, stride, padding, dilation, out_dtype)
    return (inputs, weight, output)


def conv3d_ndhwc_f16(  # pylint: disable=invalid-name,missing-docstring
    N: int,
    D: int,
    H: int,
    W: int,
    CI: int,
    CO: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    out_dtype: str = "float32",
):
    inputs = te.placeholder((N, D, H, W, CI), name="inputs", dtype="float16")
    weight = te.placeholder(
        (CO, kernel_size, kernel_size, kernel_size, CI // groups),
        name="weight",
        dtype="float16",
    )
    batch_size, in_d, in_h, in_w, _ = inputs.shape
    out_channel, k_d, k_h, k_w, channel_per_group = weight.shape
    out_channel_per_group = out_channel // groups

    out_d = (in_d + 2 * padding - dilation * (k_d - 1) - 1) // stride + 1
    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    rd = te.reduce_axis((0, k_d), name="rd")
    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")
    rc = te.reduce_axis((0, channel_per_group), name="rc")

    padded = topi.nn.pad(inputs, [0, padding, padding, padding, 0])

    output = te.compute(
        (batch_size, out_d, out_h, out_w, out_channel),
        lambda n, d, h, w, co: te.sum(
            (
                tir.Cast(
                    value=padded[
                        n,
                        d * stride + rd * dilation,
                        h * stride + rh * dilation,
                        w * stride + rw * dilation,
                        co // out_channel_per_group * channel_per_group + rc,
                    ],
                    dtype=out_dtype,
                )
                * tir.Cast(value=weight[co, rd, rh, rw, rc], dtype=out_dtype)
            ),
            axis=[rd, rh, rw, rc],
        ),
        name="conv3d_ndhwc",
    )

    return (inputs, weight, output)


def batch_matmul_nkmk_f16(  # pylint: disable=invalid-name,missing-docstring
    B: int,
    N: int,
    M: int,
    K: int,
    out_dtype: str = "float32",
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    x = te.placeholder((B, N, K), name="X", dtype="float16")
    y = te.placeholder((B, M, K), name="Y", dtype="float16")
    k = te.reduce_axis((0, K), name="k")
    z = te.compute(
        (B, N, M),
        lambda b, i, j: te.sum(
            tir.Cast(out_dtype, x[b][i][k]) * tir.Cast(out_dtype, y[b][j][k]),
            axis=[k],
        ),
        name="Z",
    )
    return (x, y, z)


def depthwise_conv2d_nhwc_f16(  # pylint: disable=invalid-name,missing-docstring
    N: int,
    H: int,
    W: int,
    C: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    factor: int = 1,
    out_dtype: str = "float32",
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    inputs = te.placeholder((N, H, W, C), dtype="float16")
    weight = te.placeholder((C, kernel_size, kernel_size, factor), dtype="float16")
    batch_size, in_h, in_w, in_channel = inputs.shape
    _, k_h, k_w, factor = weight.shape
    out_channel = in_channel * factor
    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")
    padded = topi.nn.pad(inputs, [0, padding, padding, 0])
    idxdiv = te.indexdiv
    idxmod = te.indexmod
    Conv = te.compute(
        (batch_size, out_h, out_w, in_channel, factor),
        lambda n, h, w, co, ci: te.sum(
            tir.Cast(
                out_dtype,
                padded[n, h * stride + rh * dilation, w * stride + rw * dilation, co],
            )
            * tir.Cast(out_dtype, weight[co, rh, rw, ci]),
            axis=[rh, rw],
        ),
        name="depth_conv2d_nhwc",
    )
    reshape_output = te.compute(
        (batch_size, out_h, out_w, out_channel),
        lambda n, h, w, c: Conv[n, h, w, idxdiv(c, factor), idxmod(c, factor)],
        name="reshape_output",
    )
    return (inputs, weight, reshape_output)


def conv2d_transpose_nhwc_f16(  # pylint: disable=invalid-name,missing-docstring
    N: int,
    H: int,
    W: int,
    CI: int,
    CO: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    out_dtype: str = "float32",
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    inputs = te.placeholder((N, H, W, CI), name="inputs", dtype="float16")
    weight = te.placeholder(
        (kernel_size, kernel_size, CI, CO), name="weight", dtype="float16"
    )

    batch, in_h, in_w, in_c = inputs.shape
    filter_h, filter_w, in_c, out_c = weight.shape
    stride_h, stride_w = (stride, stride)

    # compute padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = topi.nn.get_pad_tuple(
        padding, (filter_h, filter_w)
    )
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right

    # padding stage
    padded = topi.nn.pad(
        inputs,
        [
            0,
            (bpad_top + stride_h - 1) // stride_h,
            (bpad_left + stride_w - 1) // stride_w,
            0,
        ],
        [
            0,
            (bpad_bottom + stride_h - 1) // stride_h,
            (bpad_right + stride_w - 1) // stride_w,
            0,
        ],
    )

    # remove extra padding introduced by dilatation
    idx_div = te.indexdiv
    idx_mod = te.indexmod
    border_h = idx_mod(stride_h - idx_mod(bpad_top, stride_h), stride_h)
    border_w = idx_mod(stride_w - idx_mod(bpad_left, stride_w), stride_w)

    # dilation stage
    strides = [1, stride_h, stride_w, 1]
    n = len(padded.shape)

    # We should embed this dilation directly into te.compute rather than creating a new te.compute.
    # Only in this way can we use unroll to eliminate the multiplication of zeros.
    def _dilate(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if not strides[i] == 1:
                index_tuple.append(idx_div(indices[i], strides[i]))
                not_zero.append(idx_mod(indices[i], strides[i]).equal(0))
            else:
                index_tuple.append(indices[i])
        if not_zero:
            not_zero = te.all(*not_zero)
            return te.if_then_else(
                not_zero, padded(*index_tuple), tir.const(0.0, padded.dtype)
            )
        return padded(*index_tuple)

    # convolution stage
    out_h = (in_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (in_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    rc = te.reduce_axis((0, in_c), name="rc")
    rh = te.reduce_axis((0, filter_h), name="rh")
    rw = te.reduce_axis((0, filter_w), name="rw")

    output = te.compute(
        (batch, out_h, out_w, out_c),
        lambda n, h, w, co: te.sum(
            tir.Cast(out_dtype, _dilate(n, h + rh + border_h, w + rw + border_w, rc))
            * tir.Cast(out_dtype, weight[filter_h - 1 - rh, filter_w - 1 - rw, rc, co]),
            axis=[rh, rw, rc],
        ),
        name="conv2d_transpose_nhwc",
    )
    return (inputs, weight, output)


def conv2d_nhwc_bn_relu_f16(  # pylint: disable=invalid-name,missing-docstring
    N: int,
    H: int,
    W: int,
    CI: int,
    CO: int,
    kernel_size: int,
    strides: int,
    padding: int,
    dilation: int = 1,
    out_dtype="float32",
) -> Tuple[te.Tensor, te.Tensor, te.Tensor, te.Tensor, te.Tensor, te.Tensor]:
    data = te.placeholder((N, H, W, CI), name="data", dtype="float16")
    kernel = te.placeholder(
        (CO, kernel_size, kernel_size, CI), name="kernel", dtype="float16"
    )
    bias = te.placeholder((CO,), name="bias")
    bn_scale = te.placeholder((CO,), name="bn_scale")
    bn_offset = te.placeholder((CO,), name="bn_offset")
    OH = (H + 2 * padding - (kernel_size - 1) * dilation - 1) // strides + 1
    OW = (W + 2 * padding - (kernel_size - 1) * dilation - 1) // strides + 1
    conv = _conv2d_nhwc_f16(data, kernel, strides, padding, dilation, out_dtype)
    conv = te.compute(
        (N, OH, OW, CO), lambda i, j, k, l: conv[i, j, k, l] + bias[l], name="bias_add"
    )
    conv = te.compute(
        (N, OH, OW, CO),
        lambda i, j, k, l: conv[i, j, k, l] * bn_scale[l],
        name="bn_mul",
    )
    conv = te.compute(
        (N, OH, OW, CO),
        lambda i, j, k, l: conv[i, j, k, l] + bn_offset[l],
        name="bn_add",
    )
    out = topi.nn.relu(conv)
    return (data, kernel, bias, bn_offset, bn_scale, out)


def transpose_batch_matmul_f16(  # pylint: disable=invalid-name,missing-docstring
    batch: int,
    seq_len: int,
    n_head: int,
    n_dim: int,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    query = te.placeholder(
        (batch, seq_len, n_head, n_dim), name="query", dtype="float16"
    )
    value = te.placeholder(
        (batch, seq_len, n_head, n_dim), name="value", dtype="float16"
    )
    query_T = te.compute(
        (batch, n_head, seq_len, n_dim),
        lambda b, h, l, d: query[b, l, h, d],
        name="query_T",
    )
    value_T = te.compute(
        (batch, n_head, n_dim, seq_len),
        lambda b, h, d, l: value[b, l, h, d],
        name="value_T",
    )
    k = te.reduce_axis((0, n_dim), name="k")
    out = te.compute(
        (batch, n_head, seq_len, seq_len),
        lambda b, h, i, j: te.sum(
            tir.Cast("float32", query_T[b, h, i, k])
            * tir.Cast("float32", value_T[b, h, k, j]),
            axis=[k],
        ),
        name="C",
    )
    return (query, value, out)


def create_te_workload_f16(
    name: str,
    batch_size: int = 1,
    out_dtype="float32",
) -> tir.PrimFunc:
    workload_func = CONFIGS_F16[name]
    param = [batch_size] + shape_configs[name][1:]
    f = te.create_prim_func(workload_func(*param, out_dtype=out_dtype))  # type: ignore
    return f


shape_configs = load_config()

CONFIGS_F16 = {
    "C1D": conv1d_nlc_f16,
    "C2D": conv2d_nhwc_f16,
    "C3D": conv3d_ndhwc_f16,
    "GMM-1024": batch_matmul_nkmk_f16,
    "GMM-4096": batch_matmul_nkmk_f16,
    "DIL": conv2d_nhwc_f16,
    "DEP": depthwise_conv2d_nhwc_f16,
    "GRP": conv2d_nhwc_f16,
    "T2D": conv2d_transpose_nhwc_f16,
    "CBR": conv2d_nhwc_bn_relu_f16,
    # "TBG": (
    #     transpose_batch_matmul_f16,
    #     [
    #         (1, 128, 12, 64),
    #         (1, 128, 16, 64),
    #         (1, 64, 12, 128),
    #         (1, 128, 12, 128),
    #     ],
    # ),
}
