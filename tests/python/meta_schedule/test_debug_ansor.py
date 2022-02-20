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
from typing import Tuple

import tvm
from tvm import te, topi


TARGET = tvm.target.Target("nvidia/jetson-agx-xavier")

@tvm.register_func
def tvm_callback_cuda_postproc(code):
    import os
    if not os.path.exists("/tmp/perf"):
        os.mkdir("/tmp/perf")
    with open("/tmp/perf/te.cu", "w") as f:
        f.write(code)
    return code


def func(  # pylint: disable=invalid-name,missing-docstring
    N: int,
    L: int,
    CI: int,
    CO: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor, te.Tensor]:
    inputs = te.placeholder((N, L, CI), name="inputs")
    weight = te.placeholder((kernel_size, CI // groups, CO), name="weight")

    batch_size, in_len, _ = inputs.shape
    k_len, channel_per_group, out_channel = weight.shape
    out_channel_per_group = out_channel // groups
    out_len = (in_len + 2 * padding - dilation * (k_len - 1) - 1) // stride + 1
    rc = te.reduce_axis((0, channel_per_group), name="rc")
    rl = te.reduce_axis((0, k_len), name="rl")

    padded = topi.nn.pad(inputs, [0, padding, 0])
    output = te.compute(
        (batch_size, out_len, out_channel),
        lambda n, l, co: te.sum(
            (
                padded[
                    n,
                    l * stride + rl * dilation,
                    co // out_channel_per_group * channel_per_group + rc,
                ]
                * weight[rl, rc, co]
            ),
            axis=[rl, rc],
        ),
        name="conv1d_nlc",
    )
    return (inputs, weight, padded, output)


def main():
    inputs, weight, PadInput, conv1d_nlc = func(1, 256, 64, 128, 3, 2, 1)
    s = te.create_schedule(conv1d_nlc.op)
    # fmt: off
    PadInput_i0, PadInput_i1, PadInput_i2 = tuple(PadInput.op.axis) + tuple(PadInput.op.reduce_axis)
    conv1d_nlc_n, conv1d_nlc_l, conv1d_nlc_co, conv1d_nlc_rl, conv1d_nlc_rc = tuple(conv1d_nlc.op.axis) + tuple(conv1d_nlc.op.reduce_axis)
    conv1d_nlc_local, = s.cache_write([conv1d_nlc], "local")
    conv1d_nlc_local_n_c, conv1d_nlc_local_l_c, conv1d_nlc_local_co_c, conv1d_nlc_local_rl, conv1d_nlc_local_rc = tuple(conv1d_nlc_local.op.axis) + tuple(conv1d_nlc_local.op.reduce_axis)
    conv1d_nlc_local_n_c_o_i, conv1d_nlc_local_n_c_i = s[conv1d_nlc_local].split(conv1d_nlc_local_n_c, factor=1)
    conv1d_nlc_local_n_c_o_o_i, conv1d_nlc_local_n_c_o_i = s[conv1d_nlc_local].split(conv1d_nlc_local_n_c_o_i, factor=1)
    conv1d_nlc_local_n_c_o_o_o_i, conv1d_nlc_local_n_c_o_o_i = s[conv1d_nlc_local].split(conv1d_nlc_local_n_c_o_o_i, factor=1)
    conv1d_nlc_local_n_c_o_o_o_o, conv1d_nlc_local_n_c_o_o_o_i = s[conv1d_nlc_local].split(conv1d_nlc_local_n_c_o_o_o_i, factor=1)
    conv1d_nlc_local_l_c_o_i, conv1d_nlc_local_l_c_i = s[conv1d_nlc_local].split(conv1d_nlc_local_l_c, factor=1)
    conv1d_nlc_local_l_c_o_o_i, conv1d_nlc_local_l_c_o_i = s[conv1d_nlc_local].split(conv1d_nlc_local_l_c_o_i, factor=4)
    conv1d_nlc_local_l_c_o_o_o_i, conv1d_nlc_local_l_c_o_o_i = s[conv1d_nlc_local].split(conv1d_nlc_local_l_c_o_o_i, factor=8)
    conv1d_nlc_local_l_c_o_o_o_o, conv1d_nlc_local_l_c_o_o_o_i = s[conv1d_nlc_local].split(conv1d_nlc_local_l_c_o_o_o_i, factor=1)
    conv1d_nlc_local_co_c_o_i, conv1d_nlc_local_co_c_i = s[conv1d_nlc_local].split(conv1d_nlc_local_co_c, factor=2)
    conv1d_nlc_local_co_c_o_o_i, conv1d_nlc_local_co_c_o_i = s[conv1d_nlc_local].split(conv1d_nlc_local_co_c_o_i, factor=1)
    conv1d_nlc_local_co_c_o_o_o_i, conv1d_nlc_local_co_c_o_o_i = s[conv1d_nlc_local].split(conv1d_nlc_local_co_c_o_o_i, factor=16)
    conv1d_nlc_local_co_c_o_o_o_o, conv1d_nlc_local_co_c_o_o_o_i = s[conv1d_nlc_local].split(conv1d_nlc_local_co_c_o_o_o_i, factor=1)
    conv1d_nlc_local_rl_o_i, conv1d_nlc_local_rl_i = s[conv1d_nlc_local].split(conv1d_nlc_local_rl, factor=3)
    conv1d_nlc_local_rl_o_o, conv1d_nlc_local_rl_o_i = s[conv1d_nlc_local].split(conv1d_nlc_local_rl_o_i, factor=1)
    conv1d_nlc_local_rc_o_i, conv1d_nlc_local_rc_i = s[conv1d_nlc_local].split(conv1d_nlc_local_rc, factor=2)
    conv1d_nlc_local_rc_o_o, conv1d_nlc_local_rc_o_i = s[conv1d_nlc_local].split(conv1d_nlc_local_rc_o_i, factor=8)
    s[conv1d_nlc_local].reorder(conv1d_nlc_local_n_c_o_o_o_o, conv1d_nlc_local_l_c_o_o_o_o, conv1d_nlc_local_co_c_o_o_o_o, conv1d_nlc_local_n_c_o_o_o_i, conv1d_nlc_local_l_c_o_o_o_i, conv1d_nlc_local_co_c_o_o_o_i, conv1d_nlc_local_n_c_o_o_i, conv1d_nlc_local_l_c_o_o_i, conv1d_nlc_local_co_c_o_o_i, conv1d_nlc_local_rl_o_o, conv1d_nlc_local_rc_o_o, conv1d_nlc_local_rl_o_i, conv1d_nlc_local_rc_o_i, conv1d_nlc_local_n_c_o_i, conv1d_nlc_local_l_c_o_i, conv1d_nlc_local_co_c_o_i, conv1d_nlc_local_rl_i, conv1d_nlc_local_rc_i, conv1d_nlc_local_n_c_i,
    conv1d_nlc_local_l_c_i, conv1d_nlc_local_co_c_i)
    conv1d_nlc_n_o_i, conv1d_nlc_n_i = s[conv1d_nlc].split(conv1d_nlc_n, factor=1)
    conv1d_nlc_n_o_o_i, conv1d_nlc_n_o_i = s[conv1d_nlc].split(conv1d_nlc_n_o_i, factor=1)
    conv1d_nlc_n_o_o_o, conv1d_nlc_n_o_o_i = s[conv1d_nlc].split(conv1d_nlc_n_o_o_i, factor=1)
    conv1d_nlc_l_o_i, conv1d_nlc_l_i = s[conv1d_nlc].split(conv1d_nlc_l, factor=4)
    conv1d_nlc_l_o_o_i, conv1d_nlc_l_o_i = s[conv1d_nlc].split(conv1d_nlc_l_o_i, factor=8)
    conv1d_nlc_l_o_o_o, conv1d_nlc_l_o_o_i = s[conv1d_nlc].split(conv1d_nlc_l_o_o_i, factor=1)
    conv1d_nlc_co_o_i, conv1d_nlc_co_i = s[conv1d_nlc].split(conv1d_nlc_co, factor=2)
    conv1d_nlc_co_o_o_i, conv1d_nlc_co_o_i = s[conv1d_nlc].split(conv1d_nlc_co_o_i, factor=16)
    conv1d_nlc_co_o_o_o, conv1d_nlc_co_o_o_i = s[conv1d_nlc].split(conv1d_nlc_co_o_o_i, factor=1)
    s[conv1d_nlc].reorder(conv1d_nlc_n_o_o_o, conv1d_nlc_l_o_o_o, conv1d_nlc_co_o_o_o, conv1d_nlc_n_o_o_i, conv1d_nlc_l_o_o_i, conv1d_nlc_co_o_o_i, conv1d_nlc_n_o_i, conv1d_nlc_l_o_i, conv1d_nlc_co_o_i, conv1d_nlc_n_i, conv1d_nlc_l_i, conv1d_nlc_co_i)
    s[conv1d_nlc_local].compute_at(s[conv1d_nlc], conv1d_nlc_co_o_i)
    weight_shared = s.cache_read(weight, "shared", [conv1d_nlc_local])
    weight_shared_ax0, weight_shared_ax1, weight_shared_ax2 = tuple(weight_shared.op.axis)
    s[weight_shared].compute_at(s[conv1d_nlc_local], conv1d_nlc_local_rc_o_o)
    PadInput_shared = s.cache_read(PadInput, "shared", [conv1d_nlc_local])
    PadInput_shared_ax0, PadInput_shared_ax1, PadInput_shared_ax2 = tuple(PadInput_shared.op.axis)
    s[PadInput_shared].compute_at(s[conv1d_nlc_local], conv1d_nlc_local_rc_o_o)
    s[PadInput].compute_inline()
    conv1d_nlc_n_o_o_o_l_o_o_o_fused_co_o_o_o_fused = s[conv1d_nlc].fuse(conv1d_nlc_n_o_o_o, conv1d_nlc_l_o_o_o, conv1d_nlc_co_o_o_o)
    s[conv1d_nlc].bind(conv1d_nlc_n_o_o_o_l_o_o_o_fused_co_o_o_o_fused, te.thread_axis("blockIdx.x"))
    conv1d_nlc_n_o_o_i_l_o_o_i_fused_co_o_o_i_fused = s[conv1d_nlc].fuse(conv1d_nlc_n_o_o_i, conv1d_nlc_l_o_o_i, conv1d_nlc_co_o_o_i)
    s[conv1d_nlc].bind(conv1d_nlc_n_o_o_i_l_o_o_i_fused_co_o_o_i_fused, te.thread_axis("vthread"))
    conv1d_nlc_n_o_i_l_o_i_fused_co_o_i_fused = s[conv1d_nlc].fuse(conv1d_nlc_n_o_i, conv1d_nlc_l_o_i, conv1d_nlc_co_o_i)
    s[conv1d_nlc].bind(conv1d_nlc_n_o_i_l_o_i_fused_co_o_i_fused, te.thread_axis("threadIdx.x"))
    weight_shared_ax0_ax1_fused_ax2_fused = s[weight_shared].fuse(weight_shared_ax0, weight_shared_ax1, weight_shared_ax2)
    weight_shared_ax0_ax1_fused_ax2_fused_o, weight_shared_ax0_ax1_fused_ax2_fused_i = s[weight_shared].split(weight_shared_ax0_ax1_fused_ax2_fused, factor=1)
    s[weight_shared].vectorize(weight_shared_ax0_ax1_fused_ax2_fused_i)
    weight_shared_ax0_ax1_fused_ax2_fused_o_o, weight_shared_ax0_ax1_fused_ax2_fused_o_i = s[weight_shared].split(weight_shared_ax0_ax1_fused_ax2_fused_o, factor=128)
    s[weight_shared].bind(weight_shared_ax0_ax1_fused_ax2_fused_o_i, te.thread_axis("threadIdx.x"))
    PadInput_shared_ax0_ax1_fused_ax2_fused = s[PadInput_shared].fuse(PadInput_shared_ax0, PadInput_shared_ax1, PadInput_shared_ax2)
    PadInput_shared_ax0_ax1_fused_ax2_fused_o, PadInput_shared_ax0_ax1_fused_ax2_fused_i = s[PadInput_shared].split(PadInput_shared_ax0_ax1_fused_ax2_fused, factor=1)
    s[PadInput_shared].vectorize(PadInput_shared_ax0_ax1_fused_ax2_fused_i)
    PadInput_shared_ax0_ax1_fused_ax2_fused_o_o, PadInput_shared_ax0_ax1_fused_ax2_fused_o_i = s[PadInput_shared].split(PadInput_shared_ax0_ax1_fused_ax2_fused_o, factor=128)
    s[PadInput_shared].bind(PadInput_shared_ax0_ax1_fused_ax2_fused_o_i, te.thread_axis("threadIdx.x"))
    # s[conv1d_nlc_local].pragma(conv1d_nlc_local_n_c_o_o_o_o, "auto_unroll_max_step", 1024)
    # s[conv1d_nlc_local].pragma(conv1d_nlc_local_n_c_o_o_o_o, "unroll_explicit", True)
    # fmt: off
    print(tvm.lower(s, [inputs, weight, conv1d_nlc]).script())
    tvm.build(s, [inputs, weight, conv1d_nlc], target=TARGET)


if __name__ == "__main__":
    main()
