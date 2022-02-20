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
from typing import List

from tvm.meta_schedule.testing import te_workload
from tvm.meta_schedule.testing.space_generation import check_trace, create_context
from tvm.target import Target
from tvm.te import create_prim_func


def _target() -> Target:
    return Target("llvm --num-cores=16")


def test_meta_schedule_cpu_sketch_matmul():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "l2, l3, l4 = sch.get_loops(block=b0)",
            "v5, v6, v7, v8 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l9, l10, l11, l12 = sch.split(loop=l2, factors=[v5, v6, v7, v8])",
            "v13, v14, v15, v16 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l17, l18, l19, l20 = sch.split(loop=l3, factors=[v13, v14, v15, v16])",
            "v21, v22 = sch.sample_perfect_tile(loop=l4, n=2, max_innermost_factor=64)",
            "l23, l24 = sch.split(loop=l4, factors=[v21, v22])",
            "sch.reorder(l9, l17, l10, l18, l23, l11, l19, l24, l12, l20)",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v25 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v25)',
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            'b2 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")',
            "l3, l4, l5 = sch.get_loops(block=b0)",
            "v6, v7, v8, v9 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l10, l11, l12, l13 = sch.split(loop=l3, factors=[v6, v7, v8, v9])",
            "v14, v15, v16, v17 = sch.sample_perfect_tile(loop=l4, n=4, max_innermost_factor=64)",
            "l18, l19, l20, l21 = sch.split(loop=l4, factors=[v14, v15, v16, v17])",
            "v22, v23 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)",
            "l24, l25 = sch.split(loop=l5, factors=[v22, v23])",
            "sch.reorder(l10, l18, l11, l19, l24, l12, l20, l25, l13, l21)",
            "sch.reverse_compute_at(block=b2, loop=l18, preserve_unit_loops=True)",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v26 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v26)',
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            'b2 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="global")',
            "l3, l4, l5 = sch.get_loops(block=b0)",
            "v6, v7, v8, v9 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l10, l11, l12, l13 = sch.split(loop=l3, factors=[v6, v7, v8, v9])",
            "v14, v15, v16, v17 = sch.sample_perfect_tile(loop=l4, n=4, max_innermost_factor=64)",
            "l18, l19, l20, l21 = sch.split(loop=l4, factors=[v14, v15, v16, v17])",
            "v22, v23 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)",
            "l24, l25 = sch.split(loop=l5, factors=[v22, v23])",
            "sch.reorder(l10, l18, l11, l19, l24, l12, l20, l25, l13, l21)",
            "sch.reverse_compute_at(block=b2, loop=l19, preserve_unit_loops=True)",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v26 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v26)',
        ],
    ]
    # pylint: enable=line-too-long
    ctx = create_context(
        create_prim_func(
            te_workload.matmul(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=_target(),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 3
    check_trace(spaces, expected)


def test_meta_schedule_cpu_sketch_matmul_relu():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "l2, l3, l4 = sch.get_loops(block=b0)",
            "v5, v6, v7, v8 = sch.sample_perfect_tile(loop=l2, n=4, max_innermost_factor=64)",
            "l9, l10, l11, l12 = sch.split(loop=l2, factors=[v5, v6, v7, v8])",
            "v13, v14, v15, v16 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l17, l18, l19, l20 = sch.split(loop=l3, factors=[v13, v14, v15, v16])",
            "v21, v22 = sch.sample_perfect_tile(loop=l4, n=2, max_innermost_factor=64)",
            "l23, l24 = sch.split(loop=l4, factors=[v21, v22])",
            "sch.reorder(l9, l17, l10, l18, l23, l11, l19, l24, l12, l20)",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v25 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v25)',
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "b2, = sch.get_consumers(block=b0)",
            "l3, l4, l5 = sch.get_loops(block=b0)",
            "v6, v7, v8, v9 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l10, l11, l12, l13 = sch.split(loop=l3, factors=[v6, v7, v8, v9])",
            "v14, v15, v16, v17 = sch.sample_perfect_tile(loop=l4, n=4, max_innermost_factor=64)",
            "l18, l19, l20, l21 = sch.split(loop=l4, factors=[v14, v15, v16, v17])",
            "v22, v23 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)",
            "l24, l25 = sch.split(loop=l5, factors=[v22, v23])",
            "sch.reorder(l10, l18, l11, l19, l24, l12, l20, l25, l13, l21)",
            "sch.reverse_compute_at(block=b2, loop=l18, preserve_unit_loops=True)",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v26 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v26)',
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "b2, = sch.get_consumers(block=b0)",
            "l3, l4, l5 = sch.get_loops(block=b0)",
            "v6, v7, v8, v9 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l10, l11, l12, l13 = sch.split(loop=l3, factors=[v6, v7, v8, v9])",
            "v14, v15, v16, v17 = sch.sample_perfect_tile(loop=l4, n=4, max_innermost_factor=64)",
            "l18, l19, l20, l21 = sch.split(loop=l4, factors=[v14, v15, v16, v17])",
            "v22, v23 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)",
            "l24, l25 = sch.split(loop=l5, factors=[v22, v23])",
            "sch.reorder(l10, l18, l11, l19, l24, l12, l20, l25, l13, l21)",
            "sch.reverse_compute_at(block=b2, loop=l19, preserve_unit_loops=True)",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v26 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v26)',
        ],
    ]
    # pylint: enable=line-too-long
    ctx = create_context(
        create_prim_func(
            te_workload.matmul_relu(
                n=512,
                m=512,
                k=512,
            )
        ),
        target=_target(),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 3
    check_trace(spaces, expected)


def test_meta_schedule_cpu_sketch_conv2d_nchw():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="pad_temp", func_name="main")',
            'b1 = sch.get_block(name="conv2d_nchw", func_name="main")',
            'b2 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "l3, l4, l5, l6, l7, l8, l9 = sch.get_loops(block=b1)",
            "v10, v11, v12, v13 = sch.sample_perfect_tile(loop=l3, n=4, max_innermost_factor=64)",
            "l14, l15, l16, l17 = sch.split(loop=l3, factors=[v10, v11, v12, v13])",
            "v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l4, n=4, max_innermost_factor=64)",
            "l22, l23, l24, l25 = sch.split(loop=l4, factors=[v18, v19, v20, v21])",
            "v26, v27, v28, v29 = sch.sample_perfect_tile(loop=l5, n=4, max_innermost_factor=64)",
            "l30, l31, l32, l33 = sch.split(loop=l5, factors=[v26, v27, v28, v29])",
            "v34, v35, v36, v37 = sch.sample_perfect_tile(loop=l6, n=4, max_innermost_factor=64)",
            "l38, l39, l40, l41 = sch.split(loop=l6, factors=[v34, v35, v36, v37])",
            "v42, v43 = sch.sample_perfect_tile(loop=l7, n=2, max_innermost_factor=64)",
            "l44, l45 = sch.split(loop=l7, factors=[v42, v43])",
            "v46, v47 = sch.sample_perfect_tile(loop=l8, n=2, max_innermost_factor=64)",
            "l48, l49 = sch.split(loop=l8, factors=[v46, v47])",
            "v50, v51 = sch.sample_perfect_tile(loop=l9, n=2, max_innermost_factor=64)",
            "l52, l53 = sch.split(loop=l9, factors=[v50, v51])",
            "sch.reorder(l14, l22, l30, l38, l15, l23, l31, l39, l44, l48, l52, l16, l24, l32, l40, l45, l49, l53, l17, l25, l33, l41)",
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v54 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v54)',
            "l55 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l55, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="pad_temp", func_name="main")',
            'b1 = sch.get_block(name="conv2d_nchw", func_name="main")',
            'b2 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            'b3 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="global")',
            "l4, l5, l6, l7, l8, l9, l10 = sch.get_loops(block=b1)",
            "v11, v12, v13, v14 = sch.sample_perfect_tile(loop=l4, n=4, max_innermost_factor=64)",
            "l15, l16, l17, l18 = sch.split(loop=l4, factors=[v11, v12, v13, v14])",
            "v19, v20, v21, v22 = sch.sample_perfect_tile(loop=l5, n=4, max_innermost_factor=64)",
            "l23, l24, l25, l26 = sch.split(loop=l5, factors=[v19, v20, v21, v22])",
            "v27, v28, v29, v30 = sch.sample_perfect_tile(loop=l6, n=4, max_innermost_factor=64)",
            "l31, l32, l33, l34 = sch.split(loop=l6, factors=[v27, v28, v29, v30])",
            "v35, v36, v37, v38 = sch.sample_perfect_tile(loop=l7, n=4, max_innermost_factor=64)",
            "l39, l40, l41, l42 = sch.split(loop=l7, factors=[v35, v36, v37, v38])",
            "v43, v44 = sch.sample_perfect_tile(loop=l8, n=2, max_innermost_factor=64)",
            "l45, l46 = sch.split(loop=l8, factors=[v43, v44])",
            "v47, v48 = sch.sample_perfect_tile(loop=l9, n=2, max_innermost_factor=64)",
            "l49, l50 = sch.split(loop=l9, factors=[v47, v48])",
            "v51, v52 = sch.sample_perfect_tile(loop=l10, n=2, max_innermost_factor=64)",
            "l53, l54 = sch.split(loop=l10, factors=[v51, v52])",
            "sch.reorder(l15, l23, l31, l39, l16, l24, l32, l40, l45, l49, l53, l17, l25, l33, l41, l46, l50, l54, l18, l26, l34, l42)",
            "sch.reverse_compute_at(block=b3, loop=l39, preserve_unit_loops=True)",
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v55 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v55)',
            "l56 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l56, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="pad_temp", func_name="main")',
            'b1 = sch.get_block(name="conv2d_nchw", func_name="main")',
            'b2 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            'b3 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="global")',
            "l4, l5, l6, l7, l8, l9, l10 = sch.get_loops(block=b1)",
            "v11, v12, v13, v14 = sch.sample_perfect_tile(loop=l4, n=4, max_innermost_factor=64)",
            "l15, l16, l17, l18 = sch.split(loop=l4, factors=[v11, v12, v13, v14])",
            "v19, v20, v21, v22 = sch.sample_perfect_tile(loop=l5, n=4, max_innermost_factor=64)",
            "l23, l24, l25, l26 = sch.split(loop=l5, factors=[v19, v20, v21, v22])",
            "v27, v28, v29, v30 = sch.sample_perfect_tile(loop=l6, n=4, max_innermost_factor=64)",
            "l31, l32, l33, l34 = sch.split(loop=l6, factors=[v27, v28, v29, v30])",
            "v35, v36, v37, v38 = sch.sample_perfect_tile(loop=l7, n=4, max_innermost_factor=64)",
            "l39, l40, l41, l42 = sch.split(loop=l7, factors=[v35, v36, v37, v38])",
            "v43, v44 = sch.sample_perfect_tile(loop=l8, n=2, max_innermost_factor=64)",
            "l45, l46 = sch.split(loop=l8, factors=[v43, v44])",
            "v47, v48 = sch.sample_perfect_tile(loop=l9, n=2, max_innermost_factor=64)",
            "l49, l50 = sch.split(loop=l9, factors=[v47, v48])",
            "v51, v52 = sch.sample_perfect_tile(loop=l10, n=2, max_innermost_factor=64)",
            "l53, l54 = sch.split(loop=l10, factors=[v51, v52])",
            "sch.reorder(l15, l23, l31, l39, l16, l24, l32, l40, l45, l49, l53, l17, l25, l33, l41, l46, l50, l54, l18, l26, l34, l42)",
            "sch.reverse_compute_at(block=b3, loop=l40, preserve_unit_loops=True)",
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v55 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v55)',
            "l56 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l56, preserve_unit_loops=True)",
        ],
    ]
    # pylint: enable=line-too-long
    ctx = create_context(
        create_prim_func(
            te_workload.conv2d_nchw(
                n=1,
                h=56,
                w=56,
                ci=512,
                co=512,
                kh=3,
                kw=3,
                stride=1,
                padding=1,
            )
        ),
        target=_target(),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 3
    check_trace(spaces, expected)


def test_meta_schedule_cpu_sketch_conv2d_nchw_bias_bn_relu():  # pylint: disable=invalid-name
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="pad_temp", func_name="main")',
            'b1 = sch.get_block(name="conv2d_nchw", func_name="main")',
            'b2 = sch.get_block(name="bias_add", func_name="main")',
            'b3 = sch.get_block(name="bn_mul", func_name="main")',
            'b4 = sch.get_block(name="bn_add", func_name="main")',
            'b5 = sch.get_block(name="root", func_name="main")',
            "sch.compute_inline(block=b4)",
            "sch.compute_inline(block=b3)",
            "sch.compute_inline(block=b2)",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "l6, l7, l8, l9, l10, l11, l12 = sch.get_loops(block=b1)",
            "v13, v14, v15, v16 = sch.sample_perfect_tile(loop=l6, n=4, max_innermost_factor=64)",
            "l17, l18, l19, l20 = sch.split(loop=l6, factors=[v13, v14, v15, v16])",
            "v21, v22, v23, v24 = sch.sample_perfect_tile(loop=l7, n=4, max_innermost_factor=64)",
            "l25, l26, l27, l28 = sch.split(loop=l7, factors=[v21, v22, v23, v24])",
            "v29, v30, v31, v32 = sch.sample_perfect_tile(loop=l8, n=4, max_innermost_factor=64)",
            "l33, l34, l35, l36 = sch.split(loop=l8, factors=[v29, v30, v31, v32])",
            "v37, v38, v39, v40 = sch.sample_perfect_tile(loop=l9, n=4, max_innermost_factor=64)",
            "l41, l42, l43, l44 = sch.split(loop=l9, factors=[v37, v38, v39, v40])",
            "v45, v46 = sch.sample_perfect_tile(loop=l10, n=2, max_innermost_factor=64)",
            "l47, l48 = sch.split(loop=l10, factors=[v45, v46])",
            "v49, v50 = sch.sample_perfect_tile(loop=l11, n=2, max_innermost_factor=64)",
            "l51, l52 = sch.split(loop=l11, factors=[v49, v50])",
            "v53, v54 = sch.sample_perfect_tile(loop=l12, n=2, max_innermost_factor=64)",
            "l55, l56 = sch.split(loop=l12, factors=[v53, v54])",
            "sch.reorder(l17, l25, l33, l41, l18, l26, l34, l42, l47, l51, l55, l19, l27, l35, l43, l48, l52, l56, l20, l28, l36, l44)",
            'sch.annotate(block_or_loop=b5, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b5, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v57 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b5, ann_key="meta_schedule.unroll_explicit", ann_val=v57)',
            "l58 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l58, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="pad_temp", func_name="main")',
            'b1 = sch.get_block(name="conv2d_nchw", func_name="main")',
            'b2 = sch.get_block(name="bias_add", func_name="main")',
            'b3 = sch.get_block(name="bn_mul", func_name="main")',
            'b4 = sch.get_block(name="bn_add", func_name="main")',
            'b5 = sch.get_block(name="root", func_name="main")',
            "sch.compute_inline(block=b4)",
            "sch.compute_inline(block=b3)",
            "sch.compute_inline(block=b2)",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "b6, = sch.get_consumers(block=b1)",
            "l7, l8, l9, l10, l11, l12, l13 = sch.get_loops(block=b1)",
            "v14, v15, v16, v17 = sch.sample_perfect_tile(loop=l7, n=4, max_innermost_factor=64)",
            "l18, l19, l20, l21 = sch.split(loop=l7, factors=[v14, v15, v16, v17])",
            "v22, v23, v24, v25 = sch.sample_perfect_tile(loop=l8, n=4, max_innermost_factor=64)",
            "l26, l27, l28, l29 = sch.split(loop=l8, factors=[v22, v23, v24, v25])",
            "v30, v31, v32, v33 = sch.sample_perfect_tile(loop=l9, n=4, max_innermost_factor=64)",
            "l34, l35, l36, l37 = sch.split(loop=l9, factors=[v30, v31, v32, v33])",
            "v38, v39, v40, v41 = sch.sample_perfect_tile(loop=l10, n=4, max_innermost_factor=64)",
            "l42, l43, l44, l45 = sch.split(loop=l10, factors=[v38, v39, v40, v41])",
            "v46, v47 = sch.sample_perfect_tile(loop=l11, n=2, max_innermost_factor=64)",
            "l48, l49 = sch.split(loop=l11, factors=[v46, v47])",
            "v50, v51 = sch.sample_perfect_tile(loop=l12, n=2, max_innermost_factor=64)",
            "l52, l53 = sch.split(loop=l12, factors=[v50, v51])",
            "v54, v55 = sch.sample_perfect_tile(loop=l13, n=2, max_innermost_factor=64)",
            "l56, l57 = sch.split(loop=l13, factors=[v54, v55])",
            "sch.reorder(l18, l26, l34, l42, l19, l27, l35, l43, l48, l52, l56, l20, l28, l36, l44, l49, l53, l57, l21, l29, l37, l45)",
            "sch.reverse_compute_at(block=b6, loop=l42, preserve_unit_loops=True)",
            'sch.annotate(block_or_loop=b5, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b5, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v58 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b5, ann_key="meta_schedule.unroll_explicit", ann_val=v58)',
            "l59 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l59, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="pad_temp", func_name="main")',
            'b1 = sch.get_block(name="conv2d_nchw", func_name="main")',
            'b2 = sch.get_block(name="bias_add", func_name="main")',
            'b3 = sch.get_block(name="bn_mul", func_name="main")',
            'b4 = sch.get_block(name="bn_add", func_name="main")',
            'b5 = sch.get_block(name="root", func_name="main")',
            "sch.compute_inline(block=b4)",
            "sch.compute_inline(block=b3)",
            "sch.compute_inline(block=b2)",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSRSRS")',
            "b6, = sch.get_consumers(block=b1)",
            "l7, l8, l9, l10, l11, l12, l13 = sch.get_loops(block=b1)",
            "v14, v15, v16, v17 = sch.sample_perfect_tile(loop=l7, n=4, max_innermost_factor=64)",
            "l18, l19, l20, l21 = sch.split(loop=l7, factors=[v14, v15, v16, v17])",
            "v22, v23, v24, v25 = sch.sample_perfect_tile(loop=l8, n=4, max_innermost_factor=64)",
            "l26, l27, l28, l29 = sch.split(loop=l8, factors=[v22, v23, v24, v25])",
            "v30, v31, v32, v33 = sch.sample_perfect_tile(loop=l9, n=4, max_innermost_factor=64)",
            "l34, l35, l36, l37 = sch.split(loop=l9, factors=[v30, v31, v32, v33])",
            "v38, v39, v40, v41 = sch.sample_perfect_tile(loop=l10, n=4, max_innermost_factor=64)",
            "l42, l43, l44, l45 = sch.split(loop=l10, factors=[v38, v39, v40, v41])",
            "v46, v47 = sch.sample_perfect_tile(loop=l11, n=2, max_innermost_factor=64)",
            "l48, l49 = sch.split(loop=l11, factors=[v46, v47])",
            "v50, v51 = sch.sample_perfect_tile(loop=l12, n=2, max_innermost_factor=64)",
            "l52, l53 = sch.split(loop=l12, factors=[v50, v51])",
            "v54, v55 = sch.sample_perfect_tile(loop=l13, n=2, max_innermost_factor=64)",
            "l56, l57 = sch.split(loop=l13, factors=[v54, v55])",
            "sch.reorder(l18, l26, l34, l42, l19, l27, l35, l43, l48, l52, l56, l20, l28, l36, l44, l49, l53, l57, l21, l29, l37, l45)",
            "sch.reverse_compute_at(block=b6, loop=l43, preserve_unit_loops=True)",
            'sch.annotate(block_or_loop=b5, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b5, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v58 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b5, ann_key="meta_schedule.unroll_explicit", ann_val=v58)',
            "l59 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l59, preserve_unit_loops=True)",
        ],
    ]
    # pylint: enable=line-too-long
    ctx = create_context(
        create_prim_func(
            te_workload.conv2d_nchw_bias_bn_relu(
                n=1,
                h=56,
                w=56,
                ci=512,
                co=512,
                kh=3,
                kw=3,
                stride=1,
                padding=1,
            )
        ),
        target=_target(),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 3
    check_trace(spaces, expected)


def test_meta_schedule_sketch_cpu_max_pool2d_nchw():  # pylint: disable=invalid-name
    # pylint: disable=line-too-long
    expected: List[List[str]] = [
        [
            'b0 = sch.get_block(name="pad_temp", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v2 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v2)',
            "l3 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l3, preserve_unit_loops=True)",
        ],
    ]
    # pylint: enable=line-too-long
    ctx = create_context(
        create_prim_func(
            te_workload.max_pool2d_nchw(
                n=1,
                h=56,
                w=56,
                ci=512,
                padding=1,
            )
        ),
        target=_target(),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 1
    check_trace(spaces, expected)


def test_meta_schedule_cpu_sketch_batchnorm():  # pylint: disable=invalid-name
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            "l2, l3, l4 = sch.get_loops(block=b0)",
            "l5 = sch.fuse(l3, l4)",
            "v6, v7 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)",
            "l8, l9 = sch.split(loop=l5, factors=[v6, v7])",
            "b10 = sch.rfactor(loop=l8, factor_axis=1)",
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v11 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v11)',
            "b12, = sch.get_producers(block=b0)",
            'sch.unannotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer")',
            "l13 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l13, preserve_unit_loops=True)",
            "l14 = sch.sample_compute_location(block=b12)",
            "sch.compute_at(block=b12, loop=l14, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            "l2, l3, l4 = sch.get_loops(block=b0)",
            "l5 = sch.fuse(l3, l4)",
            "v6, v7 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)",
            "l8, l9 = sch.split(loop=l5, factors=[v6, v7])",
            "b10 = sch.rfactor(loop=l9, factor_axis=1)",
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v11 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v11)',
            "b12, = sch.get_producers(block=b0)",
            'sch.unannotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer")',
            "l13 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l13, preserve_unit_loops=True)",
            "l14 = sch.sample_compute_location(block=b12)",
            "sch.compute_at(block=b12, loop=l14, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v2 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v2)',
            "l3 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l3, preserve_unit_loops=True)",
        ],
    ]
    # pylint: enable=line-too-long
    ctx = create_context(
        create_prim_func(te_workload.norm_bmn(B=1, M=256, N=256)),
        target=_target(),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 3
    check_trace(spaces, expected)


def test_meta_schedule_cpu_sketch_softmax():  # pylint: disable=invalid-name
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="T_softmax_maxelem", func_name="main")',
            'b1 = sch.get_block(name="T_softmax_exp", func_name="main")',
            'b2 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            'b3 = sch.get_block(name="root", func_name="main")',
            "l4, l5 = sch.get_loops(block=b2)",
            "v6, v7 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)",
            "l8, l9 = sch.split(loop=l5, factors=[v6, v7])",
            "b10 = sch.rfactor(loop=l8, factor_axis=1)",
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.random_compute_producer", ann_val=1)',
            "l11, l12 = sch.get_loops(block=b0)",
            "v13, v14 = sch.sample_perfect_tile(loop=l12, n=2, max_innermost_factor=64)",
            "l15, l16 = sch.split(loop=l12, factors=[v13, v14])",
            "b17 = sch.rfactor(loop=l15, factor_axis=1)",
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)',
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v18 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v18)',
            "b19, = sch.get_producers(block=b2)",
            'sch.unannotate(block_or_loop=b2, ann_key="meta_schedule.random_compute_producer")',
            "l20 = sch.sample_compute_location(block=b2)",
            "sch.compute_at(block=b2, loop=l20, preserve_unit_loops=True)",
            "l21 = sch.sample_compute_location(block=b19)",
            "sch.compute_at(block=b19, loop=l21, preserve_unit_loops=True)",
            "l22 = sch.sample_compute_location(block=b1)",
            "sch.compute_at(block=b1, loop=l22, preserve_unit_loops=True)",
            "b23, = sch.get_producers(block=b0)",
            'sch.unannotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer")',
            "l24 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l24, preserve_unit_loops=True)",
            "l25 = sch.sample_compute_location(block=b23)",
            "sch.compute_at(block=b23, loop=l25, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="T_softmax_maxelem", func_name="main")',
            'b1 = sch.get_block(name="T_softmax_exp", func_name="main")',
            'b2 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            'b3 = sch.get_block(name="root", func_name="main")',
            "l4, l5 = sch.get_loops(block=b2)",
            "v6, v7 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)",
            "l8, l9 = sch.split(loop=l5, factors=[v6, v7])",
            "b10 = sch.rfactor(loop=l8, factor_axis=1)",
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.random_compute_producer", ann_val=1)',
            "l11, l12 = sch.get_loops(block=b0)",
            "v13, v14 = sch.sample_perfect_tile(loop=l12, n=2, max_innermost_factor=64)",
            "l15, l16 = sch.split(loop=l12, factors=[v13, v14])",
            "b17 = sch.rfactor(loop=l16, factor_axis=1)",
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)',
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v18 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v18)',
            "b19, = sch.get_producers(block=b2)",
            'sch.unannotate(block_or_loop=b2, ann_key="meta_schedule.random_compute_producer")',
            "l20 = sch.sample_compute_location(block=b2)",
            "sch.compute_at(block=b2, loop=l20, preserve_unit_loops=True)",
            "l21 = sch.sample_compute_location(block=b19)",
            "sch.compute_at(block=b19, loop=l21, preserve_unit_loops=True)",
            "l22 = sch.sample_compute_location(block=b1)",
            "sch.compute_at(block=b1, loop=l22, preserve_unit_loops=True)",
            "b23, = sch.get_producers(block=b0)",
            'sch.unannotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer")',
            "l24 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l24, preserve_unit_loops=True)",
            "l25 = sch.sample_compute_location(block=b23)",
            "sch.compute_at(block=b23, loop=l25, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="T_softmax_maxelem", func_name="main")',
            'b1 = sch.get_block(name="T_softmax_exp", func_name="main")',
            'b2 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            'b3 = sch.get_block(name="root", func_name="main")',
            "l4, l5 = sch.get_loops(block=b2)",
            "v6, v7 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)",
            "l8, l9 = sch.split(loop=l5, factors=[v6, v7])",
            "b10 = sch.rfactor(loop=l8, factor_axis=1)",
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.random_compute_producer", ann_val=1)',
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v11 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v11)',
            "b12, = sch.get_producers(block=b2)",
            'sch.unannotate(block_or_loop=b2, ann_key="meta_schedule.random_compute_producer")',
            "l13 = sch.sample_compute_location(block=b2)",
            "sch.compute_at(block=b2, loop=l13, preserve_unit_loops=True)",
            "l14 = sch.sample_compute_location(block=b12)",
            "sch.compute_at(block=b12, loop=l14, preserve_unit_loops=True)",
            "l15 = sch.sample_compute_location(block=b1)",
            "sch.compute_at(block=b1, loop=l15, preserve_unit_loops=True)",
            "l16 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l16, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="T_softmax_maxelem", func_name="main")',
            'b1 = sch.get_block(name="T_softmax_exp", func_name="main")',
            'b2 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            'b3 = sch.get_block(name="root", func_name="main")',
            "l4, l5 = sch.get_loops(block=b2)",
            "v6, v7 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)",
            "l8, l9 = sch.split(loop=l5, factors=[v6, v7])",
            "b10 = sch.rfactor(loop=l9, factor_axis=1)",
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.random_compute_producer", ann_val=1)',
            "l11, l12 = sch.get_loops(block=b0)",
            "v13, v14 = sch.sample_perfect_tile(loop=l12, n=2, max_innermost_factor=64)",
            "l15, l16 = sch.split(loop=l12, factors=[v13, v14])",
            "b17 = sch.rfactor(loop=l15, factor_axis=1)",
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)',
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v18 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v18)',
            "b19, = sch.get_producers(block=b2)",
            'sch.unannotate(block_or_loop=b2, ann_key="meta_schedule.random_compute_producer")',
            "l20 = sch.sample_compute_location(block=b2)",
            "sch.compute_at(block=b2, loop=l20, preserve_unit_loops=True)",
            "l21 = sch.sample_compute_location(block=b19)",
            "sch.compute_at(block=b19, loop=l21, preserve_unit_loops=True)",
            "l22 = sch.sample_compute_location(block=b1)",
            "sch.compute_at(block=b1, loop=l22, preserve_unit_loops=True)",
            "b23, = sch.get_producers(block=b0)",
            'sch.unannotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer")',
            "l24 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l24, preserve_unit_loops=True)",
            "l25 = sch.sample_compute_location(block=b23)",
            "sch.compute_at(block=b23, loop=l25, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="T_softmax_maxelem", func_name="main")',
            'b1 = sch.get_block(name="T_softmax_exp", func_name="main")',
            'b2 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            'b3 = sch.get_block(name="root", func_name="main")',
            "l4, l5 = sch.get_loops(block=b2)",
            "v6, v7 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)",
            "l8, l9 = sch.split(loop=l5, factors=[v6, v7])",
            "b10 = sch.rfactor(loop=l9, factor_axis=1)",
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.random_compute_producer", ann_val=1)',
            "l11, l12 = sch.get_loops(block=b0)",
            "v13, v14 = sch.sample_perfect_tile(loop=l12, n=2, max_innermost_factor=64)",
            "l15, l16 = sch.split(loop=l12, factors=[v13, v14])",
            "b17 = sch.rfactor(loop=l16, factor_axis=1)",
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)',
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v18 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v18)',
            "b19, = sch.get_producers(block=b2)",
            'sch.unannotate(block_or_loop=b2, ann_key="meta_schedule.random_compute_producer")',
            "l20 = sch.sample_compute_location(block=b2)",
            "sch.compute_at(block=b2, loop=l20, preserve_unit_loops=True)",
            "l21 = sch.sample_compute_location(block=b19)",
            "sch.compute_at(block=b19, loop=l21, preserve_unit_loops=True)",
            "l22 = sch.sample_compute_location(block=b1)",
            "sch.compute_at(block=b1, loop=l22, preserve_unit_loops=True)",
            "b23, = sch.get_producers(block=b0)",
            'sch.unannotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer")',
            "l24 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l24, preserve_unit_loops=True)",
            "l25 = sch.sample_compute_location(block=b23)",
            "sch.compute_at(block=b23, loop=l25, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="T_softmax_maxelem", func_name="main")',
            'b1 = sch.get_block(name="T_softmax_exp", func_name="main")',
            'b2 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            'b3 = sch.get_block(name="root", func_name="main")',
            "l4, l5 = sch.get_loops(block=b2)",
            "v6, v7 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)",
            "l8, l9 = sch.split(loop=l5, factors=[v6, v7])",
            "b10 = sch.rfactor(loop=l9, factor_axis=1)",
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.random_compute_producer", ann_val=1)',
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v11 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v11)',
            "b12, = sch.get_producers(block=b2)",
            'sch.unannotate(block_or_loop=b2, ann_key="meta_schedule.random_compute_producer")',
            "l13 = sch.sample_compute_location(block=b2)",
            "sch.compute_at(block=b2, loop=l13, preserve_unit_loops=True)",
            "l14 = sch.sample_compute_location(block=b12)",
            "sch.compute_at(block=b12, loop=l14, preserve_unit_loops=True)",
            "l15 = sch.sample_compute_location(block=b1)",
            "sch.compute_at(block=b1, loop=l15, preserve_unit_loops=True)",
            "l16 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l16, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="T_softmax_maxelem", func_name="main")',
            'b1 = sch.get_block(name="T_softmax_exp", func_name="main")',
            'b2 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            'b3 = sch.get_block(name="root", func_name="main")',
            "l4, l5 = sch.get_loops(block=b0)",
            "v6, v7 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)",
            "l8, l9 = sch.split(loop=l5, factors=[v6, v7])",
            "b10 = sch.rfactor(loop=l8, factor_axis=1)",
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)',
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v11 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v11)',
            "l12 = sch.sample_compute_location(block=b2)",
            "sch.compute_at(block=b2, loop=l12, preserve_unit_loops=True)",
            "l13 = sch.sample_compute_location(block=b1)",
            "sch.compute_at(block=b1, loop=l13, preserve_unit_loops=True)",
            "b14, = sch.get_producers(block=b0)",
            'sch.unannotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer")',
            "l15 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l15, preserve_unit_loops=True)",
            "l16 = sch.sample_compute_location(block=b14)",
            "sch.compute_at(block=b14, loop=l16, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="T_softmax_maxelem", func_name="main")',
            'b1 = sch.get_block(name="T_softmax_exp", func_name="main")',
            'b2 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            'b3 = sch.get_block(name="root", func_name="main")',
            "l4, l5 = sch.get_loops(block=b0)",
            "v6, v7 = sch.sample_perfect_tile(loop=l5, n=2, max_innermost_factor=64)",
            "l8, l9 = sch.split(loop=l5, factors=[v6, v7])",
            "b10 = sch.rfactor(loop=l9, factor_axis=1)",
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer", ann_val=1)',
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v11 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v11)',
            "l12 = sch.sample_compute_location(block=b2)",
            "sch.compute_at(block=b2, loop=l12, preserve_unit_loops=True)",
            "l13 = sch.sample_compute_location(block=b1)",
            "sch.compute_at(block=b1, loop=l13, preserve_unit_loops=True)",
            "b14, = sch.get_producers(block=b0)",
            'sch.unannotate(block_or_loop=b0, ann_key="meta_schedule.random_compute_producer")',
            "l15 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l15, preserve_unit_loops=True)",
            "l16 = sch.sample_compute_location(block=b14)",
            "sch.compute_at(block=b14, loop=l16, preserve_unit_loops=True)",
        ],
        [
            'b0 = sch.get_block(name="T_softmax_maxelem", func_name="main")',
            'b1 = sch.get_block(name="T_softmax_exp", func_name="main")',
            'b2 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            'b3 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.parallel", ann_val=256)',
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.vectorize", ann_val=32)',
            "v4 = sch.sample_categorical(candidates=[0, 16, 64, 512], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v4)',
            "l5 = sch.sample_compute_location(block=b2)",
            "sch.compute_at(block=b2, loop=l5, preserve_unit_loops=True)",
            "l6 = sch.sample_compute_location(block=b1)",
            "sch.compute_at(block=b1, loop=l6, preserve_unit_loops=True)",
            "l7 = sch.sample_compute_location(block=b0)",
            "sch.compute_at(block=b0, loop=l7, preserve_unit_loops=True)",
        ],
    ]
    # pylint: enable=line-too-long
    ctx = create_context(
        create_prim_func(te_workload.softmax_mn(m=256, n=256)),
        target=_target(),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 9
    check_trace(spaces, expected)


if __name__ == "__main__":
    test_meta_schedule_cpu_sketch_matmul()
    test_meta_schedule_cpu_sketch_matmul_relu()
    test_meta_schedule_cpu_sketch_conv2d_nchw()
    test_meta_schedule_cpu_sketch_conv2d_nchw_bias_bn_relu()
    test_meta_schedule_sketch_cpu_max_pool2d_nchw()
    test_meta_schedule_cpu_sketch_batchnorm()
    test_meta_schedule_cpu_sketch_softmax()
