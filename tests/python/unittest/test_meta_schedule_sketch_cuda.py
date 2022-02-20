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

from tvm.meta_schedule.testing import te_workload
from tvm.meta_schedule.testing.space_generation import check_trace, create_context
from tvm.target import Target
from tvm.te import create_prim_func


def _target() -> Target:
    return Target("cuda", host="llvm")


def _target_with_max_threads_per_block() -> Target:
    return Target("nvidia/geforce-rtx-3080")


def test_meta_schedule_cuda_sketch_matmul():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")',
            'b2 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")',
            "l3, l4, l5 = sch.get_loops(block=b0)",
            "v6, v7, v8, v9, v10 = sch.sample_perfect_tile(loop=l3, n=5, max_innermost_factor=64)",
            "l11, l12, l13, l14, l15 = sch.split(loop=l3, factors=[v6, v7, v8, v9, v10])",
            "v16, v17, v18, v19, v20 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64)",
            "l21, l22, l23, l24, l25 = sch.split(loop=l4, factors=[v16, v17, v18, v19, v20])",
            "v26, v27, v28 = sch.sample_perfect_tile(loop=l5, n=3, max_innermost_factor=64)",
            "l29, l30, l31 = sch.split(loop=l5, factors=[v26, v27, v28])",
            "sch.reorder(l11, l21, l12, l22, l13, l23, l29, l30, l14, l24, l31, l15, l25)",
            "l32 = sch.fuse(l11, l21)",
            'sch.bind(loop=l32, thread_axis="blockIdx.x")',
            "l33 = sch.fuse(l12, l22)",
            'sch.bind(loop=l33, thread_axis="vthread.x")',
            "l34 = sch.fuse(l13, l23)",
            'sch.bind(loop=l34, thread_axis="threadIdx.x")',
            'b35 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared")',
            "sch.compute_at(block=b35, loop=l29, preserve_unit_loops=True)",
            "l36, l37, l38, l39, l40, l41 = sch.get_loops(block=b35)",
            "l42 = sch.fuse(l40, l41)",
            "v43 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b35, ann_key="meta_schedule.cooperative_fetch", ann_val=v43)',
            'b44 = sch.cache_read(block=b0, read_buffer_index=2, storage_scope="shared")',
            "sch.compute_at(block=b44, loop=l29, preserve_unit_loops=True)",
            "l45, l46, l47, l48, l49, l50 = sch.get_loops(block=b44)",
            "l51 = sch.fuse(l49, l50)",
            "v52 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b44, ann_key="meta_schedule.cooperative_fetch", ann_val=v52)',
            "sch.reverse_compute_at(block=b2, loop=l34, preserve_unit_loops=True)",
            "v53 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v53)',
        ]
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
    assert len(spaces) == 1
    check_trace(spaces, expected)


def test_meta_schedule_cuda_sketch_matmul_relu():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="compute", func_name="main")',
            'b2 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")',
            'b3 = sch.cache_write(block=b0, write_buffer_index=0, storage_scope="local")',
            "l4, l5, l6 = sch.get_loops(block=b0)",
            "v7, v8, v9, v10, v11 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64)",
            "l12, l13, l14, l15, l16 = sch.split(loop=l4, factors=[v7, v8, v9, v10, v11])",
            "v17, v18, v19, v20, v21 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64)",
            "l22, l23, l24, l25, l26 = sch.split(loop=l5, factors=[v17, v18, v19, v20, v21])",
            "v27, v28, v29 = sch.sample_perfect_tile(loop=l6, n=3, max_innermost_factor=64)",
            "l30, l31, l32 = sch.split(loop=l6, factors=[v27, v28, v29])",
            "sch.reorder(l12, l22, l13, l23, l14, l24, l30, l31, l15, l25, l32, l16, l26)",
            "l33 = sch.fuse(l12, l22)",
            'sch.bind(loop=l33, thread_axis="blockIdx.x")',
            "l34 = sch.fuse(l13, l23)",
            'sch.bind(loop=l34, thread_axis="vthread.x")',
            "l35 = sch.fuse(l14, l24)",
            'sch.bind(loop=l35, thread_axis="threadIdx.x")',
            'b36 = sch.cache_read(block=b0, read_buffer_index=1, storage_scope="shared")',
            "sch.compute_at(block=b36, loop=l30, preserve_unit_loops=True)",
            "l37, l38, l39, l40, l41, l42 = sch.get_loops(block=b36)",
            "l43 = sch.fuse(l41, l42)",
            "v44 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b36, ann_key="meta_schedule.cooperative_fetch", ann_val=v44)',
            'b45 = sch.cache_read(block=b0, read_buffer_index=2, storage_scope="shared")',
            "sch.compute_at(block=b45, loop=l30, preserve_unit_loops=True)",
            "l46, l47, l48, l49, l50, l51 = sch.get_loops(block=b45)",
            "l52 = sch.fuse(l50, l51)",
            "v53 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b45, ann_key="meta_schedule.cooperative_fetch", ann_val=v53)',
            "sch.reverse_compute_at(block=b3, loop=l35, preserve_unit_loops=True)",
            "sch.reverse_compute_inline(block=b1)",
            "v54 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001])",
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v54)',
        ]
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
    assert len(spaces) == 1
    check_trace(spaces, expected)


def test_meta_schedule_cuda_sketch_conv2d_nchw():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="pad_temp", func_name="main")',
            'b1 = sch.get_block(name="conv2d_nchw", func_name="main")',
            'b2 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")',
            'b3 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")',
            "l4, l5, l6, l7, l8, l9, l10 = sch.get_loops(block=b1)",
            "v11, v12, v13, v14, v15 = sch.sample_perfect_tile(loop=l4, n=5, max_innermost_factor=64)",
            "l16, l17, l18, l19, l20 = sch.split(loop=l4, factors=[v11, v12, v13, v14, v15])",
            "v21, v22, v23, v24, v25 = sch.sample_perfect_tile(loop=l5, n=5, max_innermost_factor=64)",
            "l26, l27, l28, l29, l30 = sch.split(loop=l5, factors=[v21, v22, v23, v24, v25])",
            "v31, v32, v33, v34, v35 = sch.sample_perfect_tile(loop=l6, n=5, max_innermost_factor=64)",
            "l36, l37, l38, l39, l40 = sch.split(loop=l6, factors=[v31, v32, v33, v34, v35])",
            "v41, v42, v43, v44, v45 = sch.sample_perfect_tile(loop=l7, n=5, max_innermost_factor=64)",
            "l46, l47, l48, l49, l50 = sch.split(loop=l7, factors=[v41, v42, v43, v44, v45])",
            "v51, v52, v53 = sch.sample_perfect_tile(loop=l8, n=3, max_innermost_factor=64)",
            "l54, l55, l56 = sch.split(loop=l8, factors=[v51, v52, v53])",
            "v57, v58, v59 = sch.sample_perfect_tile(loop=l9, n=3, max_innermost_factor=64)",
            "l60, l61, l62 = sch.split(loop=l9, factors=[v57, v58, v59])",
            "v63, v64, v65 = sch.sample_perfect_tile(loop=l10, n=3, max_innermost_factor=64)",
            "l66, l67, l68 = sch.split(loop=l10, factors=[v63, v64, v65])",
            "sch.reorder(l16, l26, l36, l46, l17, l27, l37, l47, l18, l28, l38, l48, l54, l60, l66, l55, l61, l67, l19, l29, l39, l49, l56, l62, l68, l20, l30, l40, l50)",
            "l69 = sch.fuse(l16, l26, l36, l46)",
            'sch.bind(loop=l69, thread_axis="blockIdx.x")',
            "l70 = sch.fuse(l17, l27, l37, l47)",
            'sch.bind(loop=l70, thread_axis="vthread.x")',
            "l71 = sch.fuse(l18, l28, l38, l48)",
            'sch.bind(loop=l71, thread_axis="threadIdx.x")',
            'b72 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="shared")',
            "sch.compute_at(block=b72, loop=l66, preserve_unit_loops=True)",
            "l73, l74, l75, l76, l77, l78, l79, l80, l81, l82 = sch.get_loops(block=b72)",
            "l83 = sch.fuse(l79, l80, l81, l82)",
            "v84 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b72, ann_key="meta_schedule.cooperative_fetch", ann_val=v84)',
            'b85 = sch.cache_read(block=b1, read_buffer_index=2, storage_scope="shared")',
            "sch.compute_at(block=b85, loop=l66, preserve_unit_loops=True)",
            "l86, l87, l88, l89, l90, l91, l92, l93, l94, l95 = sch.get_loops(block=b85)",
            "l96 = sch.fuse(l92, l93, l94, l95)",
            "v97 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b85, ann_key="meta_schedule.cooperative_fetch", ann_val=v97)',
            "sch.reverse_compute_at(block=b3, loop=l71, preserve_unit_loops=True)",
            "sch.compute_inline(block=b0)",
            "v98 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001])",
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v98)',
        ]
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
    assert len(spaces) == 1
    check_trace(spaces, expected)


def test_meta_schedule_cuda_sketch_conv2d_nchw_bias_bn_relu():  # pylint: disable=invalid-name
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="pad_temp", func_name="main")',
            'b1 = sch.get_block(name="conv2d_nchw", func_name="main")',
            'b2 = sch.get_block(name="bias_add", func_name="main")',
            'b3 = sch.get_block(name="bn_mul", func_name="main")',
            'b4 = sch.get_block(name="bn_add", func_name="main")',
            'b5 = sch.get_block(name="compute", func_name="main")',
            'b6 = sch.get_block(name="root", func_name="main")',
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")',
            'b7 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")',
            "l8, l9, l10, l11, l12, l13, l14 = sch.get_loops(block=b1)",
            "v15, v16, v17, v18, v19 = sch.sample_perfect_tile(loop=l8, n=5, max_innermost_factor=64)",
            "l20, l21, l22, l23, l24 = sch.split(loop=l8, factors=[v15, v16, v17, v18, v19])",
            "v25, v26, v27, v28, v29 = sch.sample_perfect_tile(loop=l9, n=5, max_innermost_factor=64)",
            "l30, l31, l32, l33, l34 = sch.split(loop=l9, factors=[v25, v26, v27, v28, v29])",
            "v35, v36, v37, v38, v39 = sch.sample_perfect_tile(loop=l10, n=5, max_innermost_factor=64)",
            "l40, l41, l42, l43, l44 = sch.split(loop=l10, factors=[v35, v36, v37, v38, v39])",
            "v45, v46, v47, v48, v49 = sch.sample_perfect_tile(loop=l11, n=5, max_innermost_factor=64)",
            "l50, l51, l52, l53, l54 = sch.split(loop=l11, factors=[v45, v46, v47, v48, v49])",
            "v55, v56, v57 = sch.sample_perfect_tile(loop=l12, n=3, max_innermost_factor=64)",
            "l58, l59, l60 = sch.split(loop=l12, factors=[v55, v56, v57])",
            "v61, v62, v63 = sch.sample_perfect_tile(loop=l13, n=3, max_innermost_factor=64)",
            "l64, l65, l66 = sch.split(loop=l13, factors=[v61, v62, v63])",
            "v67, v68, v69 = sch.sample_perfect_tile(loop=l14, n=3, max_innermost_factor=64)",
            "l70, l71, l72 = sch.split(loop=l14, factors=[v67, v68, v69])",
            "sch.reorder(l20, l30, l40, l50, l21, l31, l41, l51, l22, l32, l42, l52, l58, l64, l70, l59, l65, l71, l23, l33, l43, l53, l60, l66, l72, l24, l34, l44, l54)",
            "l73 = sch.fuse(l20, l30, l40, l50)",
            'sch.bind(loop=l73, thread_axis="blockIdx.x")',
            "l74 = sch.fuse(l21, l31, l41, l51)",
            'sch.bind(loop=l74, thread_axis="vthread.x")',
            "l75 = sch.fuse(l22, l32, l42, l52)",
            'sch.bind(loop=l75, thread_axis="threadIdx.x")',
            'b76 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="shared")',
            "sch.compute_at(block=b76, loop=l70, preserve_unit_loops=True)",
            "l77, l78, l79, l80, l81, l82, l83, l84, l85, l86 = sch.get_loops(block=b76)",
            "l87 = sch.fuse(l83, l84, l85, l86)",
            "v88 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b76, ann_key="meta_schedule.cooperative_fetch", ann_val=v88)',
            'b89 = sch.cache_read(block=b1, read_buffer_index=2, storage_scope="shared")',
            "sch.compute_at(block=b89, loop=l70, preserve_unit_loops=True)",
            "l90, l91, l92, l93, l94, l95, l96, l97, l98, l99 = sch.get_loops(block=b89)",
            "l100 = sch.fuse(l96, l97, l98, l99)",
            "v101 = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])",
            'sch.annotate(block_or_loop=b89, ann_key="meta_schedule.cooperative_fetch", ann_val=v101)',
            "sch.reverse_compute_at(block=b7, loop=l75, preserve_unit_loops=True)",
            "sch.reverse_compute_inline(block=b5)",
            "sch.reverse_compute_inline(block=b4)",
            "sch.reverse_compute_inline(block=b3)",
            "sch.reverse_compute_inline(block=b2)",
            "sch.compute_inline(block=b0)",
            "v102 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001])",
            'sch.annotate(block_or_loop=b6, ann_key="meta_schedule.unroll_explicit", ann_val=v102)',
        ]
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
    assert len(spaces) == 1
    check_trace(spaces, expected)


def test_meta_schedule_cuda_sketch_batchnorm():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="C", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            "b2, = sch.get_consumers(block=b0)",
            "l3, = sch.get_loops(block=b2)",
            "v4 = sch.sample_categorical(candidates=[4, 8, 16, 32, 64, 128, 256, 512], probs=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])",
            "l5, l6 = sch.split(loop=l3, factors=[None, v4])",
            'sch.bind(loop=l6, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b0, loop=l5, preserve_unit_loops=True)",
            'sch.set_scope(block=b0, buffer_index=0, storage_scope="shared")',
            "l7, l8, l9, l10 = sch.get_loops(block=b0)",
            "l11 = sch.fuse(l9, l10)",
            "l12, l13 = sch.split(loop=l11, factors=[None, v4])",
            'sch.bind(loop=l13, thread_axis="threadIdx.x")',
            "v14 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v14)',
        ],
        [
            'b0 = sch.get_block(name="root", func_name="main")',
            "v1 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001])",
            'sch.annotate(block_or_loop=b0, ann_key="meta_schedule.unroll_explicit", ann_val=v1)',
        ],
    ]
    # pylint: enable=line-too-long
    ctx = create_context(
        create_prim_func(
            te_workload.norm_bmn(
                B=1,
                M=256,
                N=256,
            )
        ),
        target=_target_with_max_threads_per_block(),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 2
    check_trace(spaces, expected)


def test_meta_schedule_cuda_sketch_softmax():
    # pylint: disable=line-too-long
    expected = [
        [
            'b0 = sch.get_block(name="T_softmax_maxelem", func_name="main")',
            'b1 = sch.get_block(name="T_softmax_exp", func_name="main")',
            'b2 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            'b3 = sch.get_block(name="root", func_name="main")',
            "sch.compute_inline(block=b1)",
            "b4, = sch.get_consumers(block=b2)",
            "l5, l6 = sch.get_loops(block=b4)",
            "v7 = sch.sample_categorical(candidates=[4, 8, 16, 32, 64, 128, 256, 512], probs=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])",
            "l8, l9 = sch.split(loop=l6, factors=[None, v7])",
            'sch.bind(loop=l9, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b2, loop=l5, preserve_unit_loops=True)",
            'sch.set_scope(block=b2, buffer_index=0, storage_scope="shared")',
            "l10, l11, l12 = sch.get_loops(block=b2)",
            "l13, l14 = sch.split(loop=l12, factors=[None, v7])",
            'sch.bind(loop=l14, thread_axis="threadIdx.x")',
            "b15, b16 = sch.get_consumers(block=b0)",
            "l17, l18, l19, l20 = sch.get_loops(block=b15)",
            "sch.compute_at(block=b0, loop=l17, preserve_unit_loops=True)",
            'sch.set_scope(block=b0, buffer_index=0, storage_scope="shared")',
            "l21, l22, l23 = sch.get_loops(block=b0)",
            "l24, l25 = sch.split(loop=l23, factors=[None, v7])",
            'sch.bind(loop=l25, thread_axis="threadIdx.x")',
            "v26 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001])",
            'sch.annotate(block_or_loop=b3, ann_key="meta_schedule.unroll_explicit", ann_val=v26)',
        ],
        [
            'b0 = sch.get_block(name="T_softmax_exp", func_name="main")',
            'b1 = sch.get_block(name="T_softmax_expsum", func_name="main")',
            'b2 = sch.get_block(name="root", func_name="main")',
            "sch.compute_inline(block=b0)",
            "b3, = sch.get_consumers(block=b1)",
            "l4, l5 = sch.get_loops(block=b3)",
            "v6 = sch.sample_categorical(candidates=[4, 8, 16, 32, 64, 128, 256, 512], probs=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])",
            "l7, l8 = sch.split(loop=l5, factors=[None, v6])",
            'sch.bind(loop=l8, thread_axis="threadIdx.x")',
            "sch.compute_at(block=b1, loop=l4, preserve_unit_loops=True)",
            'sch.set_scope(block=b1, buffer_index=0, storage_scope="shared")',
            "l9, l10, l11 = sch.get_loops(block=b1)",
            "l12, l13 = sch.split(loop=l11, factors=[None, v6])",
            'sch.bind(loop=l13, thread_axis="threadIdx.x")',
            "v14 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001])",
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v14)',
        ],
        [
            'b0 = sch.get_block(name="T_softmax_maxelem", func_name="main")',
            'b1 = sch.get_block(name="T_softmax_exp", func_name="main")',
            'b2 = sch.get_block(name="root", func_name="main")',
            "sch.compute_inline(block=b1)",
            "v3 = sch.sample_categorical(candidates=[4, 8, 16, 32, 64, 128, 256, 512], probs=[0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])",
            "l4, l5 = sch.get_loops(block=b0)",
            "l6, l7 = sch.split(loop=l5, factors=[None, v3])",
            'sch.bind(loop=l7, thread_axis="threadIdx.x")',
            "v8 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001])",
            'sch.annotate(block_or_loop=b2, ann_key="meta_schedule.unroll_explicit", ann_val=v8)',
        ],
        [
            'b0 = sch.get_block(name="T_softmax_exp", func_name="main")',
            'b1 = sch.get_block(name="root", func_name="main")',
            "sch.compute_inline(block=b0)",
            "v2 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001])",
            'sch.annotate(block_or_loop=b1, ann_key="meta_schedule.unroll_explicit", ann_val=v2)',
        ],
    ]
    # pylint: enable=line-too-long
    ctx = create_context(
        create_prim_func(
            te_workload.softmax_mn(
                m=256,
                n=256,
            )
        ),
        target=_target_with_max_threads_per_block(),
    )
    spaces = ctx.space_generator.generate_design_space(mod=ctx.mod)
    assert len(spaces) == 4
    check_trace(spaces, expected)


if __name__ == "__main__":
    test_meta_schedule_cuda_sketch_matmul()
    test_meta_schedule_cuda_sketch_matmul_relu()
    test_meta_schedule_cuda_sketch_conv2d_nchw()
    test_meta_schedule_cuda_sketch_conv2d_nchw_bias_bn_relu()
    test_meta_schedule_cuda_sketch_batchnorm()
    test_meta_schedule_cuda_sketch_softmax()
