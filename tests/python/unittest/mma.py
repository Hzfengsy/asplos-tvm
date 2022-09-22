@tvm.script.ir_module
class Module:
    @tir.prim_func
    def main(X_1: tir.Buffer[(4096, 4096), "float16"], Y_1: tir.Buffer[(4096, 4096), "float16"], C_2: tir.Buffer[(4096, 4096), "float16"]) -> None:
        # function attr dict
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        # var definition
        tx = tir.env_thread("threadIdx.x")
        tx = tir.env_thread("threadIdx.x")
        tx = tir.env_thread("threadIdx.x")
        a0_1 = tir.var("int32")
        a1_1 = tir.var("int32")
        b0_1 = tir.var("int32")
        b1_1 = tir.var("int32")
        c0_1 = tir.var("int32")
        c1_1 = tir.var("int32")
        d0_2 = tir.var("int32")
        d0_3 = tir.var("int32")
        d1_2 = tir.var("int32")
        d1_3 = tir.var("int32")
        s0_2 = tir.var("int32")
        s0_3 = tir.var("int32")
        s1_2 = tir.var("int32")
        s1_3 = tir.var("int32")
        # body
        with tir.block("root"):
            tir.reads()
            tir.writes()
            tir.block_attr({"warp_execution":1})
            X_shared_dyn_1 = tir.alloc_buffer([4096, 4096], dtype="float16", scope="shared.dyn")
            Y_shared_dyn_1 = tir.alloc_buffer([4096, 4096], dtype="float16", scope="shared.dyn")
            X_shared_dyn_m16n8k8_matrixA_1 = tir.alloc_buffer([4096, 4096], dtype="float16", scope="m16n8k8.matrixA")
            Y_shared_dyn_m16n8k8_matrixB_1 = tir.alloc_buffer([4096, 4096], dtype="float16", scope="m16n8k8.matrixB")
            C_m16n8k8_matrixC_1 = tir.alloc_buffer([4096, 4096], dtype="float16", scope="m16n8k8.matrixC")
            for i0_0_0_i1_0_0_fused in tir.thread_binding(4, thread="blockIdx.y"):
                for i0_0_1_i1_0_1_fused in tir.thread_binding(256, thread="blockIdx.x"):
                    for i1_0_2_i0_0_2_fused in tir.thread_binding(4, thread="threadIdx.y"):
                        for i0_0_3_init, i1_0_3_init, i0_0_4_init, i1_0_4_init in tir.grid(1, 1, 4, 8):
                            with tir.block("C_o_init"):
                                i_o = tir.axis.spatial(256, ((0 * 32 + i0_0_1_i1_0_1_fused // 8) * 2 + i1_0_2_i0_0_2_fused % 2 + i0_0_3_init) * 4 + i0_0_4_init)
                                j_o = tir.axis.spatial(512, ((i0_0_0_i1_0_0_fused % 4 * 8 + i0_0_1_i1_0_1_fused % 8) * 2 + i1_0_2_i0_0_2_fused // 2 + i1_0_3_init) * 8 + i1_0_4_init)
                                tir.reads()
                                tir.writes(C_m16n8k8_matrixC_1[i_o * 16 : i_o * 16 + 16, j_o * 8 : j_o * 8 + 8])
                                for i0_1_0 in tir.serial(2):
                                    with tir.block("C_init_o"):
                                        i_init_o = tir.axis.spatial(2, i0_1_0)
                                        j_init_o = tir.axis.spatial(1, 0)
                                        tir.reads()
                                        tir.writes(C_m16n8k8_matrixC_1[i_o * 16 + i_init_o * 8 : i_o * 16 + i_init_o * 8 + 8, j_o * 8 : j_o * 8 + 8])
                                        dst_3 = tir.match_buffer(C_m16n8k8_matrixC_1[i_o * 16 + i_init_o * 8 : i_o * 16 + i_init_o * 8 + 8, j_o * 8 : j_o * 8 + 8], [8, 8], dtype="float16", scope="m16n8k8.matrixC", offset_factor=1)
                                        tir.launch_thread(tx, 32)
                                        for i in tir.vectorized(2):
                                            dst_3[tx // 4, tx % 4 * 2 + i] = tir.float16(0)
                        for i2_0_0 in tir.serial(128):
                            with tir.block("X_shared.dyn"):
                                v0, v1 = tir.axis.remap("SS", [i0_0_1_i1_0_1_fused, i2_0_0])
                                tir.reads(X_1[v0 // 8 * 128 : v0 // 8 * 128 + 128, v1 * 32 : v1 * 32 + 32])
                                tir.writes(X_shared_dyn_1[v0 // 8 * 128 : v0 // 8 * 128 + 128, v1 * 32 : v1 * 32 + 32])
                                tir.block_attr({"auto_copy":1, "double_buffer_scope":0, "local_stage":1, "meta_schedule.cache_type":0, "vector_bytes":16})
                                for ax0, ax1 in tir.grid(128, 32):
                                    X_shared_dyn_1[v0 // 8 * 128 + ax0, v1 * 32 + ax1] = X_1[v0 // 8 * 128 + ax0, v1 * 32 + ax1]
                            with tir.block("Y_shared.dyn"):
                                v0, v1, v2 = tir.axis.remap("SSS", [i2_0_0, i0_0_0_i1_0_0_fused, i0_0_1_i1_0_1_fused])
                                tir.reads(Y_1[v0 * 32 : v0 * 32 + 32, v1 % 4 * 1024 + v2 % 8 * 128 : v1 % 4 * 1024 + v2 % 8 * 128 + 128])
                                tir.writes(Y_shared_dyn_1[v0 * 32 : v0 * 32 + 32, v1 % 4 * 1024 + v2 % 8 * 128 : v1 % 4 * 1024 + v2 % 8 * 128 + 128])
                                tir.block_attr({"auto_copy":1, "double_buffer_scope":0, "local_stage":1, "meta_schedule.cache_type":0, "vector_bytes":16})
                                for ax0, ax1 in tir.grid(32, 128):
                                    Y_shared_dyn_1[v0 * 32 + ax0, v1 % 4 * 1024 + v2 % 8 * 128 + ax1] = Y_1[v0 * 32 + ax0, v1 % 4 * 1024 + v2 % 8 * 128 + ax1]
                            for i2_0_1 in tir.serial(4):
                                for ax0_0, ax1_0 in tir.grid(2, 1):
                                    with tir.block("X_shared.dyn_m16n8k8.matrixA_o"):
                                        v0_o = tir.axis.spatial(128, i0_0_1_i1_0_1_fused // 8 * 4 + i1_0_2_i0_0_2_fused % 2 * 2 + ax0_0)
                                        v1_o = tir.axis.spatial(512, i2_0_0 * 4 + i2_0_1)
                                        tir.reads(X_shared_dyn_1[v0_o * 32 : v0_o * 32 + 32, v1_o * 8 : v1_o * 8 + 8])
                                        tir.writes(X_shared_dyn_m16n8k8_matrixA_1[v0_o * 32 : v0_o * 32 + 32, v1_o * 8 : v1_o * 8 + 8])
                                        src_2 = tir.match_buffer(X_shared_dyn_1[v0_o * 32 : v0_o * 32 + 32, v1_o * 8 : v1_o * 8 + 8], [32, 8], dtype="float16", strides=[s0_2, s1_2], scope="shared.dyn", offset_factor=1)
                                        dst_4 = tir.match_buffer(X_shared_dyn_m16n8k8_matrixA_1[v0_o * 32 : v0_o * 32 + 32, v1_o * 8 : v1_o * 8 + 8], [32, 8], dtype="float16", strides=[d0_2, d1_2], scope="m16n8k8.matrixA", offset_factor=1)
                                        tir.launch_thread(tx, 32)
                                        tir.evaluate(tir.ptx_ldmatrix(False, 4, ".b16", dst_4.data, (dst_4.elem_offset // d0_2 // 32 * (d0_2 // 8) + dst_4.elem_offset % d0_2 // 8) * 8 + dst_4.elem_offset // d0_2 % 32 // 16 * 4, tir.tvm_access_ptr(tir.type_annotation(dtype="float16"), src_2.data, src_2.elem_offset, s0_2 * 32, 1, dtype="handle"), tx * s0_2, dtype="float16"))
                                for ax0_0, ax1_0 in tir.grid(1, 2):
                                    with tir.block("Y_shared.dyn_m16n8k8.matrixB_o"):
                                        v0_o = tir.axis.spatial(512, i2_0_0 * 4 + i2_0_1)
                                        v1_o = tir.axis.spatial(128, i0_0_0_i1_0_0_fused * 32 + i0_0_1_i1_0_1_fused % 8 * 4 + i1_0_2_i0_0_2_fused // 2 * 2 + ax1_0)
                                        tir.reads(Y_shared_dyn_1[v0_o * 8 : v0_o * 8 + 8, v1_o * 32 : v1_o * 32 + 32])
                                        tir.writes(Y_shared_dyn_m16n8k8_matrixB_1[v0_o * 8 : v0_o * 8 + 8, v1_o * 32 : v1_o * 32 + 32])
                                        src_3 = tir.match_buffer(Y_shared_dyn_1[v0_o * 8 : v0_o * 8 + 8, v1_o * 32 : v1_o * 32 + 32], [8, 32], dtype="float16", strides=[s0_3, s1_3], scope="shared.dyn", offset_factor=1)
                                        dst_5 = tir.match_buffer(Y_shared_dyn_m16n8k8_matrixB_1[v0_o * 8 : v0_o * 8 + 8, v1_o * 32 : v1_o * 32 + 32], [8, 32], dtype="float16", strides=[d0_3, d1_3], scope="m16n8k8.matrixB", offset_factor=1)
                                        tir.launch_thread(tx, 32)
                                        tir.evaluate(tir.ptx_ldmatrix(True, 4, ".b16", dst_5.data, (dst_5.elem_offset // d0_3 // 8 * (d0_3 // 32) + dst_5.elem_offset % d0_3 // 32) * 8 + dst_5.elem_offset % d0_3 % 32 // 8 * 2, tir.tvm_access_ptr(tir.type_annotation(dtype="float16"), src_3.data, src_3.elem_offset, s0_3 * 8, 1, dtype="handle"), s0_3 * (tx % 4) + 8 * (tx // 4), dtype="float16"))
                                for i0_0_3, i1_0_3, i2_0_2, i0_0_4, i1_0_4 in tir.grid(1, 1, 1, 4, 8):
                                    with tir.block("C_o_update"):
                                        i_o = tir.axis.spatial(256, ((0 * 32 + i0_0_1_i1_0_1_fused // 8) * 2 + i1_0_2_i0_0_2_fused % 2 + i0_0_3) * 4 + i0_0_4)
                                        j_o = tir.axis.spatial(512, ((i0_0_0_i1_0_0_fused % 4 * 8 + i0_0_1_i1_0_1_fused % 8) * 2 + i1_0_2_i0_0_2_fused // 2 + i1_0_3) * 8 + i1_0_4)
                                        k_o = tir.axis.reduce(512, i2_0_0 * 4 + i2_0_1 + i2_0_2)
                                        tir.reads(C_m16n8k8_matrixC_1[i_o * 16 : i_o * 16 + 16, j_o * 8 : j_o * 8 + 8], X_shared_dyn_m16n8k8_matrixA_1[i_o * 16 : i_o * 16 + 16, k_o * 8 : k_o * 8 + 8], Y_shared_dyn_m16n8k8_matrixB_1[k_o * 8 : k_o * 8 + 8, j_o * 8 : j_o * 8 + 8])
                                        tir.writes(C_m16n8k8_matrixC_1[i_o * 16 : i_o * 16 + 16, j_o * 8 : j_o * 8 + 8])
                                        with tir.block("C_o"):
                                            i_o_2 = tir.axis.spatial(1, 0)
                                            j_o_2 = tir.axis.spatial(1, 0)
                                            k_o_2 = tir.axis.reduce(1, 0)
                                            tir.reads(C_m16n8k8_matrixC_1[i_o * 16 : i_o * 16 + 16, j_o * 8 : j_o * 8 + 8], X_shared_dyn_m16n8k8_matrixA_1[i_o * 16 : i_o * 16 + 16, k_o * 8 : k_o * 8 + 8], Y_shared_dyn_m16n8k8_matrixB_1[k_o * 8 : k_o * 8 + 8, j_o * 8 : j_o * 8 + 8])
                                            tir.writes(C_m16n8k8_matrixC_1[i_o * 16 : i_o * 16 + 16, j_o * 8 : j_o * 8 + 8])
                                            A_1 = tir.match_buffer(X_shared_dyn_m16n8k8_matrixA_1[i_o * 16 : i_o * 16 + 16, k_o * 8 : k_o * 8 + 8], [16, 8], dtype="float16", strides=[a0_1, a1_1], scope="m16n8k8.matrixA", offset_factor=1)
                                            B_1 = tir.match_buffer(Y_shared_dyn_m16n8k8_matrixB_1[k_o * 8 : k_o * 8 + 8, j_o * 8 : j_o * 8 + 8], [8, 8], dtype="float16", strides=[b0_1, b1_1], scope="m16n8k8.matrixB", offset_factor=1)
                                            C_3 = tir.match_buffer(C_m16n8k8_matrixC_1[i_o * 16 : i_o * 16 + 16, j_o * 8 : j_o * 8 + 8], [16, 8], dtype="float16", strides=[c0_1, c1_1], scope="m16n8k8.matrixC", offset_factor=1)
                                            tir.evaluate(tir.ptx_mma("m16n8k8", "row", "col", "fp16", "fp16", "fp16", A_1.data, (A_1.elem_offset // a0_1 // 32 * (a0_1 // 8) + A_1.elem_offset % a0_1 // 8) * 8 + A_1.elem_offset // a0_1 % 32 // 16 * 4, B_1.data, (B_1.elem_offset // b0_1 // 8 * (b0_1 // 32) + B_1.elem_offset % b0_1 // 32) * 8 + B_1.elem_offset % b0_1 % 32 // 8 * 2, C_3.data, (C_3.elem_offset // c0_1 // 8 * (c0_1 // 8) + C_3.elem_offset % c0_1 // 8) * 2, False, dtype="float16"))
                        with tir.block("C_m16n8k8.matrixC"):
                            v0, v1, v2 = tir.axis.remap("SSS", [i0_0_1_i1_0_1_fused, i1_0_2_i0_0_2_fused, i0_0_0_i1_0_0_fused])
                            tir.reads(C_m16n8k8_matrixC_1[v0 // 8 * 128 + v1 % 2 * 64 : v0 // 8 * 128 + v1 % 2 * 64 + 64, v2 % 4 * 1024 + v0 % 8 * 128 + v1 // 2 * 64 : v2 % 4 * 1024 + v0 % 8 * 128 + v1 // 2 * 64 + 64])
                            tir.writes(C_2[v0 // 8 * 128 + v1 % 2 * 64 : v0 // 8 * 128 + v1 % 2 * 64 + 64, v2 % 4 * 1024 + v0 % 8 * 128 + v1 // 2 * 64 : v2 % 4 * 1024 + v0 % 8 * 128 + v1 // 2 * 64 + 64])
                            tir.block_attr({"auto_copy":1})
                            for ax0, ax1 in tir.grid(64, 64):
                                C_2[v0 // 8 * 128 + v1 % 2 * 64 + ax0, v2 % 4 * 1024 + v0 % 8 * 128 + v1 // 2 * 64 + ax1] = C_m16n8k8_matrixC_1[v0 // 8 * 128 + v1 % 2 * 64 + ax0, v2 % 4 * 1024 + v0 % 8 * 128 + v1 // 2 * 64 + ax1]
    
@tvm.script.ir_module
class Module:
    @tir.prim_func
    def main(X_2: tir.Buffer[16777216, "float16"], Y_2: tir.Buffer[16777216, "float16"], C_2: tir.Buffer[16777216, "float16"]) -> None:
        # function attr dict
        tir.func_attr({"global_symbol": "main", "tir.noalias": True})
        # var definition
        threadIdx_x = tir.env_thread("threadIdx.x")
        threadIdx_y = tir.env_thread("threadIdx.y")
        blockIdx_x = tir.env_thread("blockIdx.x")
        blockIdx_y = tir.env_thread("blockIdx.y")
        C_m16n8k8_matrixC_3 = tir.buffer_var("float16", "m16n8k8.matrixC")
        X_shared_dyn_m16n8k8_matrixA_1 = tir.buffer_var("float16", "m16n8k8.matrixA")
        Y_shared_dyn_m16n8k8_matrixB_1 = tir.buffer_var("float16", "m16n8k8.matrixB")
        tir.preflattened_buffer(X_2, [4096, 4096], dtype="float16", data=X_2.data)
        tir.preflattened_buffer(Y_2, [4096, 4096], dtype="float16", data=Y_2.data)
        tir.preflattened_buffer(C_2, [4096, 4096], dtype="float16", data=C_2.data)
        # body
        tir.launch_thread(blockIdx_y, 4)
        C_m16n8k8_matrixC_2 = tir.allocate([64], "float16x2", "local")
        X_shared_dyn_local_1 = tir.allocate([4], "float16x8", "local")
        X_shared_dyn_2 = tir.allocate([4096], "float16", "shared.dyn")
        X_shared_dyn_3 = tir.buffer_decl([2048], dtype="float16", data=X_shared_dyn_2.data, scope="shared.dyn")
        Y_shared_dyn_local_1 = tir.allocate([4], "float16x8", "local")
        Y_shared_dyn_1 = tir.allocate([4096], "float16", "shared.dyn")
        tir.launch_thread(blockIdx_x, 256)
        tir.launch_thread(threadIdx_y, 4)
        tir.launch_thread(threadIdx_x, 32)
        for i0_0_4_init, i1_0_4_init, i0_1_0 in tir.grid(4, 8, 2):
            C_m16n8k8_matrixC_2[i0_0_4_init * 16 + i0_1_0 * 8 + i1_0_4_init] = tir.broadcast(tir.float16(0), 2)
        for i2_0_0 in tir.serial(128):
            for ax0_ax1_fused_0_cache in tir.serial(4):
                X_shared_dyn_local_1[ax0_ax1_fused_0_cache] = X_2[blockIdx_x // 8 * 524288 + ax0_ax1_fused_0_cache * 131072 + threadIdx_y * 32768 + threadIdx_x // 4 * 4096 + i2_0_0 * 32 + threadIdx_x % 4 * 8:blockIdx_x // 8 * 524288 + ax0_ax1_fused_0_cache * 131072 + threadIdx_y * 32768 + threadIdx_x // 4 * 4096 + i2_0_0 * 32 + threadIdx_x % 4 * 8 + 8]
            for ax0_ax1_fused_0 in tir.serial(4):
                X_shared_dyn_2[ax0_ax1_fused_0 * 1024 + threadIdx_y * 256 + threadIdx_x * 8:ax0_ax1_fused_0 * 1024 + threadIdx_y * 256 + threadIdx_x * 8 + 8] = X_shared_dyn_local_1[ax0_ax1_fused_0]
            for ax0_ax1_fused_0_cache in tir.serial(4):
                Y_shared_dyn_local_1[ax0_ax1_fused_0_cache] = Y_2[i2_0_0 * 131072 + ax0_ax1_fused_0_cache * 32768 + threadIdx_y * 8192 + threadIdx_x // 16 * 4096 + blockIdx_y * 1024 + blockIdx_x % 8 * 128 + threadIdx_x % 16 * 8:i2_0_0 * 131072 + ax0_ax1_fused_0_cache * 32768 + threadIdx_y * 8192 + threadIdx_x // 16 * 4096 + blockIdx_y * 1024 + blockIdx_x % 8 * 128 + threadIdx_x % 16 * 8 + 8]
            for ax0_ax1_fused_0 in tir.serial(4):
                Y_shared_dyn_1[ax0_ax1_fused_0 * 1024 + threadIdx_y * 256 + threadIdx_x * 8:ax0_ax1_fused_0 * 1024 + threadIdx_y * 256 + threadIdx_x * 8 + 8] = Y_shared_dyn_local_1[ax0_ax1_fused_0]
            for i2_0_1 in tir.serial(4):
                for ax0_0 in tir.serial(2):
                    tir.evaluate(tir.ptx_ldmatrix(False, 4, ".b16", X_shared_dyn_m16n8k8_matrixA_1, ax0_0 * 8, tir.tvm_access_ptr(tir.type_annotation(dtype="float16"), X_shared_dyn_2.data, threadIdx_y % 2 * 2048 + ax0_0 * 1024 + i2_0_1 * 8, 1024, 1, dtype="handle"), threadIdx_x * 32, dtype="float16"))
                for ax1_0 in tir.serial(2):
                    tir.evaluate(tir.ptx_ldmatrix(True, 4, ".b16", Y_shared_dyn_m16n8k8_matrixB_1, ax1_0 * 8, tir.tvm_access_ptr(tir.type_annotation(dtype="float16"), Y_shared_dyn_1.data, i2_0_1 * 1024 + threadIdx_y // 2 * 64 + ax1_0 * 32, 1024, 1, dtype="handle"), threadIdx_x % 4 * 128 + threadIdx_x // 4 * 8, dtype="float16"))
                for i0_0_4, i1_0_4 in tir.grid(4, 8):
                    cse_var_1_1: tir.int32 = i1_0_4 * 2
                    tir.evaluate(tir.ptx_mma("m16n8k8", "row", "col", "fp16", "fp16", "fp16", X_shared_dyn_m16n8k8_matrixA_1, i0_0_4 * 4, Y_shared_dyn_m16n8k8_matrixB_1, cse_var_1_1, C_m16n8k8_matrixC_3, i0_0_4 * 32 + cse_var_1_1, False, dtype="float16"))
        for ax0_0 in tir.serial(8):
            for ax1_0 in tir.serial(8):
                X_shared_dyn_3[threadIdx_y * 512 + ax1_0 * 64 + threadIdx_x * 2:threadIdx_y * 512 + ax1_0 * 64 + threadIdx_x * 2 + 2] = C_m16n8k8_matrixC_2[ax0_0 * 8 + ax1_0]
            for i1_0_2_i0_0_2_fused_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 in tir.serial(16):
                C_2[blockIdx_x // 8 * 524288 + i1_0_2_i0_0_2_fused_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 % 8 // 4 * 262144 + ax0_0 * 32768 + threadIdx_y % 2 * 16384 + threadIdx_x // 8 * 4096 + blockIdx_y * 1024 + blockIdx_x % 8 * 128 + i1_0_2_i0_0_2_fused_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 // 8 * 64 + i1_0_2_i0_0_2_fused_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 % 4 * 16 + threadIdx_y // 2 * 8 + threadIdx_x % 8] = X_shared_dyn_3[i1_0_2_i0_0_2_fused_cache_ax1_0_cache_ax0_1_cache_ax1_1_cache_fused_0 * 128 + threadIdx_y * 32 + threadIdx_x]
    
