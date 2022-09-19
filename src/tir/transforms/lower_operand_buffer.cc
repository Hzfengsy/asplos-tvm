/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {

class OperandBufferLayoutTransformer : public StmtExprMutator {
 public:
  Stmt VisitStmt_(const BlockNode* op) {
    Block block = GetRef<Block>(op);
    auto* n = block.CopyOnWrite();
    auto fmutate = [this](const Buffer& buffer) {
      if (buffer.scope() == "m16n8k8.matrixC") {
        // m16n8k8.matrixC
        // Shape
        ICHECK_EQ(buffer->shape.size(), 2);
        const IntImmNode* dim0 = buffer->shape[0].as<IntImmNode>();
        const IntImmNode* dim1 = buffer->shape[1].as<IntImmNode>();
        ICHECK(dim0 != nullptr && dim1 != nullptr);
        ICHECK(dim0->value % 8 == 0 && dim1->value % 8 == 0);
        std::vector<PrimExpr> new_shape = {
          Integer(dim0->value / 8), Integer(dim1->value / 8), 2
        };

        Buffer new_buffer = decl_buffer(std::move(new_shape), buffer->dtype, buffer->name, "local", buffer->axis_separators);
        this->buffer_map_.insert({buffer, new_buffer});
        return std::move(new_buffer);
      } else if (buffer.scope() == "m16n8k8.matrixA") {
        // m16n8k8.matrixA
        ICHECK_EQ(buffer->shape.size(), 2);
        const IntImmNode* dim0 = buffer->shape[0].as<IntImmNode>();
        const IntImmNode* dim1 = buffer->shape[1].as<IntImmNode>();
        ICHECK(dim0 != nullptr && dim1 != nullptr);
        ICHECK(dim0->value % 32 == 0 && dim1->value % 8 == 0);
        std::vector<PrimExpr> new_shape = {
          Integer(dim0->value / 32), Integer(dim1->value / 8), 4, 2
        };

        Buffer new_buffer = decl_buffer(std::move(new_shape), buffer->dtype, buffer->name, "local", buffer->axis_separators);
        this->buffer_map_.insert({buffer, new_buffer});
        return std::move(new_buffer);
      } else if (buffer.scope() == "m16n8k8.matrixB") {
        // m16n8k8.matrixB
        ICHECK_EQ(buffer->shape.size(), 2);
        const IntImmNode* dim0 = buffer->shape[0].as<IntImmNode>();
        const IntImmNode* dim1 = buffer->shape[1].as<IntImmNode>();
        ICHECK(dim0 != nullptr && dim1 != nullptr);
        ICHECK(dim0->value % 8 == 0 && dim1->value % 32 == 0);
        std::vector<PrimExpr> new_shape = {
          Integer(dim0->value / 8), Integer(dim1->value / 32), 4, 2
        };

        Buffer new_buffer = decl_buffer(std::move(new_shape), buffer->dtype, buffer->name, "local", buffer->axis_separators);
        this->buffer_map_.insert({buffer, new_buffer});
        return std::move(new_buffer);
      }
      return buffer;
    };
    n->alloc_buffers.MutateByApply(fmutate);
    n->body = VisitStmt(n->body);
    return block;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    if (buffer_map_.count(store->buffer)) {
      auto* n = store.CopyOnWrite();
      if (store->buffer.scope() == "m16n8k8.matrixC") {
        const auto* index_map_func = runtime::Registry::Get("tir.index_map_m16n8k8.matrixC");
        ICHECK(index_map_func);
        auto index_map = IndexMap::FromFunc(2, *index_map_func);
        auto new_indices = index_map->MapIndices(store->indices);
        n->buffer = buffer_map_[store->buffer];
        n->indices = std::move(new_indices);
      } else if (store->buffer.scope() == "m16n8k8.matrixA" || store->buffer.scope() == "m16n8k8.matrixB") {
        n->buffer = buffer_map_[store->buffer];
      }
    }
    return std::move(store);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) {
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    if (buffer_map_.count(load->buffer)) {
      auto* n = load.CopyOnWrite();
      if (load->buffer.scope() == "m16n8k8.matrixC") {
        const auto* index_map_func = runtime::Registry::Get("tir.index_map_m16n8k8.matrixC");
        ICHECK(index_map_func);
        auto index_map = IndexMap::FromFunc(2, *index_map_func);
        auto new_indices = index_map->MapIndices(load->indices);
        n->buffer = buffer_map_[load->buffer];
        n->indices = std::move(new_indices);
      } else if (load->buffer.scope() == "m16n8k8.matrixA" || load->buffer.scope() == "m16n8k8.matrixB") {
        n->buffer = buffer_map_[load->buffer];
      }
    }
    return std::move(load);
  }

 private:
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
  arith::Analyzer analyzer;
};

namespace transform {

Pass LowerOperandBuffer() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = OperandBufferLayoutTransformer()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerOperandBuffer", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerOperandBuffer").set_body_typed(LowerOperandBuffer);

}  // namespace transform

}  // namespace tir
}  // namespace tvm