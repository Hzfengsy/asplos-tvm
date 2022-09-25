/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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

/*!
 * \file flatten_buffer.cc
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/arith/analyzer.h>
#include <tvm/tir/index_map.h>

#include "../../support/utils.h"
#include "ir_utils.h"

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
        size_t size = buffer->shape.size();
        ICHECK_GE(size, 2);
        const IntImmNode* dim0 = buffer->shape[size - 2].as<IntImmNode>();
        const IntImmNode* dim1 = buffer->shape[size - 1].as<IntImmNode>();
        ICHECK(dim0 != nullptr && dim1 != nullptr);
        ICHECK(dim0->value % 16 == 0 && dim1->value % 8 == 0);

        std::vector<PrimExpr> new_shape;
        for (size_t i = 0; i < size - 2; ++i) {
          new_shape.push_back(buffer->shape[i]);
        }
        new_shape.insert(new_shape.end(), {
          Integer(dim0->value / 16), Integer(dim1->value / 8), 2, 2
        });

        Buffer new_buffer = decl_buffer(std::move(new_shape), buffer->dtype, buffer->name, "local", buffer->axis_separators);
        this->buffer_map_.insert({buffer, new_buffer});
        this->buffer_var_map_.insert({buffer->data, new_buffer->data});
        return std::move(new_buffer);
      } else if (buffer.scope() == "m16n8k8.matrixA") {
        // m16n8k8.matrixA
        size_t size = buffer->shape.size();
        ICHECK_GE(size, 2);
        const IntImmNode* dim0 = buffer->shape[size - 2].as<IntImmNode>();
        const IntImmNode* dim1 = buffer->shape[size - 1].as<IntImmNode>();
        ICHECK(dim0 != nullptr && dim1 != nullptr);
        ICHECK(dim0->value % 32 == 0 && dim1->value % 8 == 0);
        std::vector<PrimExpr> new_shape;
        for (size_t i = 0; i < size - 2; ++i) {
          new_shape.push_back(buffer->shape[i]);
        }
        new_shape.insert(new_shape.end(), {
          Integer(dim0->value / 32), Integer(dim1->value / 8), 4, 2
        });

        Buffer new_buffer = decl_buffer(std::move(new_shape), buffer->dtype, buffer->name, "local", buffer->axis_separators);
        this->buffer_map_.insert({buffer, new_buffer});
        this->buffer_var_map_.insert({buffer->data, new_buffer->data});
        return std::move(new_buffer);
      } else if (buffer.scope() == "m16n8k8.matrixB") {
        // m16n8k8.matrixB
        size_t size = buffer->shape.size();
        ICHECK_GE(size, 2);
        const IntImmNode* dim0 = buffer->shape[size - 2].as<IntImmNode>();
        const IntImmNode* dim1 = buffer->shape[size - 1].as<IntImmNode>();
        ICHECK(dim0 != nullptr && dim1 != nullptr);
        ICHECK(dim0->value % 8 == 0 && dim1->value % 32 == 0);
        std::vector<PrimExpr> new_shape;
        for (size_t i = 0; i < size - 2; ++i) {
          new_shape.push_back(buffer->shape[i]);
        }
        new_shape.insert(new_shape.end(), {
          Integer(dim0->value / 8), Integer(dim1->value / 32), 4, 2
        });

        Buffer new_buffer = decl_buffer(std::move(new_shape), buffer->dtype, buffer->name, "local", buffer->axis_separators);
        this->buffer_map_.insert({buffer, new_buffer});
        this->buffer_var_map_.insert({buffer->data, new_buffer->data});
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

  PrimExpr VisitExpr_(const VarNode* op) {
    if (buffer_var_map_.count(GetRef<Var>(op))) {
      return buffer_var_map_[GetRef<Var>(op)];
    }
    return GetRef<Var>(op);
  }

 private:
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> buffer_var_map_;
  arith::Analyzer analyzer;
};


PrimExpr BufferArea(const Buffer& buffer) {
  if (buffer->strides.size()) {
    ICHECK(buffer->shape.size() == buffer->strides.size());
    return buffer->strides[0] * buffer->shape[0];
  }
  PrimExpr area = Integer(1);
  for (const PrimExpr& dim : buffer->shape) {
    area = area * dim;
  }
  return area;
}

/*!
 * \brief Transform multi-dimension BufferLoad/BufferStore into device-supported dimension
 */
class BufferFlattener : public StmtExprMutator {
 public:
  static PrimFunc Flatten(PrimFunc func) {
    Map<Var, Buffer> preflattened_buffer_map =
        Merge(func->buffer_map, func->preflattened_buffer_map);
    auto pass = BufferFlattener(func->buffer_map);
    auto writer = func.CopyOnWrite();
    writer->body = pass.VisitStmt(func->body);
    writer->preflattened_buffer_map = preflattened_buffer_map;
    writer->buffer_map = pass.updated_extern_buffer_map_;
    return func;
  }

 private:
  explicit BufferFlattener(const Map<Var, Buffer>& extern_buffer_map) {
    for (const auto& kv : extern_buffer_map) {
      updated_extern_buffer_map_.Set(kv.first, GetFlattenedBuffer(kv.second));
    }
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    // We have convert blocks into opaque blocks in previous passes.
    ICHECK(op->iter_values.empty()) << "Non-opaque blocks are not allowed in FlattenBuffer. Please "
                                       "call pass ConvertBlocksToOpaque before.";
    // Step 1. Visit the body
    Block new_block = Downcast<Block>(this->VisitStmt(op->block));
    PrimExpr predicate = this->VisitExpr(op->predicate);
    // Step 2. Transform the `predicate` to if-then-else
    Stmt body = new_block->body;
    if (!is_one(predicate)) {
      body = IfThenElse(predicate, std::move(body));
    }
    // Step 3. Handle allocations in reverse order
    for (size_t i = new_block->alloc_buffers.size(); i > 0; --i) {
      Buffer buffer = GetFlattenedBuffer(new_block->alloc_buffers[i - 1]);
      body = Allocate(buffer->data, buffer->dtype, buffer->shape, const_true(), std::move(body));
    }
    return body;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // Step 1. Update unit loop info.
    PrimExpr min = this->VisitExpr(op->min);
    PrimExpr extent = this->VisitExpr(op->extent);
    if (is_one(extent) && op->annotations.empty()) {
      // handling unit loop
      unit_loop_vars_[op->loop_var] = min;
    }
    // Step 2. Visit recursively
    Stmt body = this->VisitStmt(op->body);
    // Step 3. Create new For loop accordingly
    if (op->kind == ForKind::kThreadBinding) {
      // Case 1. Thread binding
      ICHECK(op->thread_binding.defined());
      String thread_tag = op->thread_binding.value()->thread_tag;
      body = MakeLaunchThread(min, extent, op->loop_var, thread_tag, body);
    } else if (is_one(extent) && op->annotations.empty()) {
      // Case 2. Unit loop
      return body;
    } else {
      // Case 3. An ordinary loop
      body = For(op->loop_var, std::move(min), std::move(extent), op->kind, std::move(body));
    }
    // Step 4. Handle annotations
    std::set<std::string> ordered_ann_keys;
    for (const auto& annotation : op->annotations) {
      ordered_ann_keys.insert(annotation.first);
    }
    for (auto it = ordered_ann_keys.rbegin(); it != ordered_ann_keys.rend(); ++it) {
      const std::string& ann_key = *it;
      const ObjectRef& ann_value = op->annotations.at(ann_key);
      if (attr::IsPragmaKey(ann_key)) {
        body =
            AttrStmt(op->loop_var, ann_key, ConvertAttrValue(ann_key, ann_value), std::move(body));
      }
    }
    return body;
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    auto it = unit_loop_vars_.find(var);
    if (it == unit_loop_vars_.end()) {
      return std::move(var);
    } else {
      PrimExpr expr = it->second;
      if (expr.dtype() != var.dtype()) {
        expr = tvm::cast(var.dtype(), std::move(expr));
      }
      return expr;
    }
  }

  Buffer GetFlattenedBuffer(Buffer buf) {
    auto it = buffer_remap_.find(buf);
    if (it != buffer_remap_.end()) {
      return it->second;
    }

    auto flattened = buf.GetFlattenedBuffer();

    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (flattened->dtype == DataType::Bool()) {
      auto writer = flattened.CopyOnWrite();
      writer->dtype = DataType::Int(8);
    }

    buffer_remap_[buf] = flattened;
    return flattened;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    bool store_returns_bool = (op->value.dtype() == DataType::Bool());
    store = VisitBufferAccess(store);

    // Handle casts from the value's dtype to the dtype of the
    // backing array.
    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (store_returns_bool) {
      ICHECK_EQ(store->buffer->dtype, DataType::Int(8))
          << "Expected int8 backing array for boolean tensor";
      auto writer = store.CopyOnWrite();
      writer->value = tvm::cast(DataType::Int(8), store->value);
      return store;
    }
    return store;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    bool load_returns_bool = (op->dtype == DataType::Bool());
    BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    load = VisitBufferAccess(load);
    // Handle casts from dtype of the backing array to value's dtype.
    // TODO(Lunderberg): Move the handling of boolean into a
    // dedicated pass.
    if (load_returns_bool) {
      ICHECK_EQ(load->buffer->dtype, DataType::Int(8))
          << "Expected int8 backing array for boolean tensor";
      load.CopyOnWrite()->dtype = DataType::Int(8);
      return tvm::cast(DataType::Bool(), load);
    } else {
      return std::move(load);
    }
  }

  template <typename Node>
  Node VisitBufferAccess(Node node) {
    ICHECK(node->buffer.defined());
    auto flattened_indices = node->buffer->ElemOffset(node->indices);
    Buffer flattened_buffer = GetFlattenedBuffer(node->buffer);

    auto writer = node.CopyOnWrite();
    writer->buffer = flattened_buffer;
    writer->indices = flattened_indices;
    return node;
  }

  static Stmt MakeLaunchThread(PrimExpr min, PrimExpr extent, Var var, String thread_tag,
                               Stmt body) {
    IterVar iter_var(/*dom=*/Range::FromMinExtent(min, extent),
                     /*var=*/std::move(var),
                     /*iter_type=*/IterVarType::kThreadIndex,
                     /*thread_tag=*/thread_tag);
    String attr_key = (thread_tag == "vthread" || thread_tag == "vthread.x" ||
                       thread_tag == "vthread.y" || thread_tag == "vthread.z")
                          ? attr::virtual_thread
                          : attr::thread_extent;
    return AttrStmt(/*node=*/std::move(iter_var),
                    /*attr_key=*/std::move(attr_key),
                    /*value=*/std::move(extent),
                    /*body=*/std::move(body));
  }

  /*! \brief Convert attr value from annotation map into PrimExpr. */
  PrimExpr ConvertAttrValue(const String& key, const ObjectRef& obj) {
    if (!obj.defined()) {
      return PrimExpr();
    } else if (const PrimExprNode* expr = obj.as<PrimExprNode>()) {
      return GetRef<PrimExpr>(expr);
    } else if (const StringObj* str = obj.as<StringObj>()) {
      return std::move(StringImm(str->data));
    } else {
      LOG(FATAL) << "Illegal attribute of key " << key << ", value type " << obj->GetTypeKey()
                 << " not supported";
      return PrimExpr();
    }
  }

  /*! \brief Record the loop_var and loop start value of unit loops, whose extent is one. */
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> unit_loop_vars_;

  /*! \brief Map of buffers being remapped. */
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_remap_;

  /*! \brief The updated external buffer map. */
  Map<Var, Buffer> updated_extern_buffer_map_;
};

PrimFunc FlattenBuffer(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    return BufferFlattener::Flatten(f);
  } else {
    return f;
  }
}

namespace transform {

Pass FlattenBuffer() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = OperandBufferLayoutTransformer()(std::move(n->body));
    return FlattenBuffer(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.FlattenBuffer", {});
}

TVM_REGISTER_GLOBAL("tir.transform.FlattenBuffer").set_body_typed(FlattenBuffer);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
