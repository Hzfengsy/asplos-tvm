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

#include "../utils.h"

namespace tvm {
namespace tir {

struct EinSum {
  Buffer output_buffer;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> input_buffer;

  std::vector<Var> output_indices;
  std::unordered_map<Buffer, std::vector<Var>, ObjectPtrHash, ObjectPtrEqual> input_indices;
};

template <typename T>
void ExtractBufferIndices(const T& buffer_access, std::vector<Var>* indices) {
  for (const PrimExpr& index : buffer_access->indices) {
    const VarNode* var = index.as<VarNode>();
    ICHECK(var != nullptr);
    indices->push_back(GetRef<Var>(var));
  }
}

template <typename T>
bool CompareBufferIndices(const T& buffer_access, std::vector<Var>* indices) {
  ICHECK(buffer_access->indices.size() == indices->size());
  for (size_t i = 0; i < buffer_access->indices.size(); ++i) {
    const PrimExpr& index = buffer_access->indices[i];
    const VarNode* var = index.as<VarNode>();
    ICHECK(var != nullptr);
    if (!indices->at(i).same_as(GetRef<Var>(var))) return false;
  }
  return true;
}

class EinSumPatternMatcher : public ExprVisitor {
 public:
  explicit EinSumPatternMatcher(EinSum* ein_sum) : ein_sum_(ein_sum) {}

  void Match(const PrimExpr& update) {
    const AddNode* add = update.as<AddNode>();
    if (add == nullptr) {
      fail_ = true;
      return;
    }
    const BufferLoadNode* a = add->a.as<BufferLoadNode>();
    const BufferLoadNode* b = add->b.as<BufferLoadNode>();
    if (a == nullptr && b != nullptr) {
      std::swap(a, b);
    }
    if (a == nullptr || !a->buffer.same_as(ein_sum_->output_buffer) ||
        !CompareBufferIndices(GetRef<BufferLoad>(a), &ein_sum_->output_indices)) {
      fail_ = true;
      return;
    }
    VisitExpr(add->b);
  }

  void VisitExpr(const PrimExpr& n) final {
    if (n->IsInstance<BufferLoadNode>() || n->IsInstance<MulNode>() || n->IsInstance<CastNode>()) {
      ExprVisitor::VisitExpr(n);
    } else {
      fail_ = true;
      return;
    }
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    auto it = ein_sum_->input_buffer.find(op->buffer);
    if (it != ein_sum_->input_buffer.end()) {
      fail_ = true;
      return;
    }
    std::vector<Var> indices;
    ExtractBufferIndices(GetRef<BufferLoad>(op), &indices);
    ein_sum_->input_indices.insert(std::make_pair(op->buffer, std::move(indices)));
    ein_sum_->input_buffer.insert(op->buffer);
  }

  void VisitExpr_(const CastNode* op) { VisitExpr(op->value); }

  bool Fail() { return fail_; }

 private:
  EinSum* ein_sum_;
  bool fail_{false};
};

void ExtractEinSum(const Block& block, EinSum* ein_sum) {
  ICHECK(block->init != nullptr);

  const BufferStoreNode* update = block->body.as<BufferStoreNode>();
  ICHECK(update != nullptr);
  ExtractBufferIndices(GetRef<BufferStore>(update), &(ein_sum->output_indices));
  ein_sum->output_buffer = update->buffer;

  const BufferStoreNode* init = (block->init.value()).as<BufferStoreNode>();
  ICHECK(init != nullptr);
  ICHECK(CompareBufferIndices(GetRef<BufferStore>(init), &(ein_sum->output_indices)));

  EinSumPatternMatcher matcher(ein_sum);
  matcher.Match(update->value);
  ICHECK(!matcher.Fail());
}

Buffer PaddingBufferProducer(
    ScheduleState self, const StmtSRef& block_sref, const Array<IntImm>& padding,
    const PrimExpr& value,
    std::unordered_map<For, Stmt, ObjectPtrHash, ObjectPtrEqual>* padded_stmt_map) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  ICHECK(IsTrivialBinding(self, block_sref));

  const BufferStoreNode* store = block->body.as<BufferStoreNode>();
  ICHECK(store != nullptr);

  ICHECK(IsCompleteBlock(self, block_sref, GetScopeRoot(self, block_sref, true)));

  Array<StmtSRef> loops = GetLoops(block_sref);
  CheckGetSingleChildBlockRealizeOnSRefTree(self, loops[0]);

  std::unordered_map<Var, int64_t, ObjectPtrHash, ObjectPtrEqual> var_extent_map;
  for (const IterVar& iter : block->iter_vars) {
    ICHECK(is_zero(iter->dom->min));
    const IntImmNode* extent = iter->dom->extent.as<IntImmNode>();
    ICHECK(extent != nullptr);
    var_extent_map[iter->var] = extent->value;
  }

  std::unordered_map<Var, int64_t, ObjectPtrHash, ObjectPtrEqual> var_pad_to_map;
  ICHECK(padding.size() == store->indices.size());
  for (size_t i = 0; i < padding.size(); ++i) {
    const VarNode* var = store->indices[i].as<VarNode>();
    ICHECK(var != nullptr);
    auto it = var_pad_to_map.find(GetRef<Var>(var));
    if (it == var_pad_to_map.end()) {
      int64_t extent = var_extent_map.at(GetRef<Var>(var));
      if (extent > padding[i]->value) {
        LOG(FATAL);
      } else if (extent < padding[i]->value) {
        var_pad_to_map[GetRef<Var>(var)] = padding[i]->value;
      }
    } else {
      ICHECK_EQ(it->second, padding[i]->value);
    }
  }

  // Create padded buffer store with predicate guard
  PrimExpr predicate = Bool(true);
  for (const auto it : var_pad_to_map) {
    predicate = predicate && (it.first < IntImm(it.first.dtype(), var_extent_map[it.first]));
  }
  BufferStore padded_store =
      BufferStore(store->buffer, if_then_else(predicate, store->value, value), store->indices);
  // Create padded producer block
  auto padded_block_ptr = make_object<BlockNode>(*block);
  std::vector<IterVar> enlarged_iters;
  for (const IterVar& iter : block->iter_vars) {
    auto it = var_pad_to_map.find(iter->var);
    if (it != var_pad_to_map.end()) {
      enlarged_iters.push_back(
          IterVar(Range::FromMinExtent(0, IntImm(iter->var->dtype, it->second)), iter->var,
                  iter->iter_type));
    } else {
      enlarged_iters.push_back(iter);
    }
  }
  padded_block_ptr->iter_vars = std::move(enlarged_iters);
  padded_block_ptr->body = padded_store;
  Block padded_block = Block(padded_block_ptr);
  // Create padded producer BlockRealize
  auto padded_realize_ptr =
      make_object<BlockRealizeNode>(*(GetBlockRealize(self, block_sref).get()));
  padded_realize_ptr->block = padded_block;
  Stmt body = BlockRealize(padded_realize_ptr);
  // Create loops above padded producer
  for (int i = loops.size() - 1; i >= 0; i--) {
    auto it = var_pad_to_map.find(block->iter_vars[i]->var);
    const ForNode* loop_ptr = TVM_SREF_TO_FOR(loop_ptr, loops[i]);
    auto loop = make_object<ForNode>(*(loop_ptr));
    loop->body = body;
    if (it != var_pad_to_map.end()) {
      loop->extent = IntImm(loop->loop_var.dtype(), it->second);
    }
    body = For(loop);
  }
  (*padded_stmt_map)[GetRef<For>(loops[0]->StmtAs<ForNode>())] = body;
  // Return the padded buffer
  Buffer buffer = store->buffer;
  auto padded_buffer_ptr = make_object<BufferNode>(*(buffer.get()));
  std::vector<PrimExpr> new_shape;
  for (const IntImm& pad : padding) new_shape.push_back(pad);
  padded_buffer_ptr->shape = std::move(new_shape);
  padded_buffer_ptr->name = padded_buffer_ptr->name + "_padded";
  Buffer padded_buffer = Buffer(padded_buffer_ptr);
  return padded_buffer;
}

class PaddingScopeCreator : public StmtExprMutator {
 public:
  explicit PaddingScopeCreator(
      std::unordered_map<For, Stmt, ObjectPtrHash, ObjectPtrEqual>* padded_stmt_map,
      std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>* padded_buffer_map)
      : padded_stmt_map_(padded_stmt_map), padded_buffer_map_(padded_buffer_map) {}

  Stmt VisitStmt_(const ForNode* op) final {
    auto it = padded_stmt_map_->find(GetRef<For>(op));
    if (it != padded_stmt_map_->end()) {
      set_reuse = false;
      Stmt mutated = VisitStmt(it->second);
      set_reuse = true;
      const ForNode* mutated_for = mutated.as<ForNode>();
      ICHECK(mutated_for != nullptr);
      reuse_map[GetBlock(op)] = GetBlock(mutated_for);
      return GetRef<For>(mutated_for);
    } else {
      return StmtMutator::VisitStmt_(op);
    }
  }

  Stmt VisitStmt_(const BlockNode* op) final {
    Array<BufferRegion> reads = Mutate(op->reads);
    Array<BufferRegion> writes = Mutate(op->writes);
    Array<MatchBufferRegion> match_buffers = Mutate(op->match_buffers);
    Array<Buffer> alloc_buffers = Mutate(op->alloc_buffers);
    Optional<Stmt> init = NullOpt;
    if (op->init.defined()) {
      init = VisitStmt(op->init.value());
    }
    Stmt body = VisitStmt(op->body);
    if (reads.same_as(op->reads) && writes.same_as(op->writes) &&
        match_buffers.same_as(op->match_buffers) && alloc_buffers.same_as(alloc_buffers) &&
        init.same_as(op->init) && body.same_as(op->body)) {
      return GetRef<Block>(op);
    } else {
      auto n = make_object<BlockNode>(*op);
      n->reads = std::move(reads);
      n->writes = std::move(writes);
      n->body = std::move(body);
      n->init = std::move(init);
      n->match_buffers = std::move(match_buffers);
      n->alloc_buffers = std::move(alloc_buffers);
      Block block = Block(n);
      if (set_reuse) {
        reuse_map[GetRef<Block>(op)] = block;
      }
      return block;
    }
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto it = padded_buffer_map_->find(op->buffer);
    if (it == padded_buffer_map_->end()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    PrimExpr value = this->VisitExpr(op->value);
    auto fmutate = [this](const PrimExpr& e) { return this->VisitExpr(e); };
    Array<PrimExpr> indices = MutateArray(op->indices, fmutate);
    return BufferStore(it->second, value, indices);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto it = padded_buffer_map_->find(op->buffer);
    if (it == padded_buffer_map_->end()) {
      return StmtExprMutator::VisitExpr_(op);
    }
    auto fmutate = [this](const PrimExpr& e) { return this->VisitExpr(e); };
    Array<PrimExpr> indices = MutateArray(op->indices, fmutate);
    return BufferLoad(it->second, indices);
  }

  bool set_reuse{true};
  std::unordered_map<Block, Block, ObjectPtrHash, ObjectPtrEqual> reuse_map;

 private:
  Block GetBlock(const ForNode* loop) {
    for (const ForNode* cur = loop; ;) {
      const BlockRealizeNode* realize = cur->body.as<BlockRealizeNode>();
      if (realize != nullptr) return realize->block;
      cur = cur->body.as<ForNode>();
      ICHECK(cur != nullptr);
    }
  }

  Array<BufferRegion> Mutate(const Array<BufferRegion>& arr) {
    auto fmutate = [&](const BufferRegion& buffer_region) {
      auto it = padded_buffer_map_->find(buffer_region->buffer);
      if (it == padded_buffer_map_->end()) {
        return buffer_region;
      } else {
        return BufferRegion(it->second, buffer_region->region);
      }
    };
    return MutateArray(arr, fmutate);
  }

  Array<MatchBufferRegion> Mutate(const Array<MatchBufferRegion>& arr) {
    auto fmutate = [&](const MatchBufferRegion& match_buffer_region) {
      auto it = padded_buffer_map_->find(match_buffer_region->buffer);
      if (it == padded_buffer_map_->end()) {
        return match_buffer_region;
      } else {
        return MatchBufferRegion(it->second, match_buffer_region->source);
      }
    };
    return MutateArray(arr, fmutate);
  }

  Array<Buffer> Mutate(const Array<Buffer>& arr) {
    auto fmutate = [&](const Buffer& buffer) {
      auto it = padded_buffer_map_->find(buffer);
      if (it == padded_buffer_map_->end()) {
        return buffer;
      } else {
        return it->second;
      }
    };
    return MutateArray(arr, fmutate);
  }

  std::unordered_map<For, Stmt, ObjectPtrHash, ObjectPtrEqual>* padded_stmt_map_;
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual>* padded_buffer_map_;
};

void PaddingEinSum(ScheduleState self, const StmtSRef& block_sref, const Array<IntImm>& padding) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  BlockRealize realize = GetBlockRealize(self, block_sref);

  // CHECK
  // Reduction Block & Trivial Binding & Single Line Loop
  const StmtSRef& scope_sref = GetScopeRoot(self, block_sref, true);
  ICHECK(block->iter_vars.size() == padding.size());
  ICHECK(IsTrivialBinding(self, block_sref));
  ICHECK(IsReductionBlock(self, block_sref, scope_sref));
  Array<StmtSRef> loops = GetLoops(block_sref);
  CheckGetSingleChildBlockRealizeOnSRefTree(self, loops[0]);

  EinSum ein_sum;
  ExtractEinSum(GetRef<Block>(block), &ein_sum);
  // map from var to extent after padding
  std::unordered_map<Var, int64_t, ObjectPtrHash, ObjectPtrEqual> padded_extent;
  for (size_t i = 0; i < padding.size(); ++i) {
    const IterVar& block_var = block->iter_vars[i];
    ICHECK(is_zero(block_var->dom->min));
    const IntImmNode* old_extent = block_var->dom->extent.as<IntImmNode>();
    ICHECK(old_extent != nullptr && old_extent->value <= padding[i]->value);
    padded_extent[block_var->var] = padding[i]->value;
  }
  auto new_shape = [&](const std::vector<Var>& indices) {
    std::vector<IntImm> padding;
    for (const Var& index : indices) {
      padding.push_back(IntImm(index.dtype(), padded_extent.at(index)));
    }
    return padding;
  };
  // *****************************************************
  // *                 IR Manipulation                   *
  // *****************************************************
  std::unordered_map<Buffer, Buffer, ObjectPtrHash, ObjectPtrEqual> padded_buffer_map;
  std::unordered_map<For, Stmt, ObjectPtrHash, ObjectPtrEqual> padded_stmt_map;
  // *****************************************************
  // *                 Compute Block                     *
  // *****************************************************
  // Create padded compute Block
  auto padded_block_ptr = make_object<BlockNode>(*block);
  std::vector<IterVar> enlarged_iters;
  for (size_t i = 0; i < padding.size(); ++i) {
    const IterVar& iter = block->iter_vars[i];
    enlarged_iters.push_back(
        IterVar(Range::FromMinExtent(0, IntImm(iter->var->dtype, padding[i]->value)), iter->var,
                iter->iter_type));
  }
  padded_block_ptr->iter_vars = std::move(enlarged_iters);
  Block padded_block = Block(padded_block_ptr);
  // Create padded compute BlockRealize
  auto padded_realize_ptr =
      make_object<BlockRealizeNode>(*(GetBlockRealize(self, block_sref).get()));
  padded_realize_ptr->block = padded_block;
  Stmt body = BlockRealize(padded_realize_ptr);
  // Create padded loops above padded compute Block
  for (int i = loops.size() - 1; i >= 0; i--) {
    const ForNode* loop_ptr = TVM_SREF_TO_FOR(loop_ptr, loops[i]);
    auto loop = make_object<ForNode>(*(loop_ptr));
    loop->body = body;
    ICHECK(is_zero(loop->min));
    const IntImmNode* loop_extent = loop->extent.as<IntImmNode>();
    ICHECK(loop_extent != nullptr && loop_extent->value <= padding[i]->value);
    loop->extent = IntImm(loop->loop_var.dtype(), padding[i]->value);
    body = For(loop);
  }
  padded_stmt_map[GetRef<For>(loops[0]->StmtAs<ForNode>())] = body;
  // Create padded output buffer
  auto padded_buffer_ptr = make_object<BufferNode>(*(ein_sum.output_buffer.get()));
  std::vector<PrimExpr> padded_shape;
  for (const IntImm& pad : new_shape(ein_sum.output_indices)) padded_shape.push_back(pad);
  padded_buffer_ptr->shape = std::move(padded_shape);
  padded_buffer_ptr->name = padded_buffer_ptr->name + "_padded";
  Buffer padded_buffer = Buffer(padded_buffer_ptr);
  padded_buffer_map[ein_sum.output_buffer] = padded_buffer;
  // *****************************************************
  // *                 Producer Blocks                   *
  // *****************************************************
  // Create padded producers and padded input buffers
  Array<StmtSRef> producers = GetProducers(self, block_sref);
  std::unordered_map<Buffer, StmtSRef, ObjectPtrHash, ObjectPtrEqual> buffer_producer_map;
  for (const StmtSRef& producer : producers) {
    const BlockNode* producer_block = TVM_SREF_TO_BLOCK(producer_block, producer);
    // Producer only produces this buffer
    ICHECK_EQ(producer_block->writes.size(), 1);
    const Buffer& buffer = producer_block->writes[0]->buffer;
    // One buffer has only one producer
    ICHECK(!buffer_producer_map.count(buffer));
    buffer_producer_map[buffer] = producer;
    // Get padded producer
    Buffer padded_buffer =
        PaddingBufferProducer(self, producer, new_shape(ein_sum.input_indices[buffer]),
                              buffer->dtype.is_int() ? PrimExpr(IntImm(buffer->dtype, 0))
                                                     : PrimExpr(FloatImm(buffer->dtype, 0.0)),
                              &padded_stmt_map);
    padded_buffer_map[buffer] = padded_buffer;
  }
  // Create the new scope
  const BlockNode* scope_block = TVM_SREF_TO_BLOCK(scope_block, scope_sref);
  PaddingScopeCreator creator(&padded_stmt_map, &padded_buffer_map);
  Stmt new_scope_block = creator(GetRef<Block>(scope_block));
  self->Replace(scope_sref, new_scope_block, creator.reuse_map);
}

/******** Instruction Registration ********/

struct PaddingEinSumTraits : public UnpackedInstTraits<PaddingEinSumTraits> {
  static constexpr const char* kName = "PaddingEinSum";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 1;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, BlockRV block, Array<IntImm> padding) {
    sch->PaddingEinSum(block, padding);
  }

  static String UnpackedAsPython(Array<String> outputs, String block, Array<IntImm> padding) {
    PythonAPICall py("padding_einsum");
    py.Input("block", block);
    py.Input("padding", padding);
    return py.Str();
  }

  template <typename>
  friend struct ::tvm::tir::UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(PaddingEinSumTraits);

}  // namespace tir
}  // namespace tvm