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

void PaddingBuffer(ScheduleState self, const StmtSRef& block_sref, const Array<IntImm>& padding,
                   const PrimExpr& value) {
  const BlockNode* block = block_sref->StmtAs<BlockNode>();
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

  // Change buffer shape
  // TODO: more careful check here over stride or so
  Buffer buffer = store->buffer;
  auto padded_buffer_ptr = make_object<BufferNode>(*(buffer.get()));
  std::vector<PrimExpr> new_shape;
  for (const IntImm& pad : padding) new_shape.push_back(pad);
  padded_buffer_ptr->shape = std::move(new_shape);
  padded_buffer_ptr->name = padded_buffer_ptr->name + "_padded";
  Buffer padded_buffer = Buffer(padded_buffer_ptr);

  // Change buffer store
  PrimExpr predicate = Bool(true);
  for (const auto it : var_pad_to_map) {
    predicate = predicate && (it.first < IntImm(it.first.dtype(), var_extent_map[it.first]));
  }
  BufferStore padded_store =
      BufferStore(padded_buffer, if_then_else(predicate, store->value, value), store->indices);

  // Change block
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

  // Change BlockRealize
  auto padded_realize_ptr =
      make_object<BlockRealizeNode>(*(GetBlockRealize(self, block_sref).get()));
  padded_realize_ptr->block = padded_block;
  Stmt body = BlockRealize(padded_realize_ptr);

  // Change loop extent
  for (int i = loops.size() - 1; i >= 0; i--) {
    auto it = var_pad_to_map.find(block->iter_vars[i]->var);
    auto loop = make_object<ForNode>(*(loops[i]->StmtAs<ForNode>()));
    loop->body = body;
    if (it != var_pad_to_map.end()) {
      loop->extent = IntImm(loop->loop_var.dtype(), it->second);
    }
    body = For(loop);
  }
  self->Replace(loops[0], body, {{GetRef<Block>(block), padded_block}});

  // Replace old buffer with padded buffer
  Map<Block, Block> block_sref_reuse;
  const StmtSRef& padded_block_sref = self->stmt2ref[padded_block.get()];
  StmtSRef scope_sref = GetScopeRoot(self, padded_block_sref, true);
  const BlockNode* scope_block_ptr = TVM_SREF_TO_BLOCK(scope_block_ptr, scope_sref);
  ReplaceBufferMutator buffer_replacer(buffer, padded_buffer, &block_sref_reuse);
  Block new_scope_block = Downcast<Block>(buffer_replacer(GetRef<Block>(scope_block_ptr)));
  self->Replace(scope_sref, new_scope_block, block_sref_reuse);
}

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

void PaddingEinSum(ScheduleState self, const StmtSRef& block_sref, const Array<IntImm>& padding) {
  const BlockNode* block = block_sref->StmtAs<BlockNode>();
  BlockRealize realize = GetBlockRealize(self, block_sref);

  ICHECK(block->iter_vars.size() == padding.size());
  ICHECK(IsTrivialBinding(self, block_sref));
  ICHECK(IsReductionBlock(self, block_sref, GetScopeRoot(self, block_sref, true)));
  Array<StmtSRef> loops = GetLoops(block_sref);
  CheckGetSingleChildBlockRealizeOnSRefTree(self, loops[0]);

  EinSum ein_sum;
  ExtractEinSum(GetRef<Block>(block), &ein_sum);

  std::unordered_map<Var, int64_t, ObjectPtrHash, ObjectPtrEqual> var_extent;
  for (size_t i = 0; i < padding.size(); ++i) {
    const IterVar& block_var = block->iter_vars[i];
    ICHECK(is_zero(block_var->dom->min));
    var_extent[block_var->var] = padding[i]->value;
  }

  // Change block
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

  // Change BlockRealize
  auto padded_realize_ptr =
      make_object<BlockRealizeNode>(*(GetBlockRealize(self, block_sref).get()));
  padded_realize_ptr->block = padded_block;
  Stmt body = BlockRealize(padded_realize_ptr);

  // Change loops
  for (int i = loops.size() - 1; i >= 0; i--) {
    auto loop = make_object<ForNode>(*(loops[i]->StmtAs<ForNode>()));
    loop->body = body;
    ICHECK(is_zero(loop->min));
    const IntImmNode* loop_extent = loop->extent.as<IntImmNode>();
    ICHECK(loop_extent != nullptr);
    ICHECK_LE(loop_extent->value, padding[i]->value);
    loop->extent = IntImm(loop->loop_var.dtype(), padding[i]->value);
    body = For(loop);
  }
  self->Replace(loops[0], body, {{GetRef<Block>(block), padded_block}});

  // Change producer / input buffers
  const StmtSRef& padded_block_sref = self->stmt2ref[padded_block.get()];
  Array<StmtSRef> producers = GetProducers(self, padded_block_sref);
  std::unordered_map<Buffer, StmtSRef, ObjectPtrHash, ObjectPtrEqual> producer_map;
  auto new_shape = [&](const std::vector<Var>& indices) {
    std::vector<IntImm> padding;
    for (const Var& index : indices) {
      padding.push_back(IntImm(index.dtype(), var_extent.at(index)));
    }
    return padding;
  };
  for (const StmtSRef& producer : producers) {
    const BlockNode* producer_block = producer->StmtAs<BlockNode>();
    ICHECK_EQ(producer_block->writes.size(), 1);
    const Buffer& buffer = producer_block->writes[0]->buffer;
    ICHECK(!producer_map.count(buffer));
    producer_map[buffer] = producer;
    PaddingBuffer(self, self->stmt2ref.at(producer_block), new_shape(ein_sum.input_indices[buffer]),
                  buffer->dtype.is_int() ? PrimExpr(IntImm(buffer->dtype, 0))
                                         : PrimExpr(FloatImm(buffer->dtype, 0.0)));
  }

  // Change output buffer
  auto padded_buffer_ptr = make_object<BufferNode>(*(ein_sum.output_buffer.get()));
  std::vector<PrimExpr> padded_shape;
  for (const IntImm& pad : new_shape(ein_sum.output_indices)) padded_shape.push_back(pad);
  padded_buffer_ptr->shape = std::move(padded_shape);
  padded_buffer_ptr->name = padded_buffer_ptr->name + "_padded";
  Buffer padded_buffer = Buffer(padded_buffer_ptr);

  Map<Block, Block> block_sref_reuse;
  ICHECK(padded_block_sref.defined());
  StmtSRef scope_sref = GetScopeRoot(self, padded_block_sref, true);
  const BlockNode* scope_block_ptr = TVM_SREF_TO_BLOCK(scope_block_ptr, scope_sref);
  ReplaceBufferMutator buffer_replacer(ein_sum.output_buffer, padded_buffer, &block_sref_reuse);
  Block new_scope_block = Downcast<Block>(buffer_replacer(GetRef<Block>(scope_block_ptr)));
  self->Replace(scope_sref, new_scope_block, block_sref_reuse);
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