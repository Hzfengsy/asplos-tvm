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
#include "./multi_level_tiling.h"

#include <tvm/meta_schedule/schedule_rule.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "../utils.h"
#include "analysis.h"

namespace tvm {
namespace meta_schedule {

using tir::BlockRV;
using tir::IterVarType;
using tir::LoopRV;
using tir::Schedule;

// Do nothing; Inherited from ScheduleRuleNode
void MultiLevelTilingNode::InitializeWithTuneContext(const TuneContext& context) {
  if (Optional<Integer> v = context->target.value()->GetAttr<Integer>("max_threads_per_block")) {
    this->max_threads_per_block_ = v.value()->value;
    if (Optional<Integer> v = context->target.value()->GetAttr<Integer>("thread_warp_size")) {
      this->thread_warp_size_ = v.value()->value;
    } else {
      TVM_PY_LOG(INFO, context->logging_func) << "'thread_warp_size' is not defined in the target";
    }
  }
}

// Entry of the mega rule; Inherited from ScheduleRuleNode
Array<Schedule> MultiLevelTilingNode::Apply(const Schedule& sch, const BlockRV& block_rv) {
  if (!NeedsMultiLevelTiling(sch->state(), sch->GetSRef(block_rv))) {
    return {sch};
  }
  sch->Annotate(block_rv, tir::attr::meta_schedule_tiling_structure, structure);

  Array<Schedule> results;
  for (auto&& state : ApplySubRules({State(sch, block_rv)})) {
    results.push_back(std::move(state.sch));
  }
  return results;
}

std::vector<State> MultiLevelTilingNode::ApplySubRules(std::vector<State> states) {
  states = SubRule(std::move(states), [&](State state) { return SeekForTensorCore(state); });
  states = SubRule(std::move(states), [&](State state) { return TileLoopNest(state); });
  states = SubRule(std::move(states), [&](State state) { return AddWriteReuse(state); });
  states = SubRule(std::move(states), [&](State state) { return AddReadReuse(state); });
  return states;
}

std::vector<State> MultiLevelTilingNode::AddWriteReuse(State state) const {
  const ReuseConfig& config = this->reuse_write_;
  if (config.req == ReuseType::kNoReuse) {
    if (state.tensor_core_is_used) state = TensorCoreStore(state);
    return {std::move(state)};
  }
  std::vector<int> levels = config.levels;
  ReuseType req = config.req;
  if (Optional<Array<Integer>> ann = tir::GetAnn<Array<Integer>>(
    state.sch->GetSRef(state.block_rv), "meta_schedule.write_cache_level")) {
    req = ReuseType::kMustReuse;
    levels = std::vector<int>(ann.value().begin(), ann.value().end());
  }
  std::vector<State> results;
  if (req == ReuseType::kMayReuse) {
    // Case 1. If the write cache is already there, we don't need to add another.
    Array<BlockRV> consumer_rvs = state.sch->GetConsumers(state.block_rv);
    if (consumer_rvs.size() == 1 && IsWriteCache(state.sch->GetSRef(consumer_rvs[0]))) {
      std::vector<State> results;
      results.push_back(state);

      for (int level : levels) {
        State new_state = state;
        new_state.sch = state.sch->Copy();
        new_state.sch->Seed(state.sch->ForkSeed());
        if (new_state.tensor_core_is_used) {
          new_state = TensorCoreStore(new_state);
        } else {
          const LoopRV& loop_rv = new_state.tiles[level - 1].back();
          new_state.sch->ReverseComputeAt(consumer_rvs[0], loop_rv, true);
        }
        results.push_back(std::move(new_state));
      }
      return results;
    } else {
      // Case 2. No write cache is added
      State new_state(/*sch=*/state.sch->Copy(), /*block_rv=*/state.block_rv);
      new_state.sch->Seed(state.sch->ForkSeed());
      if (new_state.tensor_core_is_used) new_state = TensorCoreStore(new_state);
      results.emplace_back(std::move(new_state));
    }
  }

  // Case 3. Add one write cache
  BlockRV write_cache = state.sch->CacheWrite(/*block_rv=*/state.block_rv, /*read_buffer_index=*/0,
                                              /*storage_scope=*/config.scope);
  for (int level : config.levels) {
    State new_state = state;
    new_state.sch = state.sch->Copy();
    new_state.sch->Seed(state.sch->ForkSeed());
    ICHECK(!new_state.tensor_core_is_used) << "not supported"; // FIXME
    const LoopRV& loop_rv = new_state.tiles[level - 1].back();
    new_state.sch->ReverseComputeAt(write_cache, loop_rv, true);
    results.push_back(std::move(new_state));
  }
  return results;
}

std::vector<State> MultiLevelTilingNode::TileLoopNest(State state) const {
  Schedule& sch = state.sch;
  const BlockRV& block_rv = state.block_rv;
  // Step 1. Assuming trivial binding, pair the loops and their iter-var-types
  Array<LoopRV> loops = sch->GetLoops(block_rv);
  std::vector<IterVarType> iter_types = GetBlockVarTypes(sch->GetSRef(state.block_rv));
  ICHECK_EQ(loops.size(), iter_types.size());
  // Step 2. For each loop axis, tile it
  int64_t spatial_loop_product = 1;
  std::vector<Array<LoopRV>> tiles(s_indices_.size() + r_indices_.size());
  for (int i = 0, n = loops.size(); i < n; ++i) {
    LoopRV loop = loops[i];
    const std::vector<int>* idx = nullptr;
    if (iter_types[i] == IterVarType::kDataPar) {
      idx = &s_indices_;
      if (spatial_loop_product != -1) {
        if (const int64_t* extent = tir::GetLoopIntExtent(sch->Get(loop).get())) {
          spatial_loop_product *= *extent;
        } else {
          spatial_loop_product = -1;
        }
      }
    } else if (iter_types[i] == IterVarType::kCommReduce) {
      idx = &r_indices_;
    } else {
      continue;
    }
    // Do the split
    int n_tiles = idx->size();
    Array<tir::ExprRV> factors = sch->SamplePerfectTile(
        /*loop=*/loop,
        /*n=*/n_tiles,
        /*max_innermost_factor=*/max_innermost_factor);
    Array<tir::LoopRV> splits = sch->Split(/*loop=*/loop,
                                           /*factors=*/{factors.begin(), factors.end()});
    // Put every tile to its slot
    for (int j = 0; j < n_tiles; ++j) {
      tiles[idx->at(j)].push_back(splits[j]);
    }
  }
  // Step 3. Reorder to organize the tiles
  sch->Reorder(support::ConcatArrayList<LoopRV>(tiles.begin(), tiles.end()));
  // Step 4. Bind the tiles to threads
  int n_binds = std::min(tile_binds.size(), tiles.size());
  for (int i = 0; i < n_binds; ++i) {
    LoopRV fused = sch->Fuse(tiles[i]);
    sch->Bind(fused, tile_binds[i]);
    tiles[i] = {fused};
  }
  state.tiles = Array<Array<LoopRV>>{tiles.begin(), tiles.end()};
  if (this->thread_warp_size_ != -1) {
    int64_t low_inclusive = 1;
    int64_t high_inclusive = this->max_threads_per_block_;
    if (spatial_loop_product > 2 * this->thread_warp_size_) {
      low_inclusive = this->thread_warp_size_;
    }
    sch->Annotate(block_rv, tir::attr::meta_schedule_thread_extent_low_inclusive,
                  Integer(low_inclusive));
    sch->Annotate(block_rv, tir::attr::meta_schedule_thread_extent_high_inclusive,
                  Integer(high_inclusive));
  }
  return {state};
}

std::vector<State> MultiLevelTilingNode::AddReadReuse(State state) const {
  const ReuseConfig& config = this->reuse_read_;
  if (config.req == ReuseType::kNoReuse) {
    if (state.tensor_core_is_used) state = TensorCoreLoad(state);
    return {std::move(state)};
  }
  ICHECK(config.req != ReuseType::kMayReuse);
  const BlockRV& block_rv = state.block_rv;
  std::vector<State> results;
  results.reserve(config.levels.size());
  for (int level : config.levels) {
    Schedule sch = state.sch->Copy();
    sch->Seed(state.sch->ForkSeed());
    const LoopRV& loop_rv = state.tiles[level - 1].back();
    // Enumerate all buffers that are read but not written
    std::vector<int> read_buffer_ndims = tir::GetReadBufferNDims(sch->GetSRef(block_rv));
    for (int i = 0, n_reads = read_buffer_ndims.size(); i < n_reads; ++i) {
      int buffer_ndim = read_buffer_ndims[i];
      if (buffer_ndim == -1) {
        continue;
      }
      // Do cache_read
      BlockRV cache_read_block;
      if (state.tensor_core_is_used) {
        cache_read_block = sch->ReadAt(loop_rv, block_rv, i, config.scope);
        runtime::StorageScope scope = runtime::StorageScope::Create(config.scope);
        Array<FloatImm> probs(3, FloatImm(DataType::Float(64), 1.0 / 3));
        PrimExpr ann_val = sch->SampleCategorical({4, 8, 16}, probs);
        sch->Annotate(cache_read_block, tir::attr::vector_bytes, ann_val);
        if (scope.rank == runtime::StorageRank::kShared && add_local_stage) {
          sch->Annotate(cache_read_block, tir::attr::local_stage, Integer(1));
        }
        if (scope.rank == runtime::StorageRank::kShared) {
          sch->Annotate(cache_read_block, tir::attr::double_buffer_scope, Integer(0));
        }
      } else {
        cache_read_block = sch->CacheRead(block_rv, i, config.scope);
        // Insert cache_read block to the proper place
        sch->ComputeAt(cache_read_block, loop_rv, true);
        // Fuse the iterators of the cache_read
        Array<LoopRV> buffer_loops = sch->GetLoops(cache_read_block);
        LoopRV fused = sch->Fuse(Array<LoopRV>{buffer_loops.end() - buffer_ndim,  //
                                             buffer_loops.end()});
        // Annotate cooperative fetching
        if (!vector_load_lens.empty()) {
          int n = vector_load_lens.size();
          double prob = 1.0 / n;
          tir::ExprRV vector_load_len =
              sch->SampleCategorical(support::AsArray<int, Integer>(vector_load_lens),
                                     Array<FloatImm>(n, FloatImm(DataType::Float(64), prob)));
          sch->Annotate(cache_read_block, tir::attr::meta_schedule_cooperative_fetch,
                        vector_load_len);
        }
      }
    }
    State new_state = state;
    new_state.sch = sch;
    if (new_state.tensor_core_is_used) new_state = TensorCoreLoad(new_state);
    // add software pipeline annotations
    tir::For loop = new_state.sch->Get(loop_rv);
    Array<Integer> stage;
    Array<Integer> order;
    if (tir::IsCacheReadSharedPattern(loop)) {
      stage = {0, 0, 0, 0, 0, 1, 1};
      order = {0, 3, 1, 4, 5, 2, 6};
    } else {
      tir::FallbackRule(loop, &stage, &order);
    }
    new_state.sch->Annotate(loop_rv, tir::attr::software_pipeline_stage, stage);
    new_state.sch->Annotate(loop_rv, tir::attr::software_pipeline_order, order);
    results.push_back(std::move(new_state));
  }
  return results;
}

// Constructor

ScheduleRule ScheduleRule::MultiLevelTiling(String structure, Optional<Array<String>> tile_binds,
                                            bool use_tensor_core, bool add_local_stage,
                                            Optional<Integer> max_innermost_factor,
                                            Optional<Array<Integer>> vector_load_lens,
                                            Optional<Map<String, ObjectRef>> reuse_read,
                                            Optional<Map<String, ObjectRef>> reuse_write) {
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingNode>(
      structure, tile_binds, max_innermost_factor, vector_load_lens, reuse_read, reuse_write);
  node->use_tensor_core = use_tensor_core;
  if (use_tensor_core) {
    // Check whether corresponding wmma intrinsics are registered
    tir::TensorIntrin::Get("wmma_sync");
    tir::TensorIntrin::Get("wmma_load_a");
    tir::TensorIntrin::Get("wmma_load_b");
    tir::TensorIntrin::Get("wmma_store");
    tir::TensorIntrin::Get("wmma_fill");
  }
  node->add_local_stage = add_local_stage;
  return ScheduleRule(node);
}

Optional<LoopRV> MultiLevelTilingNode::TransformWithTensorIntrin(State& state, const String& intrin_name) const {
  BlockRV block_rv = state.block_rv;
  Optional<tir::LayoutInfo> opt_layout_info =
      GetTensorizeLayoutInfo(state.sch->state(), state.sch->GetSRef(block_rv),
                             tir::TensorIntrin::Get(intrin_name)->desc);
  ICHECK(opt_layout_info.defined());
  if (!opt_layout_info) return NullOpt;
  const tir::LayoutInfoNode* info = opt_layout_info.value().get();

  tir::StmtSRef block_sref = state.sch->GetSRef(state.block_rv);
  const tir::BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  // collect the buffers
  std::unordered_map<tir::Buffer, std::pair<size_t, bool>, ObjectPtrHash, ObjectEqual> buffers;
  for (size_t i = 0; i < block->reads.size(); ++i) {
    buffers[block->reads[i]->buffer] = std::move(std::make_pair(i, true));
  }
  for (size_t i = 0; i < block->writes.size(); ++i) {
    buffers[block->writes[i]->buffer] = std::move(std::make_pair(i, false));
  }
  // Reindex buffers and insert reindex stage
  state.tensor_core_reindex_store = state.sch->ReIndex(block_rv, 0, tir::BufferIndexType::kWrite);
  state.tensor_core_reindex_A = state.sch->ReIndex(block_rv, 0, tir::BufferIndexType::kRead);
  state.tensor_core_reindex_B = state.sch->ReIndex(block_rv, 1, tir::BufferIndexType::kRead);
  block_sref = state.sch->GetSRef(state.block_rv);
  block = TVM_SREF_TO_BLOCK(block, block_sref);

  // Transform the layout of reindex buffers accordingly
  std::unordered_set<tir::Var, ObjectPtrHash, ObjectPtrEqual> unmapped_vars;
  std::unordered_map<tir::Var, tir::Var, ObjectPtrHash, ObjectPtrEqual> representer_map;
  std::unordered_map<tir::Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> tgt_iter_map;
  size_t offset = info->mapping->final_indices.size() - info->rhs_iters.size();
  ICHECK_EQ(info->lhs_iters.size(), info->mapping->initial_indices.size());
  for (size_t i = 0; i < info->lhs_iters.size(); ++i) {
    representer_map[info->lhs_iters[i]->var] = info->mapping->initial_indices[i];
  }
  for (size_t i = 0; i < offset; ++i) {
    const tir::VarNode* var_ptr = info->mapping->final_indices[i].as<tir::VarNode>();
    ICHECK(var_ptr != nullptr);
    unmapped_vars.insert(GetRef<tir::Var>(var_ptr));
  }
  for (size_t i = offset; i < info->mapping->final_indices.size(); ++i) {
    tgt_iter_map[info->rhs_iters[i - offset]->var] = info->mapping->final_indices[i];
  }
  for (const auto& it : buffers) {
    // organize the mappings for buffer layout transformation
    const tir::Buffer& rhs_buffer = info->lhs_buffer_map[it.first];
    std::vector<tir::Var> sub_representers;
    std::vector<PrimExpr> sub_target_iters;
    // Refresh block sref and handler
    block_sref = state.sch->GetSRef(state.block_rv);
    block = TVM_SREF_TO_BLOCK(block, block_sref);
    const tir::BufferRegion& region = it.second.second ? block->reads[it.second.first] : block->writes[it.second.first];
    for (const Range& range : region->region) {
      ICHECK(tir::is_one(range->extent));
      const tir::VarNode* var_ptr = range->min.as<tir::VarNode>();
      ICHECK(var_ptr != nullptr);
      const tir::Var& representer = representer_map[GetRef<tir::Var>(var_ptr)];
      sub_representers.push_back(representer);
      if (unmapped_vars.count(representer)) {
        sub_target_iters.push_back(representer);
      }
    }
    for (size_t i = 0; i < info->rhs_indices_map[rhs_buffer].size(); ++i) {
      const tir::VarNode* var = info->rhs_indices_map[rhs_buffer][i].as<tir::VarNode>();
      ICHECK(var != nullptr);
      sub_target_iters.push_back(tgt_iter_map[GetRef<tir::Var>(var)]);
    }
    state.sch->TransformLayout(state.block_rv, it.second.first,
                               it.second.second ? tir::BufferIndexType::kRead : tir::BufferIndexType::kWrite,
                               tir::IndexMap(sub_representers, sub_target_iters));
  }
  // Transform the layout of current block and reindex blocks
  state.sch->TransformBlockLayout(state.tensor_core_reindex_store.value(), info->mapping);
  state.sch->TransformBlockLayout(state.tensor_core_reindex_A.value(), info->mapping);
  state.sch->TransformBlockLayout(state.tensor_core_reindex_B.value(), info->mapping);
  state.sch->TransformBlockLayout(state.block_rv, info->mapping);

  Array<LoopRV> loops = state.sch->GetLoops(state.block_rv);
  return loops[loops.size() - info->rhs_iters.size()];
}

State MultiLevelTilingNode::TensorCoreLoad(State state) const {
  const Array<LoopRV>& r_tiles = state.tiles[r_indices_[r_indices_.size() - 2]];
  ICHECK(!r_tiles.empty()) << "ValueError: Cannot find the suitable reduction loop in the block";
  state.tensor_core_load_A =
      state.sch->ReadAt(r_tiles.back(), state.block_rv, 0, "wmma.matrix_a");
  state.tensor_core_load_B =
      state.sch->ReadAt(r_tiles.back(), state.block_rv, 1, "wmma.matrix_b");
  state.sch->ComputeInline(state.tensor_core_reindex_A.value());
  state.sch->ComputeInline(state.tensor_core_reindex_B.value());
  tir::For loop = state.sch->Get(r_tiles.back());
  const tir::SeqStmtNode* pipeline_body_seq = loop->body.as<tir::SeqStmtNode>();
  ICHECK(pipeline_body_seq);
  // add software pipeline annotation
  Array<Integer> stage;
  Array<Integer> order;
  tir::FallbackRule(loop, &stage, &order);
  state.sch->Annotate(r_tiles.back(), tir::attr::software_pipeline_stage, stage);
  state.sch->Annotate(r_tiles.back(), tir::attr::software_pipeline_order, order);
  return state;
}

State MultiLevelTilingNode::TensorCoreStore(State state) const {
  // Add the cache read stage for Tensor Core
  int level = r_indices_.front() - 1;
  const LoopRV& loop = state.tiles[level].back();
  state.tensor_core_store = state.sch->WriteAt(loop, state.block_rv, 0, "wmma.accumulator");
  state.sch->ReverseComputeInline(state.tensor_core_reindex_store.value());
  Array<FloatImm> probs(3, FloatImm(DataType::Float(64), 1.0 / 3));
  PrimExpr ann_val = state.sch->SampleCategorical({4, 8, 16}, probs);
  state.sch->Annotate(state.tensor_core_store.value(), tir::attr::vector_bytes, ann_val);
  return state;
}

BlockRV MultiLevelTilingNode::GetRootBlockRV(const Schedule& sch, BlockRV block_rv) const {
  const tir::StmtSRefNode* block = sch->GetSRef(block_rv).get();
  for (; block->parent != nullptr; block = block->parent)
    ;
  for (const auto& kv : sch->mod()->functions) {
    const GlobalVar& gv = kv.first;
    const BaseFunc& base_func = kv.second;
    if (const auto* func = base_func.as<tir::PrimFuncNode>()) {
      const tir::BlockNode* root = func->body.as<tir::BlockRealizeNode>()->block.get();
      if (root == block->StmtAs<tir::BlockNode>()) {
        BlockRV root_rv = sch->GetBlock(root->name_hint, gv->name_hint);
        return root_rv;
      }
    }
  }
  ICHECK(false) << "Ill schedule data structure";
  throw;
}

inline std::vector<State> MultiLevelTilingNode::SeekForTensorCore(State state) const {
  std::vector<State> result;
  // If Tensor Core is not allowed, we skip this subrule
  if (!use_tensor_core) return {state};
  // Do block & buffer layout transform to match Tensor Core wmma sync intrin
  Optional<LoopRV> transformed_loop_rv = TransformWithTensorIntrin(state, "wmma_sync");
  ICHECK(transformed_loop_rv.defined());
  if (!transformed_loop_rv.defined()) return {state};
  // Do tiling to match Tensor Core wmma sync intrin
  BlockRV block_rv = state.block_rv;
  Optional<LoopRV> tiled_loop_rv = TilingwithTensorIntrin(state.sch, block_rv, "wmma_sync");
  ICHECK(tiled_loop_rv.defined());
  // Do blockize
  state.block_rv = state.sch->Blockize(tiled_loop_rv.value());
  // Annotate the block
  state.sch->Annotate(block_rv, tir::attr::meta_schedule_auto_tensorize, String("wmma_sync"));
  state.sch->Annotate(state.block_rv, tir::attr::meta_schedule_auto_tensorize, String("wmma_fill"));
  state.tensor_core_is_used = true;
  // Annotate the root block to notify the following postprocessors
  state.sch->Annotate(GetRootBlockRV(state.sch, state.block_rv),
                      tir::attr::meta_schedule_tensor_core_enabled, String("1"));
  // Annotate the root block to represent the constraint that the extent of threadIdx.x should be 32
  state.sch->Annotate(GetRootBlockRV(state.sch, state.block_rv), tir::attr::warp_execution,
                      Integer(1));
  result.push_back(state);
  return result;
}


TVM_REGISTER_NODE_TYPE(MultiLevelTilingNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleMultiLevelTiling")
    .set_body_typed(ScheduleRule::MultiLevelTiling);

}  // namespace meta_schedule
}  // namespace tvm
