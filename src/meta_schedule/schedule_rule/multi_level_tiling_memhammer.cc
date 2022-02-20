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
#include <unordered_map>

#include "../utils.h"
#include "./analysis.h"
#include "./multi_level_tiling.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief The mega rule: multi-level tiling with data reuse for MemHammer
 */
class MultiLevelTilingMemHammerNode : public ScheduleRuleNode {
 public:
  // SubRule 0. detect compute intrin
  inline std::vector<State> DetectTensorCore(State state) const;
  // SubRule 1. tile the loop nest
  inline std::vector<State> TileLoopNest(State state) const;
  // SubRule 2. add read cache
  inline std::vector<State> AddReadReuse(State state) const;
  // SubRule 3. add write cache
  inline std::vector<State> AddWriteReuse(State state) const;

  State TensorCoreLoad(State state) const {
    const Array<LoopRV>& r_tiles = state.tiles[r_indices_[r_indices_.size() - 2]];
    ICHECK(!r_tiles.empty()) << "ValueError: Cannot find the suitable reduction loop in the block";
    state.tensor_core_load_A =
        state.sch->ReadAt(r_tiles.back(), state.block_rv, 1, "wmma.matrix_a");
    state.tensor_core_load_B =
        state.sch->ReadAt(r_tiles.back(), state.block_rv, 2, "wmma.matrix_b");
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

  State TensorCoreStore(State state) const {
    // Add the cache read stage for Tensor Core
    int level = r_indices_.front() - 1;
    const LoopRV& loop = state.tiles[level].back();
    state.tensor_core_store = state.sch->WriteAt(loop, state.block_rv, 0, "wmma.accumulator");
    Array<FloatImm> probs(3, FloatImm(DataType::Float(64), 1.0 / 3));
    PrimExpr ann_val = state.sch->SampleCategorical({4, 8, 16}, probs);
    state.sch->Annotate(state.tensor_core_store.value(), tir::attr::vector_bytes, ann_val);
    return state;
  }

  BlockRV GetRootBlockRV(const Schedule& sch, BlockRV block_rv) const {
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

  // Do nothing; Inherited from ScheduleRuleNode
  void InitializeWithTuneContext(const TuneContext& context) final {}
  // Entry of the mega rule; Inherited from ScheduleRuleNode
  Array<Schedule> Apply(const Schedule& sch, const BlockRV& block_rv) final {
    if (!NeedsMultiLevelTiling(sch->state(), sch->GetSRef(block_rv))) {
      return {sch};
    }
    std::vector<State> states{State(sch, block_rv)};
    states = SubRule(std::move(states), [&](State state) { return DetectTensorCore(state); });
    states = SubRule(std::move(states), [&](State state) { return TileLoopNest(state); });
    states = SubRule(std::move(states), [&](State state) { return AddReadReuse(state); });
    states = SubRule(std::move(states), [&](State state) { return AddWriteReuse(state); });
    Array<Schedule> results;
    for (auto&& state : states) {
      results.push_back(std::move(state.sch));
    }
    return results;
  }

 public:
  /*!
   * \brief The tiling structure. Recommended:
   * - 'SSRSRS' on CPU
   * - 'SSSRRSRS' on GPU
   */
  String structure;
  /*! \brief For each level of tiles, which thread axis it is bound to */
  Array<String> tile_binds;
  /*! \brief Whether to use Tensor Core */
  bool use_tensor_core;
  /*! \brief Whether to add local stage when loading from global to shared */
  bool add_local_stage;
  /*! \brief The maximum size of the innermost factor */
  int max_innermost_factor;
  /*! \brief The length of vector lane in vectorized cooperative fetching */
  int vector_load_max_len;
  /*! \brief Data reuse configuration for reading */
  ReuseConfig reuse_read_;
  /*! \brief Data reuse configuration for writing */
  ReuseConfig reuse_write_;
  /*! \brief The indices of spatial tiles in `structure` */
  std::vector<int> s_indices_;
  /*! \brief The indices of reduction tiles in `structure` */
  std::vector<int> r_indices_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("structure", &structure);
    v->Visit("tile_binds", &tile_binds);
    v->Visit("use_tensor_core", &use_tensor_core);
    v->Visit("add_local_stage", &add_local_stage);
    v->Visit("max_innermost_factor", &max_innermost_factor);
    v->Visit("vector_load_max_len", &vector_load_max_len);
    // `reuse_read_` is not visited
    // `reuse_write_` is not visited
    // `s_indices_` is not visited
    // `r_indices_` is not visited
  }

  static constexpr const char* _type_key = "meta_schedule.MultiLevelTilingMemHammer";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiLevelTilingMemHammerNode, ScheduleRuleNode);
};

inline std::vector<State> MultiLevelTilingMemHammerNode::DetectTensorCore(State state) const {
  std::vector<State> result;
  // If Tensor Core is not allowed, we skip this subrule
  if (!use_tensor_core) return {state};
  // Do tiling to match Tensor Core wmma sync intrin
  BlockRV block_rv = state.block_rv;
  Optional<LoopRV> tiled_loop_rv = TilingwithTensorIntrin(state.sch, block_rv, "wmma_sync");
  if (!tiled_loop_rv.defined()) return {state};
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

inline std::vector<State> MultiLevelTilingMemHammerNode::AddWriteReuse(State state) const {
  const ReuseConfig& config = this->reuse_write_;
  if (config.req == ReuseType::kNoReuse) {
    if (state.tensor_core_is_used) state = TensorCoreStore(state);
    return {std::move(state)};
  }
  // Case 1. If the write cache is already there, we don't need to add another.
  if (config.req == ReuseType::kMayReuse) {
    Array<BlockRV> consumer_rvs = state.sch->GetConsumers(state.block_rv);
    if (consumer_rvs.size() == 1 && IsWriteCache(state.sch->GetSRef(consumer_rvs[0]))) {
      state.write_cache = consumer_rvs[0];
      state.write_cache_is_added = false;
      std::vector<State> results;
      results.push_back(state);

      BlockRV consumer = state.write_cache.value();
      // Enumerate the level of tile to be fused at
      for (int level : config.levels) {
        State new_state = state;
        new_state.sch = state.sch->Copy();
        new_state.sch->Seed(state.sch->ForkSeed());
        if (new_state.tensor_core_is_used) {
          new_state = TensorCoreStore(new_state);
        }
        const LoopRV& loop_rv = new_state.tiles[level - 1].back();
        new_state.sch->ReverseComputeAt(consumer, loop_rv, true);
        results.push_back(std::move(new_state));
      }
      return results;
    }
  }
  std::vector<State> results;
  // Case 2. No write cache is added
  if (config.req == ReuseType::kMayReuse) {
    State new_state(/*sch=*/state.sch->Copy(), /*block_rv=*/state.block_rv,
                    /*write_cache=*/NullOpt,
                    /*write_cache_is_added=*/false);
    new_state.sch->Seed(state.sch->ForkSeed());
    if (new_state.tensor_core_is_used) new_state = TensorCoreStore(new_state);
    results.emplace_back(std::move(new_state));
  }
  // Case 3. Add one write cache
  for (int level : config.levels) {
    State new_state = state;
    new_state.sch = state.sch->Copy();
    new_state.sch->Seed(state.sch->ForkSeed());
    if (new_state.tensor_core_is_used) {
      new_state = TensorCoreStore(new_state);
    }
    const LoopRV& loop_rv = new_state.tiles[level - 1].back();
    BlockRV write_cache =
        new_state.sch->WriteAt(/*loop_rv=*/loop_rv, /*block_rv=*/new_state.block_rv,
                               /*write_buffer_index=*/0,
                               /*storage_scope=*/config.scope);
    new_state.write_cache = write_cache;
    {
      tir::Annotate(new_state.sch->state(), new_state.sch->GetSRef(write_cache),  //
                    tir::attr::meta_schedule_cache_type,                          //
                    Integer(tir::attr::meta_schedule_cache_type_write));
    }
    results.push_back(std::move(new_state));
  }

  return results;
}

inline std::vector<State> MultiLevelTilingMemHammerNode::TileLoopNest(State state) const {
  Schedule& sch = state.sch;
  const BlockRV& block_rv = state.block_rv;
  // Step 1. Assuming trivial binding, pair the loops and their iter-var-types
  Array<LoopRV> loops = sch->GetLoops(block_rv);
  std::vector<IterVarType> iter_types = GetBlockVarTypes(sch->GetSRef(state.block_rv));
  ICHECK_EQ(loops.size(), iter_types.size());
  // Step 2. For each loop axis, tile it
  std::vector<Array<LoopRV>> tiles(s_indices_.size() + r_indices_.size());
  for (int i = 0, n = loops.size(); i < n; ++i) {
    const std::vector<int>* idx = nullptr;
    if (iter_types[i] == IterVarType::kDataPar) {
      idx = &s_indices_;
    } else if (iter_types[i] == IterVarType::kCommReduce) {
      idx = &r_indices_;
    } else {
      continue;
    }
    // Do the split
    int n_tiles = idx->size();
    LoopRV loop = loops[i];
    Array<tir::ExprRV> factors = sch->SamplePerfectTile(
        /*loop=*/loop,
        /*n=*/n_tiles,
        /*max_innermost_factor=*/max_innermost_factor);
    Array<LoopRV> splits = sch->Split(/*loop=*/loop,
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
  return {state};
}

inline std::vector<State> MultiLevelTilingMemHammerNode::AddReadReuse(State state) const {
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
      BlockRV cache_read_block = sch->ReadAt(loop_rv, block_rv, i, config.scope);
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
      {
        tir::Annotate(sch->state(), sch->GetSRef(cache_read_block),  //
                      tir::attr::meta_schedule_cache_type,
                      Integer(tir::attr::meta_schedule_cache_type_read));
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

ScheduleRule ScheduleRule::MultiLevelTilingMemHammer(String structure,
                                                     Optional<Array<String>> tile_binds,
                                                     bool use_tensor_core, bool add_local_stage,
                                                     Optional<Integer> max_innermost_factor,
                                                     Optional<Integer> vector_load_max_len,
                                                     Optional<Map<String, ObjectRef>> reuse_read,
                                                     Optional<Map<String, ObjectRef>> reuse_write) {
  ObjectPtr<MultiLevelTilingMemHammerNode> n = make_object<MultiLevelTilingMemHammerNode>();
  n->structure = structure;
  n->tile_binds = tile_binds.value_or({});
  n->use_tensor_core = use_tensor_core;
  if (use_tensor_core) {
    // Check whether corresponding wmma intrinsics are registered
    tir::TensorIntrin::Get("wmma_sync");
    tir::TensorIntrin::Get("wmma_load_a");
    tir::TensorIntrin::Get("wmma_load_b");
    tir::TensorIntrin::Get("wmma_store");
    tir::TensorIntrin::Get("wmma_fill");
  }
  n->add_local_stage = add_local_stage;
  n->max_innermost_factor = max_innermost_factor.value_or(Integer(-1))->value;
  n->vector_load_max_len = vector_load_max_len.value_or(Integer(-1))->value;
  n->reuse_read_ = reuse_read.defined() ? ReuseConfig(reuse_read.value()) : ReuseConfig();
  n->reuse_write_ = reuse_write.defined() ? ReuseConfig(reuse_write.value()) : ReuseConfig();
  for (int i = 0, len = structure.size(); i < len; ++i) {
    char c = structure.data()[i];
    if (c == 'S') {
      n->s_indices_.push_back(i);
    } else if (c == 'R') {
      n->r_indices_.push_back(i);
    } else {
      LOG(FATAL) << "ValueError: Invalid tiling structure: " << structure;
    }
  }
  return ScheduleRule(n);
}

TVM_REGISTER_NODE_TYPE(MultiLevelTilingMemHammerNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleMultiLevelTilingMemHammer")
    .set_body_typed(ScheduleRule::MultiLevelTilingMemHammer);

}  // namespace meta_schedule
}  // namespace tvm
