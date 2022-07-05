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

#include "../../tir/schedule/transform.h"
#include "../utils.h"
#include "multi_level_tiling.h"

#include "analysis.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Tile a subset of loops in the block according to the given tensor intrinsic, and annotate
 * the tiled block for tensorization by postproc rewrite.
 */
tir::BlockRV TileForIntrin(tir::Schedule sch, tir::BlockRV block, const std::string& intrin_name,
                           Array<BlockRV>* reindex_block_rvs) {
  Optional<LoopRV> transformed_loop_rv = TransformWithTensorIntrin(sch, block, intrin_name, reindex_block_rvs);
  if(!transformed_loop_rv.defined()){
    return block;
  }
  Optional<tir::LoopRV> tiled_loop_rv = tir::TilingwithTensorIntrin(sch, block, intrin_name);
  if (!tiled_loop_rv.defined()) {
    reindex_block_rvs->clear();
    return block;
  }
  tir::BlockRV outer_block_rv = sch->Blockize(tiled_loop_rv.value());
  tir::Block outer_block = sch->Get(outer_block_rv);
  String inner_block_name = Downcast<String>(outer_block->annotations.Get("inner_block").value());
  BlockRV inner_block_rv = sch->GetBlock(inner_block_name);
  tir::Var vi("i");
  tir::Var vj("j");
  tir::Var vk("k");
  Array<tir::Var> before_indices{vi, vj, vk};
  Array<PrimExpr> after_indices{floordiv(vk, 4), vi, vj, floormod(vk, 4)};
  sch->TransformBlockLayout(inner_block_rv, tir::IndexMap(before_indices, after_indices));
  sch->Annotate(outer_block_rv, tir::attr::meta_schedule_auto_tensorize, String(intrin_name));
  return outer_block_rv;
}

/*!
 * \brief Extension of MultiLevelTiling for auto-tensorizing with a single intrinsic.
 */
class MultiLevelTilingWithIntrinNode : public MultiLevelTilingNode {
 protected:
  // Override ApplySubRules to tile the inner loops according to the given tensor intrinsic, then
  // tile the outerloops.
  virtual std::vector<State> ApplySubRules(std::vector<State> states) {
    states = SubRule(std::move(states), [&](State state) {
      State old_state = state;
      old_state.sch = state.sch->Copy();
      old_state.sch->Seed(state.sch->ForkSeed());
      Array<BlockRV> reindex_block_rvs;
      state.block_rv = TileForIntrin(state.sch, state.block_rv, intrin_name, &reindex_block_rvs);
      if(!reindex_block_rvs.empty()){
        state.reindex_store = reindex_block_rvs[0];
        state.reindex_A = reindex_block_rvs[1];
        state.reindex_B = reindex_block_rvs[2];
        return std::vector<State>(1, state);
      }
      return std::vector<State>(1, old_state);
    });
    states= MultiLevelTilingNode::ApplySubRules(states);
    states = SubRule(std::move(states), [&](State state) {
      if(state.reindex_store.defined()){
        // state.sch->ReverseComputeInline(state.reindex_store.value());
        state.sch->ComputeInline(state.reindex_A.value());
        state.sch->ComputeInline(state.reindex_B.value());
      }
      return std::vector<State>(1, state);
    });
    return states;
  }

 public:
  /*! \brief The name of a tensor intrinsic. */
  String intrin_name;

  static constexpr const char* _type_key = "meta_schedule.MultiLevelTilingWithIntrin";
  TVM_DECLARE_FINAL_OBJECT_INFO(MultiLevelTilingWithIntrinNode, MultiLevelTilingNode);
};

ScheduleRule ScheduleRule::MultiLevelTilingWithIntrin(
    String intrin_name, String structure, Optional<Array<String>> tile_binds,
    Optional<Integer> max_innermost_factor, Optional<Array<Integer>> vector_load_lens,
    Optional<Map<String, ObjectRef>> reuse_read, Optional<Map<String, ObjectRef>> reuse_write) {
  ICHECK(tir::TensorIntrin::Get(intrin_name).defined())
      << "Provided tensor intrinsic " << intrin_name << " is not registered.";
  auto node = MultiLevelTilingInitCommon<MultiLevelTilingWithIntrinNode>(
      structure, tile_binds, max_innermost_factor, vector_load_lens, reuse_read, reuse_write);
  node->intrin_name = intrin_name;
  return ScheduleRule(node);
}

TVM_REGISTER_NODE_TYPE(MultiLevelTilingWithIntrinNode);
TVM_REGISTER_GLOBAL("meta_schedule.ScheduleRuleMultiLevelTilingWithIntrin")
    .set_body_typed(ScheduleRule::MultiLevelTilingWithIntrin);

}  // namespace meta_schedule
}  // namespace tvm
