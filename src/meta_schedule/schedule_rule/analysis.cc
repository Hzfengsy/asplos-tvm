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

#include "analysis.h"
namespace tvm {
namespace tir {

std::vector<int> GetReadBufferNDims(const StmtSRef& block_sref) {
  const BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  const BufferNode* write_buffer = block->writes[0]->buffer.get();
  int n = block->reads.size();
  std::vector<int> results(n, -1);
  for (int i = 0; i < n; ++i) {
    const BufferNode* read_buffer = block->reads[i]->buffer.get();
    if (read_buffer != write_buffer) {
      results[i] = read_buffer->shape.size();
    }
  }
  return results;
}

Optional<LoopRV> TilingwithTensorIntrin(const Schedule& sch, const BlockRV& block_rv,
                                        const String& intrin_name) {
  Optional<tir::TensorizeInfo> opt_tensorize_info = GetTensorizeLoopMapping(
      sch->state(), sch->GetSRef(block_rv), tir::TensorIntrin::Get(intrin_name)->desc);
  if (!opt_tensorize_info) return NullOpt;
  const tir::TensorizeInfoNode* info = opt_tensorize_info.value().get();
  // Construct a mapping from tir loops back to LoopRVs
  Map<tir::StmtSRef, LoopRV> loop2rv;
  {
    Array<LoopRV> loop_rvs = sch->GetLoops(block_rv);
    for (const LoopRV& loop_rv : loop_rvs) {
      loop2rv.Set(sch->GetSRef(loop_rv), loop_rv);
    }
  }
  // Split the loops
  arith::Analyzer analyzer;
  std::unordered_set<const tir::StmtSRefNode*> inner_loops;
  std::vector<LoopRV> reorder_suffix;
  reorder_suffix.resize(info->loop_map.size());
  for (const auto& kv : info->loop_map) {
    // Extract mapping (block_loop => desc_loop)
    const tir::StmtSRef& block_loop_sref = kv.first;
    const tir::ForNode* block_loop = block_loop_sref->StmtAs<tir::ForNode>();
    const tir::ForNode* desc_loop = kv.second.get();
    ICHECK(block_loop != nullptr && desc_loop != nullptr);
    // Extract the loop extent
    PrimExpr block_extent = analyzer.Simplify(block_loop->extent);
    PrimExpr desc_extent = analyzer.Simplify(desc_loop->extent);
    const auto* int_block_extent = block_extent.as<IntImmNode>();
    const auto* int_desc_extent = desc_extent.as<IntImmNode>();
    ICHECK(int_block_extent != nullptr && int_desc_extent != nullptr);
    // Check divisibility
    int64_t total = int_block_extent->value;
    int64_t inner = int_desc_extent->value;
    ICHECK_EQ(total % inner, 0);
    int64_t outer = int_block_extent->value / int_desc_extent->value;
    // Do the split
    Array<LoopRV> split = sch->Split(loop2rv.at(block_loop_sref), {Integer(outer), Integer(inner)});
    ICHECK_EQ(split.size(), 2);
    inner_loops.insert(sch->GetSRef(split[1]).operator->());
    // The inner split will be reordered to the loop domain that is tensorized
    int desc_loop_index = info->desc_loop_indexer.at(GetRef<tir::For>(desc_loop));
    reorder_suffix[desc_loop_index] = split[1];
  }
  // Reorder the loops
  std::vector<LoopRV> reorder_list;
  bool meet = false;
  Array<LoopRV> all_loops = sch->GetLoops(block_rv);
  for (const LoopRV& loop : all_loops) {
    if (inner_loops.count(sch->GetSRef(loop).operator->())) {
      meet = true;
    } else if (meet) {
      reorder_list.push_back(loop);
    }
  }
  reorder_list.insert(reorder_list.end(), reorder_suffix.begin(), reorder_suffix.end());
  sch->Reorder(reorder_list);
  ICHECK(!reorder_suffix.empty());
  return reorder_suffix[0];
}

bool IsCacheReadSharedPattern(const For& loop) {
  Stmt pipeline_body;
  if (const auto* realize = loop->body.as<BlockRealizeNode>()) {
    const auto& block = realize->block;
    pipeline_body = block->body;
  } else {
    pipeline_body = loop->body;
  }
  const SeqStmtNode* pipeline_body_seq = pipeline_body.as<SeqStmtNode>();
  if (pipeline_body_seq->size() != 3) {
    return false;
  }
  for (int i = 0; i < 2; i++) {
    if (const auto* realize = pipeline_body_seq->seq[i].as<BlockRealizeNode>()) {
      if (is_one(Downcast<Integer>(
              realize->block->annotations.Get("auto_copy").value_or(Integer(0))))) {
        if (is_one(Downcast<Integer>(
                realize->block->annotations.Get(tir::attr::local_stage).value_or(Integer(0))))) {
          Buffer src = realize->block->reads[0]->buffer;
          Buffer tgt = realize->block->writes[0]->buffer;
          runtime::StorageScope src_scope = runtime::StorageScope::Create(src.scope());
          runtime::StorageScope tgt_scope = runtime::StorageScope::Create(tgt.scope());
          if (src_scope.rank == runtime::StorageRank::kGlobal &&
              tgt_scope.rank == runtime::StorageRank::kShared) {
            continue;
          }
        }
      }
    }
    return false;
  }
  if (const auto* loop = pipeline_body_seq->seq[2].as<ForNode>()) {
    if (loop->annotations.count(tir::attr::software_pipeline_stage)) {
      return true;
    }
  }
  return false;
}

void FallbackRule(const For& loop, Array<Integer>* stage, Array<Integer>* order) {
  Stmt pipeline_body;
  if (const auto* realize = loop->body.as<BlockRealizeNode>()) {
    const auto& block = realize->block;
    pipeline_body = block->body;
  } else {
    pipeline_body = loop->body;
  }
  const SeqStmtNode* pipeline_body_seq = pipeline_body.as<SeqStmtNode>();
  for (int i = 0, ct = 0; i < static_cast<int>(pipeline_body_seq->size()); i++) {
    if (const auto* realize = pipeline_body_seq->seq[i].as<BlockRealizeNode>()) {
      if (is_one(Downcast<Integer>(
              realize->block->annotations.Get("auto_copy").value_or(Integer(0))))) {
        stage->push_back(0);
        order->push_back(ct++);
        if (is_one(Downcast<Integer>(
                realize->block->annotations.Get(tir::attr::local_stage).value_or(Integer(0))))) {
          stage->push_back(0);
          order->push_back(ct++);
        }
        continue;
      }
    } else if (const auto* loop = pipeline_body_seq->seq[i].as<ForNode>()) {
      if (loop->annotations.count(tir::attr::software_pipeline_stage)) {
        for (int j = 0; j < 3; j++) {
          stage->push_back(1);
          order->push_back(ct++);
        }
        continue;
      }
    }
    stage->push_back(1);
    order->push_back(ct++);
  }
}

}  // namespace tir
}  // namespace tvm

