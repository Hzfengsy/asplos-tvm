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
namespace meta_schedule {

using tir::BlockRV;
using tir::LoopRV;

using BlockPosition = std::tuple<String, String, String>;

class RewriteTensorCoreNode : public PostprocNode {
 public:
  // Inherited from PostprocNode
  void InitializeWithTuneContext(const TuneContext& context) final {}

  // Inherited from PostprocNode
  bool Apply(const tir::Schedule& sch) final;

  void VisitAttrs(tvm::AttrVisitor* v) {}

  static constexpr const char* _type_key = "meta_schedule.RewriteTensorCore";
  TVM_DECLARE_FINAL_OBJECT_INFO(RewriteTensorCoreNode, PostprocNode);
};

void CollectTensorized(const tir::Schedule& sch, const String& func_name,
                       const tir::PrimFuncNode* func, std::vector<BlockPosition>& tasks) {
  // FIXME: remove visit_init_block
  tir::PreOrderVisit(
      func->body,
      [&](const ObjectRef& obj) -> bool {
        if (const auto* block = obj.as<tir::BlockNode>()) {
          tir::StmtSRef block_sref = sch->GetSRef(block);
          if (Optional<String> intrin_name =
                  tir::GetAnn<String>(block_sref, tir::attr::meta_schedule_auto_tensorize)) {
            tasks.push_back(std::make_tuple(block_sref->StmtAs<tir::BlockNode>()->name_hint,
                                            func_name, intrin_name.value()));
          }
        }
        return true;
      },
      /*visit_init_block=*/false);
}

bool RewriteTensorCoreNode::Apply(const tir::Schedule& sch) {
  std::vector<BlockPosition> tasks;
  for (const auto& kv : sch->mod()->functions) {
    GlobalVar g_var = kv.first;
    BaseFunc base_func = kv.second;
    if (const tir::PrimFuncNode* prim_func = base_func.as<tir::PrimFuncNode>()) {
      CollectTensorized(sch, g_var->name_hint, prim_func, tasks);
    }
  }
  for (const BlockPosition& task : tasks) {
    // Retrieve the block rv according to the task noted down before
    BlockRV block_rv = sch->GetBlock(std::get<0>(task), std::get<1>(task));
    String intrin_name = std::get<2>(task);
    sch->Unannotate(block_rv, tir::attr::meta_schedule_auto_tensorize);
    Optional<LoopRV> tiled_loop_rv = TilingwithTensorIntrin(sch, block_rv, intrin_name);
    if (!tiled_loop_rv.defined()) continue;
    sch->Tensorize(tiled_loop_rv.value(), intrin_name);
  }
  return true;
}

Postproc Postproc::RewriteTensorCore() {
  ObjectPtr<RewriteTensorCoreNode> n = make_object<RewriteTensorCoreNode>();
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(RewriteTensorCoreNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocRewriteTensorCore")
    .set_body_typed(Postproc::RewriteTensorCore);

}  // namespace meta_schedule
}  // namespace tvm
