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

/*! \brief Find all the blocks that are not bound */
class BlockFinder : private StmtVisitor {
 public:
  static std::pair<String, String> Find(const ScheduleState& self) {
    BlockFinder finder(self);
    for (const auto& kv : self->mod->functions) {
      GlobalVar g_var = kv.first;
      BaseFunc base_func = kv.second;
      if (const auto* prim_func = base_func.as<PrimFuncNode>()) {
        finder.func_var_name_ = g_var->name_hint;
        finder(Downcast<BlockRealize>(prim_func->body)->block->body);
      }
    }
    return std::make_pair(finder.func_var_name_, finder.block_var_name_);
  }

 private:

  void VisitStmt_(const BlockNode* block) final {
    block_var_name_ = block->name_hint;
  }

  explicit BlockFinder(const ScheduleState& self)
      : self_{self} {}

  /*! \brief The schedule state */
  const ScheduleState& self_;
  /*! \brief The name of the global var */
  String block_var_name_;
  String func_var_name_;
};

}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace meta_schedule {

/*! \brief Add thread binding to unbound blocks */
class InjectKernelCodeNode : public PostprocNode {
 public:
  // Inherited from PostprocNode
  void InitializeWithTuneContext(const TuneContext& context) final {
  }

  // Inherited from PostprocNode
  bool Apply(const tir::Schedule& sch) final;

 public:
    String code_;

  void VisitAttrs(tvm::AttrVisitor* v) {
    // `max_threads_per_block_` is not visited
    // `max_threadblocks_` is not visited
  }

  static constexpr const char* _type_key = "meta_schedule.InjectKernelCode";
  TVM_DECLARE_FINAL_OBJECT_INFO(InjectKernelCodeNode, PostprocNode);
};

bool InjectKernelCodeNode::Apply(const tir::Schedule& sch) {
  using tir::BlockRV;
  using tir::ExprRV;
  using tir::LoopRV;
  using tir::Schedule;
  std::pair<String, String> pair = tir::BlockFinder::Find(sch->state());
  BlockRV block_rv = sch->GetBlock(pair.second, pair.first);
  LoopRV loop_rv = sch->GetLoops(block_rv)[0];
  sch->Annotate(loop_rv, "pragma_import_llvm", code_);
  return true;
}

Postproc Postproc::InjectKernelCode(String string) {
  ObjectPtr<InjectKernelCodeNode> n = make_object<InjectKernelCodeNode>();
  n->code_ = string;
  return Postproc(n);
}

TVM_REGISTER_NODE_TYPE(InjectKernelCodeNode);
TVM_REGISTER_GLOBAL("meta_schedule.PostprocInjectKernelCode")
    .set_body_typed(Postproc::InjectKernelCode);

}  // namespace meta_schedule
}  // namespace tvm
