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
#include <tvm/arith/iter_affine_map.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../arith/const_fold.h"
#include "../../arith/pattern_match.h"

namespace tvm {
namespace tir {

IntImm int32(int value) { return IntImm(DataType::Int(32), value); }

struct Attach {
 public:
  enum class AttachType : int {
    kAddition = 0,
    kFloordiv = 1,
    kFloormod = 2,
  };

  // type == kAddition: cur_var = dependent_var * c1 + c2
  // type == kFloordiv: cur_var = floordiv(dependent_var, c1)
  // type == kFloormod: cur_var = floormod(dependent_var, c1)
  // dependent var finally depends on attach_loop
  Var attach_loop, cur_var, dependent_var;
  AttachType type;
  IntImm c1, c2;

  Attach() {}
  Attach(Var attach_loop, Var cur_var, Var dependent_var, AttachType type, IntImm c1, IntImm c2)
      : attach_loop(attach_loop),
        cur_var(cur_var),
        dependent_var(dependent_var),
        type(type),
        c1(c1),
        c2(c2) {}

  PrimExpr Init(const PrimExpr& init) const {
    if (type == AttachType::kAddition) {
      return init * c1 + c2;
    } else if (type == AttachType::kFloordiv) {
      return floordiv(init, c1);
    }
    return floormod(init, c1);
  }

  arith::IntSet Range(const arith::IntSet& range) const {
    if (type == AttachType::kAddition) {
      return arith::EvalSet(dependent_var * c1 + c2, {{dependent_var, range}});
    } else if (type == AttachType::kFloordiv) {
      return arith::EvalSet(floordiv(dependent_var, c1), {{dependent_var, range}});
    }
    return arith::EvalSet(floormod(dependent_var, c1), {{dependent_var, range}});
  }
};

/*!
 * \brief Fuse multiple iterators by summing them with scaling.
 *  result = sum_{i} (vars[i] * scale[i]) + base
 */
class SumFormNode : public PrimExprNode {
 public:
  Array<Var> vars;
  Array<IntImm> scales;
  IntImm base;

  // overrides
  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("vars", &vars);
    v->Visit("scales", &scales);
    v->Visit("base", &base);
  }

  bool SEqualReduce(const SumFormNode* other, SEqualReducer equal) const {
    return equal(vars, other->vars) && equal(scales, other->scales) && equal(base, other->base);
  }

  void SHashReduce(SHashReducer hash_reduce) const {
    hash_reduce(vars);
    hash_reduce(scales);
    hash_reduce(base);
  }

  static constexpr const char* _type_key = "SumForm";
  TVM_DECLARE_FINAL_OBJECT_INFO(SumFormNode, PrimExprNode);
};

class SumForm : public PrimExpr {
 public:
  /*!
   * \brief constructor.
   * \param vars The vars to the sum.
   * \param scale The scales to multiply.
   * \param base The base
   */
  SumForm(Array<Var> vars, Array<IntImm> scales, IntImm base) {
    ICHECK_EQ(vars.size(), scales.size());
    auto n = make_object<SumFormNode>();
    n->dtype = base->dtype;
    n->vars = std::move(vars);
    n->scales = std::move(scales);
    n->base = std::move(base);
    data_ = std::move(n);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(SumForm, PrimExpr, SumFormNode);
};

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<SumFormNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const SumFormNode*>(node.get());
      p->stream << "SumForm(" << op->vars << ", " << op->scales << ", " << op->base << ")";
    });

SumForm AddSumForm(const SumForm& a, const SumForm& b, bool neg = false) {
  int coeff = neg ? -1 : 1;
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> covered;
  std::vector<Var> vars;
  std::vector<IntImm> scales;
  for (size_t i = 0; i < a->vars.size(); ++i) {
    vars.push_back(a->vars[i]);
    covered.insert(a->vars[i]);

    size_t j;
    for (j = 0; j < b->vars.size(); ++j) {
      if (a->vars[i].same_as(b->vars[j])) {
        break;
      }
    }
    scales.push_back(j == b->vars.size()
                         ? a->scales[i]
                         : int32(a->scales[i]->value + coeff * b->scales[j]->value));
  }
  for (size_t i = 0; i < b->vars.size(); ++i) {
    if (!covered.count(b->vars[i])) {
      vars.push_back(b->vars[i]);
      scales.push_back(int32(coeff * b->scales[i]->value));
    }
  }
  return SumForm(std::move(vars), std::move(scales),
                 int32(a->base->value + coeff * b->base->value));
}

SumForm MulSumForm(const SumForm& a, const IntImm& b) {
  std::vector<IntImm> scales;
  for (const IntImm scale : a->scales) {
    scales.push_back(int32(scale->value * b->value));
  }
  return SumForm(a->vars, scales, int32(a->base->value * b->value));
}

class LetVarBindingCanonicalizer : public ExprMutator {
 public:
  explicit LetVarBindingCanonicalizer(
      std::unordered_map<Var, arith::IntSet, ObjectPtrHash, ObjectPtrEqual>* var_range)
      : var_range_(var_range) {}

  bool Canonicalize(const Var& top_var, const PrimExpr& binding) {
    PrimExpr res = this->VisitExpr(Substitute(binding, replace_map));
    if (fail) return false;

    const SumFormNode* ret = res.as<SumFormNode>();
    if (ret == nullptr) return false;
    ICHECK_EQ(ret->vars.size(), 1);
    if (!is_one(ret->scales[0]) || !is_zero(ret->base)) {
      let_var_buffer_map[top_var] =
          decl_buffer({int32(1)}, DataType::Int(32), top_var->name_hint, "local");
      BuildAttachMap(top_var, ret->vars[0], Attach::AttachType::kAddition, ret->scales[0],
                     ret->base);
    } else {
      replace_map[top_var] = ret->vars[0];
    }
    return true;
  }

  std::unordered_map<Var, std::vector<Attach>, ObjectPtrHash, ObjectPtrEqual> inv_attach_map;
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> let_var_buffer_map;
  std::unordered_map<Var, Attach, ObjectPtrHash, ObjectPtrEqual> attach_map;
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> replace_map;

 private:
  void BuildAttachMap(const Var& cur_var, const Var& dependent_var, Attach::AttachType type,
                      IntImm c1, IntImm c2) {
    // Append new attach
    Attach attach;
    auto it = attach_map.find(dependent_var);
    if (it == attach_map.end()) {
      attach = Attach(dependent_var, cur_var, dependent_var, type, c1, c2);
    } else {
      attach = Attach(it->second.attach_loop, cur_var, dependent_var, type, c1, c2);
    }
    inv_attach_map[dependent_var].push_back(attach);
    attach_map[cur_var] = attach;
    // calculate var range
    (*var_range_)[cur_var] = attach.Range((*var_range_)[dependent_var]);
  }

  Optional<Var> SearchExisitingAttach(const Var& dependent_var, Attach::AttachType type,
                                      IntImm c1) {
    auto it = inv_attach_map.find(dependent_var);
    if (it != inv_attach_map.end()) {
      for (const Attach& attach : it->second) {
        if (attach.dependent_var.same_as(dependent_var) && attach.type == type &&
            attach.c1->value == c1->value) {
          return attach.cur_var;
        }
      }
    }
    return NullOpt;
  }

  PrimExpr VisitExprDefault_(const Object* op) final {
    fail = true;
    return SumForm({}, {}, int32(0));
  }

  PrimExpr VisitExpr_(const IntImmNode* op) final { return SumForm({}, {}, int32(op->value)); }

  PrimExpr VisitExpr_(const VarNode* op) final {
    return SumForm({GetRef<Var>(op)}, {int32(1)}, int32(0));
  }

  PrimExpr VisitExpr_(const FloorDivNode* op) final {
    PrimExpr ret_a = this->VisitExpr(op->a);
    PrimExpr ret_b = this->VisitExpr(op->b);
    const SumFormNode* a = ret_a.as<SumFormNode>();
    const SumFormNode* b = ret_b.as<SumFormNode>();
    if (a != nullptr && b != nullptr) {
      ICHECK(a->vars.size() <= 1 && b->vars.size() <= 1);
      if (b->vars.size() == 0) {
        if (a->vars.size() == 0) return SumForm({}, {}, int32(0));
        // define let var for a
        Var inner;
        if (is_one(a->scales[0]) && is_zero(a->base)) {
          // we don't have to introduce an intermediate var
          inner = a->vars[0];
        } else {
          // introduce an intermediate var
          inner = a->vars[0].copy_with_suffix("_lin");
          let_var_buffer_map[inner] =
              decl_buffer({int32(1)}, DataType::Int(32), inner->name_hint, "local");
          BuildAttachMap(inner, a->vars[0], Attach::AttachType::kAddition, a->scales[0], a->base);
        }
        // define let var for div, and a conjugate let var for mod
        // first search for existing vars
        Optional<Var> var_div =
            SearchExisitingAttach(inner, Attach::AttachType::kFloordiv, b->base);
        Optional<Var> var_mod =
            SearchExisitingAttach(inner, Attach::AttachType::kFloormod, b->base);
        // introduce new intermediate vars if doesn't exist now
        if (!var_div.defined()) {
          var_div = inner.copy_with_suffix("_div_" + std::to_string(b->base->value));
          let_var_buffer_map[var_div.value()] =
              decl_buffer({int32(1)}, DataType::Int(32), var_div.value()->name_hint, "local");
          BuildAttachMap(var_div.value(), inner, Attach::AttachType::kFloordiv, b->base, int32(0));
        }
        if (!var_mod.defined()) {
          var_mod = inner.copy_with_suffix("_mod_" + std::to_string(b->base->value));
          let_var_buffer_map[var_mod.value()] =
              decl_buffer({int32(1)}, DataType::Int(32), var_mod.value()->name_hint, "local");
          BuildAttachMap(var_mod.value(), inner, Attach::AttachType::kFloormod, b->base, int32(0));
        }
        return SumForm({var_div.value()}, {int32(1)}, int32(0));
      }
    }
    fail = true;
    return SumForm({}, {}, int32(0));
  }

  PrimExpr VisitExpr_(const FloorModNode* op) final {
    PrimExpr ret_a = this->VisitExpr(op->a);
    PrimExpr ret_b = this->VisitExpr(op->b);
    const SumFormNode* a = ret_a.as<SumFormNode>();
    const SumFormNode* b = ret_b.as<SumFormNode>();
    if (a != nullptr && b != nullptr) {
      ICHECK(a->vars.size() <= 1 && b->vars.size() <= 1);
      if (b->vars.size() == 0) {
        if (a->vars.size() == 0) return SumForm({}, {}, int32(0));
        // define let var for a
        Var inner;
        if (is_one(a->scales[0]) && is_zero(a->base)) {
          // we don't have to introduce an intermediate var
          inner = a->vars[0];
        } else {
          // introduce an intermediate var
          inner = a->vars[0].copy_with_suffix("_lin_");
          let_var_buffer_map[inner] =
              decl_buffer({int32(1)}, DataType::Int(32), inner->name_hint, "local");
          BuildAttachMap(inner, a->vars[0], Attach::AttachType::kAddition, a->scales[0], a->base);
        }
        // define let var for mod
        // first search for existing vars
        Optional<Var> var_mod =
            SearchExisitingAttach(inner, Attach::AttachType::kFloormod, b->base);
        // introduce new intermediate var if doesn't exist now
        if (!var_mod.defined()) {
          var_mod = inner.copy_with_suffix("_mod_" + std::to_string(b->base->value));
          let_var_buffer_map[var_mod.value()] =
              decl_buffer({int32(1)}, DataType::Int(32), var_mod.value()->name_hint, "local");
          BuildAttachMap(var_mod.value(), inner, Attach::AttachType::kFloormod, b->base, int32(0));
        }
        return SumForm({var_mod.value()}, {int32(1)}, int32(0));
      }
    }
    fail = true;
    return SumForm({}, {}, int32(0));
  }

  PrimExpr VisitExpr_(const AddNode* op) final {
    PrimExpr ret_a = this->VisitExpr(op->a);
    PrimExpr ret_b = this->VisitExpr(op->b);
    const SumFormNode* a = ret_a.as<SumFormNode>();
    const SumFormNode* b = ret_b.as<SumFormNode>();
    if (a != nullptr && b != nullptr) {
      ICHECK(a->vars.size() <= 1 && b->vars.size() <= 1);
      SumForm ret = AddSumForm(GetRef<SumForm>(a), GetRef<SumForm>(b));
      if (ret->vars.size() <= 1) {
        return ret;
      }
    }
    fail = true;
    return SumForm({}, {}, int32(0));
  }

  PrimExpr VisitExpr_(const SubNode* op) final {
    PrimExpr ret_a = this->VisitExpr(op->a);
    PrimExpr ret_b = this->VisitExpr(op->b);
    const SumFormNode* a = ret_a.as<SumFormNode>();
    const SumFormNode* b = ret_b.as<SumFormNode>();
    if (a != nullptr && b != nullptr) {
      ICHECK(a->vars.size() <= 1 && b->vars.size() <= 1);
      SumForm ret = AddSumForm(GetRef<SumForm>(a), GetRef<SumForm>(b), true);
      if (ret->vars.size() <= 1) {
        return ret;
      }
    }
    fail = true;
    return SumForm({}, {}, int32(0));
  }

  PrimExpr VisitExpr_(const MulNode* op) final {
    PrimExpr ret_a = this->VisitExpr(op->a);
    PrimExpr ret_b = this->VisitExpr(op->b);
    const SumFormNode* a = ret_a.as<SumFormNode>();
    const SumFormNode* b = ret_b.as<SumFormNode>();
    if (a != nullptr && b != nullptr) {
      ICHECK(a->vars.size() <= 1 && b->vars.size() <= 1);
      if (a->vars.size() == 0) {
        return MulSumForm(GetRef<SumForm>(b), a->base);
      } else if (b->vars.size() == 0) {
        return MulSumForm(GetRef<SumForm>(a), b->base);
      }
    }
    fail = true;
    return SumForm({}, {}, int32(0));
  }

  bool fail{false};

  std::unordered_map<Var, arith::IntSet, ObjectPtrHash, ObjectPtrEqual>* var_range_;
};

class LoadAddressLinearizer : public ExprMutator {
 public:
  explicit LoadAddressLinearizer(
      std::unordered_map<Var, arith::IntSet, ObjectPtrHash, ObjectPtrEqual>* var_range)
      : var_range_(var_range) {}

  bool Linearize(const PrimExpr& addr) {
    result = Downcast<SumForm>(this->VisitExpr(addr));
    return !fail;
  }

  SumForm result;

 private:
  PrimExpr VisitExprDefault_(const Object* op) final {
    fail = true;
    return SumForm({}, {}, int32(0));
  }

  PrimExpr VisitExpr_(const IntImmNode* op) final { return SumForm({}, {}, int32(op->value)); }

  PrimExpr VisitExpr_(const VarNode* op) final {
    return var_range_->count(GetRef<Var>(op)) ? SumForm({GetRef<Var>(op)}, {int32(1)}, int32(0))
                                              : SumForm({}, {}, int32(0));
  }

  PrimExpr VisitExpr_(const FloorDivNode* op) final {
    PrimExpr ret_a = this->VisitExpr(op->a);
    PrimExpr ret_b = this->VisitExpr(op->b);
    const SumFormNode* a = ret_a.as<SumFormNode>();
    const SumFormNode* b = ret_b.as<SumFormNode>();
    if (a != nullptr && b != nullptr) {
      if (a->vars.size() == 0 && b->vars.size() == 0) {
        return SumForm({}, {}, int32(0));
      }
    }
    fail = true;
    return SumForm({}, {}, int32(0));
  }

  PrimExpr VisitExpr_(const FloorModNode* op) final {
    PrimExpr ret_a = this->VisitExpr(op->a);
    PrimExpr ret_b = this->VisitExpr(op->b);
    const SumFormNode* a = ret_a.as<SumFormNode>();
    const SumFormNode* b = ret_b.as<SumFormNode>();
    if (a != nullptr && b != nullptr) {
      if (a->vars.size() == 0 && b->vars.size() == 0) {
        return SumForm({}, {}, int32(0));
      }
    }
    fail = true;
    return SumForm({}, {}, int32(0));
  }

  PrimExpr VisitExpr_(const AddNode* op) final {
    PrimExpr ret_a = this->VisitExpr(op->a);
    PrimExpr ret_b = this->VisitExpr(op->b);
    const SumFormNode* a = ret_a.as<SumFormNode>();
    const SumFormNode* b = ret_b.as<SumFormNode>();
    if (a != nullptr && b != nullptr) {
      return AddSumForm(GetRef<SumForm>(a), GetRef<SumForm>(b));
    }
    fail = true;
    return SumForm({}, {}, int32(0));
  }

  PrimExpr VisitExpr_(const SubNode* op) final {
    PrimExpr ret_a = this->VisitExpr(op->a);
    PrimExpr ret_b = this->VisitExpr(op->b);
    const SumFormNode* a = ret_a.as<SumFormNode>();
    const SumFormNode* b = ret_b.as<SumFormNode>();
    if (a != nullptr && b != nullptr) {
      return AddSumForm(GetRef<SumForm>(a), GetRef<SumForm>(b));
    }
    fail = true;
    return SumForm({}, {}, int32(0));
  }

  PrimExpr VisitExpr_(const MulNode* op) final {
    PrimExpr ret_a = this->VisitExpr(op->a);
    PrimExpr ret_b = this->VisitExpr(op->b);
    const SumFormNode* a = ret_a.as<SumFormNode>();
    const SumFormNode* b = ret_b.as<SumFormNode>();
    if (a != nullptr && b != nullptr) {
      if (a->vars.size() == 0 && !is_zero(a->base)) {
        return MulSumForm(GetRef<SumForm>(b), a->base);
      } else if (b->vars.size() == 0 && !is_zero(b->base)) {
        return MulSumForm(GetRef<SumForm>(a), b->base);
      }
    }
    fail = true;
    return SumForm({}, {}, int32(0));
  }

  bool fail{false};
  std::unordered_map<Var, arith::IntSet, ObjectPtrHash, ObjectPtrEqual>* var_range_;
};

class PredicatePrecompute : public StmtMutator {
 public:
  Stmt VisitStmt_(const LetStmtNode* op) final {
    let_stmt_stack_.push_back(GetRef<LetStmt>(op));
    Stmt result = StmtMutator::VisitStmt_(op);
    let_stmt_stack_.pop_back();
    return result;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // enter
    var_range_[op->loop_var] = arith::IntSet::FromMinExtent(op->min, op->extent);
    For result = Downcast<For>(StmtMutator::VisitStmt_(op));
    // Handle attach map
    auto* new_for = result.CopyOnWrite();
    auto it = attach_map_.find(op->loop_var);
    if (it != attach_map_.end()) {
      // append index update stmts inside the loop
      new_for->body =
          AppendStmt(new_for->body, AttachUpdateStmt(op->loop_var, int32(1), it->second));
    }
    // Handle addr update map
    auto itt = addr_update_map_.find(op->loop_var);
    if (itt != addr_update_map_.end()) {
      std::vector<Stmt> outside{result};
      // append addr update stmts inside and outside the loop
      for (const auto info : itt->second) {
        new_for->body = AppendStmt(
            new_for->body,
            BufferStore(info.first, BufferLoad(info.first, {int32(0)}) + info.second, {int32(0)}));
        outside.push_back(BufferStore(
            info.first, BufferLoad(info.first, {int32(0)}) - info.second * op->extent, {int32(0)}));
      }
      return SeqStmt::Flatten(outside);
    }
    return result;
  }

  Stmt VisitStmt_(const AttrStmtNode* attr) final {
    pre_computed = false;
    AttrStmt result = Downcast<AttrStmt>(StmtMutator::VisitStmt_(attr));
    auto* result_ptr = result.CopyOnWrite();
    // append the pre-computation of predicate buffers
    if (!pre_computed) {
      // append the initialization of addr buffer
      std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> replace;
      for (const auto it : let_var_buffer_map_) {
        replace[it.first] = Load(it.first);
      }
      for (const auto it : var_range_) {
        if (!let_var_buffer_map_.count(it.first)) {
          replace[it.first] = it.second.min();
        }
      }
      for (const auto it : addr_map_) {
        result_ptr->body = AppendStmt(
            BufferStore(it.first, Substitute(it.second.first, replace), {it.second.second->data}),
            result_ptr->body);
      }
      // append the initilziation of index buffers
      for (const auto it : attach_map_) {
        const std::vector<Attach>& attaches = it.second;
        std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> init_map;
        for (const Attach& attach : attaches) {
          PrimExpr value;
          auto itt = init_map.find(attach.dependent_var);
          if (itt == init_map.end()) {
            // depend on loop var
            ICHECK(var_range_.count(attach.dependent_var));
            value = attach.Init(var_range_[attach.dependent_var].min());
          } else {
            value = attach.Init(itt->second);
          }
          init_map[attach.cur_var] = value;
          result_ptr->body = AppendStmt(Store(attach.cur_var, value), result_ptr->body);
        }
      }
      attach_map_.clear();
      for (const auto it : predicate_map_) {
        result_ptr->body = AppendStmt(it.second, result_ptr->body);
      }
      pre_computed = true;
    }
    return result;
  }

  Stmt VisitStmt_(const BufferStoreNode* store) final { return TransformPredicateLoad(store); }

  Stmt VisitStmt_(const AllocateNode* allocate) final {
    Stmt result = StmtMutator::VisitStmt_(allocate);
    // append the definition of predicate buffers, index buffers and addr buffer
    for (const auto it : predicate_map_) {
      result = Allocate(it.first->data, it.first->dtype, it.first->shape, Bool(true), result);
    }
    for (const auto it : let_var_buffer_map_) {
      result = Allocate(it.second->data, it.second->dtype, it.second->shape, Bool(true), result);
    }
    for (const auto it : addr_map_) {
      result = Allocate(
          it.first->data, it.first->dtype, it.first->shape, Bool(true), result,
          {{attr::cached_address, runtime::String(DLDataType2String(it.second.second->dtype))}});
    }
    predicate_map_.clear();
    let_var_buffer_map_.clear();
    addr_map_.clear();
    return result;
  }

 private:
  PrimExpr Load(const Var& var) {
    const auto it = let_var_buffer_map_.find(var);
    if (it == let_var_buffer_map_.end()) {
      return var;
    } else {
      return BufferLoad(it->second, {int32(0)});
    }
  }

  Stmt Store(const Var& var, const PrimExpr& value) {
    const auto it = let_var_buffer_map_.find(var);
    ICHECK(it != let_var_buffer_map_.end());
    return BufferStore(it->second, value, {int32(0)});
  }

 private:
  void AppendVarUpdate(std::vector<Stmt>* body, const Var& var, IntImm delta,
                       const std::vector<Attach>& attaches) {
    body->push_back(Store(var, Load(var) + delta));
    const auto it = addr_update_map_.find(var);
    if (it != addr_update_map_.end()) {
      for (const auto info : it->second) {
        body->push_back(BufferStore(
            info.first, BufferLoad(info.first, {int32(0)}) + delta * info.second, {int32(0)}));
      }
    }
    body->push_back(AttachUpdateStmt(var, delta, attaches));
  }

  Stmt AttachUpdateStmt(Var dependent_var, IntImm inc, const std::vector<Attach>& attaches) {
    std::vector<Stmt> result;
    ICHECK(inc->value != 0);
    for (const Attach& attach : attaches) {
      if (attach.dependent_var.same_as(dependent_var)) {
        if (attach.type == Attach::AttachType::kAddition) {
          IntImm delta = Downcast<IntImm>(inc * attach.c1);
          result.push_back(Store(attach.cur_var, Load(attach.cur_var) + delta));
          result.push_back(AttachUpdateStmt(attach.cur_var, delta, attaches));
        } else if (attach.type == Attach::AttachType::kFloormod) {
          // Search for conjugate div attach
          size_t j;
          for (j = 0; j < attaches.size(); ++j) {
            if (attaches[j].dependent_var.same_as(attach.dependent_var) &&
                attaches[j].type == Attach::AttachType::kFloordiv &&
                attaches[j].c1->value == attach.c1->value) {
              break;
            }
          }
          // x <- x + C
          // floormod(x + C, c1) <- floormod(floormod(x, c1) + floormod(C, c1), c1)
          // 1) = floormod(x, c1) + floormod(C, c1)
          //      floordiv(x + C, c1) <- floordiv(x, c1) + floodiv(C, c1)
          // 2) = floormod(x, c1) + floormod(C, c1) - c1 (if overflow)
          //      floordiv(x + C, c1) <- floordiv(x, c1) + floodiv(C, c1) + 1
          // 3) = floormod(x, c1) + floormod(C, c1) + c1 (if underflow)
          //      floordiv(x + C, c1) <- floordiv(x, c1) + floodiv(C, c1) - 1
          IntImm delta_mod = Downcast<IntImm>(floormod(inc, attach.c1));
          if (delta_mod->value != 0) {
            AppendVarUpdate(&result, attach.cur_var, delta_mod, attaches);
          }
          if (j < attaches.size()) {
            const Attach& attach_div = attaches[j];
            IntImm delta_div = Downcast<IntImm>(floordiv(inc, attach.c1));
            if (delta_div->value != 0) {
              AppendVarUpdate(&result, attach_div.cur_var, delta_div, attaches);
            }
          }
          // Construct the if body
          std::vector<Stmt> if_body;
          int sign = delta_mod->value > 0 ? -1 : 1;
          IntImm delta_mod_if = int32(sign * attach.c1->value);
          AppendVarUpdate(&if_body, attach.cur_var, delta_mod_if, attaches);
          if (j < attaches.size()) {
            const Attach& attach_div = attaches[j];
            IntImm delta_div_if = int32(-sign);
            AppendVarUpdate(&if_body, attach_div.cur_var, delta_div_if, attaches);
          }
          result.push_back(IfThenElse(delta_mod->value > 0
                                          ? (greater_equal(Load(attach.cur_var), attach.c1))
                                          : less(Load(attach.cur_var), int32(0)),
                                      SeqStmt::Flatten(if_body)));
        }
        // Floordiv will be updated together with FloorMod
      }
    }
    return SeqStmt::Flatten(result);
  }

  bool MatchLetVars(LetVarBindingCanonicalizer* canonicalizer) {
    for (const LetStmt& let : let_stmt_stack_) {
      if (!canonicalizer->Canonicalize(let->var, let->value)) {
        return false;
      }
    }
    return true;
  }

  void SplitPredicate(PrimExpr predicate, std::vector<PrimExpr>* sub_predicates) {
    arith::PVar<PrimExpr> sub_predicate, rest;
    for (;;) {
      if ((rest && sub_predicate).Match(predicate)) {
        sub_predicates->push_back(sub_predicate.Eval());
        predicate = rest.Eval();
      } else {
        sub_predicates->push_back(predicate);
        return;
      }
    }
  }

  BufferStore TransformPredicateLoad(const BufferStoreNode* store) {
    // Canonicalize the let var bindings
    LetVarBindingCanonicalizer canonicalizer(&var_range_);
    LoadAddressLinearizer linearizer(&var_range_);
    if (!MatchLetVars(&canonicalizer)) return GetRef<BufferStore>(store);
    local_predicate_map_.clear();
    // Replace the buffer store
    BufferStore replaced_store = Downcast<BufferStore>(Substitute(GetRef<BufferStore>(store), canonicalizer.replace_map));
    // Check the pattern of load address and predicate
    const CallNode* call = replaced_store->value.as<CallNode>();
    if (call != nullptr) {
      const OpNode* op = call->op.as<OpNode>();
      if (op != nullptr && op->name == "tir.if_then_else") {
        ICHECK_EQ(call->args.size(), 3);
        const PrimExpr& predicate = call->args[0];
        const PrimExpr& lhs = call->args[1];
        const PrimExpr& rhs = call->args[2];
        // handle load address
        PrimExpr addr;
        bool lhs_fail{true};
        const BufferLoadNode* load = lhs.as<BufferLoadNode>();
        if (load != nullptr) {
          addr = load->indices[0];
          if (const RampNode* ramp = load->indices[0].as<RampNode>()) {
            addr = ramp->base;
          }
          if (linearizer.Linearize(addr)) {
            lhs_fail = false;
          }
        }
        // handle predicate
        if (!lhs_fail) {
          // split predicates into sub predicates
          std::vector<PrimExpr> sub_predicates, new_sub_predicates;
          SplitPredicate(predicate, &sub_predicates);
          // Note down let var buffer map
          let_var_buffer_map_.insert(canonicalizer.let_var_buffer_map.begin(),
                                     canonicalizer.let_var_buffer_map.end());
          // parameterize sub-predicates
          for (const PrimExpr& sub_predicate : sub_predicates) {
            std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> covered;
            auto collect = [&](const ObjectRef& obj) -> bool {
              if (const tir::VarNode* var = obj.as<tir::VarNode>()) {
                if (!threads_.count(var->name_hint)) {
                  covered.insert(GetRef<tir::Var>(var));
                }
              }
              return true;
            };
            PreOrderVisit(sub_predicate, collect);
            // validate parameterization
            if (covered.size() == 0) {
              // we don't have to pre-compute this sub-predicate
              new_sub_predicates.push_back(sub_predicate);
            } else if (covered.size() == 1) {
              // p(var), max(var) <= 32
              const Var& var = *(covered.begin());
              const IntImmNode* min = var_range_[var].min().as<IntImmNode>();
              const IntImmNode* max = var_range_[var].max().as<IntImmNode>();
              if (min != nullptr && max != nullptr && max->value <= 32) {
                // allocate buffer for this sub-predicate
                const Buffer& buffer =
                    decl_buffer({int32(1)}, DataType::Int(32), "predicate", "local");
                // Create pre-compute loops for this sub-predicate
                Var loop_var = var.copy_with_suffix("_pre");
                Stmt init = BufferStore(buffer, 0, {0});
                For compute = For(
                    loop_var, int32(min->value), int32(max->value - min->value + 1),
                    ForKind::kSerial,
                    BufferStore(buffer,
                                BufferLoad(buffer, {0}) |
                                    ((Substitute(sub_predicate, {{var, loop_var}})) << loop_var),
                                {0}));
                local_predicate_map_[buffer] = SeqStmt({init, compute});
                // rewrite this sub-prediate
                new_sub_predicates.push_back((BufferLoad(buffer, {0}) >> Load(var)) & 1);
                continue;
              }
            } else if (covered.size() == 2) {
              // p(var1, var2), max(max(var1), max(var2)) <= 32, min(max(var1), max(var2)) <= 5
              auto it = covered.begin();
              Var var1 = (*it);
              Var var2 = (*(++it));
              const IntImmNode* min1 = var_range_[var1].min().as<IntImmNode>();
              const IntImmNode* max1 = var_range_[var1].max().as<IntImmNode>();
              const IntImmNode* min2 = var_range_[var2].min().as<IntImmNode>();
              const IntImmNode* max2 = var_range_[var2].max().as<IntImmNode>();
              if (max1 != nullptr && max2 != nullptr) {
                if (max1->value > max2->value) {
                  std::swap(var1, var2);
                  std::swap(min1, min2);
                  std::swap(max1, max2);
                }
                if (max1->value <= 5 && max2->value <= 64) {
                  // allocate buffer for this sub-predicate
                  const Buffer& buffer = decl_buffer({GetRef<IntImm>(max1) + 1}, DataType::Int(32),
                                                     "predicate", "local");
                  local_predicate_map_[buffer] = Evaluate(0);
                  // Create pre-compute loops for this sub-predicate
                  Var loop_var1 = var1.copy_with_suffix("_pre");
                  Var loop_var2 = var2.copy_with_suffix("_pre");
                  For compute = For(
                      loop_var1, int32(min1->value), int32(max1->value - min1->value + 1),
                      ForKind::kSerial,
                      SeqStmt({BufferStore(buffer, 0, {loop_var1}),
                               For(loop_var2, int32(min2->value),
                                   int32(max2->value - min2->value + 1), ForKind::kSerial,
                                   BufferStore(buffer,
                                               BufferLoad(buffer, {loop_var1}) |
                                                   ((Substitute(sub_predicate, {{var1, loop_var1},
                                                                                {var2, loop_var2}}))
                                                    << loop_var2),
                                               {loop_var1}))}));
                  local_predicate_map_[buffer] = compute;
                  // rewrite this sub-prediate
                  new_sub_predicates.push_back((BufferLoad(buffer, {Load(var1)}) >> Load(var2)) &
                                               1);
                  continue;
                }
              }
            }
            // fail case
            return GetRef<BufferStore>(store);
          }
          // Note down attach map
          for (const auto it : canonicalizer.attach_map) {
            attach_map_[it.second.attach_loop].push_back(it.second);
          }
          // Note down predicate buffers
          predicate_map_.insert(local_predicate_map_.begin(), local_predicate_map_.end());
          // Make new predicate
          PrimExpr new_predicate = new_sub_predicates[0];
          for (size_t i = 1; i < new_sub_predicates.size(); ++i) {
            new_predicate = new_sub_predicates[i] & new_predicate;
          }
          // introduce a var for this addr
          const Buffer& buffer = decl_buffer({int32(1)}, DataType::Int(32), "addr", "local");
          addr_map_[buffer] = std::make_pair(addr, load->buffer);
          // note down update info
          for (size_t i = 0; i < linearizer.result->vars.size(); ++i) {
            addr_update_map_[linearizer.result->vars[i]].push_back(
                std::make_pair(buffer, linearizer.result->scales[i]));
          }
          PrimExpr new_lhs;
          if (const RampNode* ramp = load->indices[0].as<RampNode>()) {
            new_lhs = BufferLoad(load->buffer,
                                 {Ramp(BufferLoad(buffer, {int32(0)}), ramp->stride, ramp->lanes)});
          } else {
            new_lhs = BufferLoad(load->buffer, {BufferLoad(buffer, {int32(0)})});
          }
          return BufferStore(
              replaced_store->buffer,
              if_then_else(cast(DataType::Bool(1), new_predicate), new_lhs, rhs, replaced_store->span),
              replaced_store->indices);
        }
      }
    }
    return GetRef<BufferStore>(store);
  }

  Stmt AppendStmt(Stmt body, Stmt stmt) {
    const SeqStmtNode* body_ptr = body.as<SeqStmtNode>();
    if (body_ptr == nullptr) {
      return SeqStmt::Flatten(body, stmt);
    } else {
      return SeqStmt::Flatten(body_ptr->seq, stmt);
    }
  }

  arith::Analyzer analyzer_;
  std::vector<LetStmt> let_stmt_stack_;
  bool pre_computed{false};
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> let_var_buffer_map_;
  std::unordered_map<Var, arith::IntSet, ObjectPtrHash, ObjectPtrEqual> var_range_;

  std::unordered_map<Var, std::vector<Attach>, ObjectPtrHash, ObjectPtrEqual> attach_map_;

  std::unordered_map<Buffer, std::pair<PrimExpr, Buffer>, ObjectPtrHash, ObjectPtrEqual> addr_map_;
  std::unordered_map<Var, std::vector<std::pair<Buffer, IntImm>>, ObjectPtrHash, ObjectPtrEqual>
      addr_update_map_;
  std::unordered_map<Buffer, Stmt, ObjectPtrHash, ObjectPtrEqual> predicate_map_;
  std::unordered_map<Buffer, Stmt, ObjectPtrHash, ObjectPtrEqual> local_predicate_map_;

  std::unordered_set<std::string> threads_{"blockIdx.x",  "blockIdx.y",  "blockIdx.z",
                                           "threadIdx.x", "threadIdx.y", "threadIdx.z"};
};

namespace transform {

Pass OptimizePredicatedLoad(bool enable_predicated_load_optimizer) {
  auto pass_func = [enable_predicated_load_optimizer](PrimFunc f, IRModule m, PassContext ctx) {
    if (enable_predicated_load_optimizer) {
      auto* n = f.CopyOnWrite();
      n->body = PredicatePrecompute()(n->body);
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.OptimizePredicatedLoad", {});
}

// The pass can now be invoked via the pass infrastructure, but we also add a Python binding for it
TVM_REGISTER_GLOBAL("tir.transform.OptimizePredicatedLoad").set_body_typed(OptimizePredicatedLoad);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
