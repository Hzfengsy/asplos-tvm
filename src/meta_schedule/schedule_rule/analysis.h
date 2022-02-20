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
#ifndef TVM_META_SCHEDULE_SCHEDULE_RULE_ANALYSIS_H_
#define TVM_META_SCHEDULE_SCHEDULE_RULE_ANALYSIS_H_

#include <unordered_map>

#include "../utils.h"
namespace tvm {
namespace tir {

/*!
 * \brief Get the buffer dimensions for all the read buffers of a block, but marks the reduction
 * buffers' dimensions as -1
 * \param block_sref The block to be processed
 * \return The buffer dimensions for all the read buffers of a block, except for reduction buffers
 * \note The method is not designed for generic analysis and relies on assumptions in the scenario
 * of multi-level tiling, so it's intentionally kept inside this file not in the analysis header
 */
std::vector<int> GetReadBufferNDims(const StmtSRef& block_sref);

Optional<LoopRV> TilingwithTensorIntrin(const tir::Schedule& sch, const tir::BlockRV& block_rv,
                                        const String& intrin_name);

/*!
 * \brief whether the loop's body has the pattern: 2 cache read shared followed by a nested
 * software pipeline
 * \param the loop
 * \return whether the loop's body has the pattern
 */
bool IsCacheReadSharedPattern(const For& loop);

/*!
 * \brief calculate the software pipeline annotations for a loop that doesn't have special patterns
 * \param loop The loop
 * \param stage The result array of software pipeline stage
 * \param order The result array of software pipeline order
 */
void FallbackRule(const For& loop, Array<Integer>* stage, Array<Integer>* order);

}  // namespace tir
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_SCHEDULE_RULE_ANALYSIS_H_
