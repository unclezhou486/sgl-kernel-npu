/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

#ifndef COMPRESSOR_IMPL_H
#define COMPRESSOR_IMPL_H

#include <ATen/Tensor.h>
#include <c10/util/Optional.h>

at::Tensor compressor(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate,
                      const at::Tensor &state_cache, const at::Tensor &ape, const at::Tensor &norm_weight,
                      const at::Tensor &rope_sin, const at::Tensor &rope_cos,
                      const c10::optional<at::Tensor> &state_block_table, const c10::optional<at::Tensor> &cu_seqlens,
                      const c10::optional<at::Tensor> &seqused, const c10::optional<at::Tensor> &start_pos,
                      int64_t rope_head_dim, int64_t cmp_ratio, int64_t coff, double norm_eps, int64_t rotary_mode,
                      int64_t cache_mode, int64_t state_cache_stride_dim0);

#endif  // COMPRESSOR_IMPL_H
