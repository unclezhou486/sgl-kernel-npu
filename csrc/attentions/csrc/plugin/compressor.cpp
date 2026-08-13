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

#include <torch/library.h>

#include "torch_npu/csrc/framework/utils/OpAdapter.h"
#include "torch_npu/csrc/core/npu/NPUFormat.h"
#include "pytorch_npu_helper.h"
#include "compressor.h"

using namespace at;

constexpr std::string_view COMPRESSOR_NAME = "aclnnCompressor";

at::Tensor compressor(const at::Tensor &x, const at::Tensor &wkv, const at::Tensor &wgate,
                      const at::Tensor &state_cache, const at::Tensor &ape, const at::Tensor &norm_weight,
                      const at::Tensor &rope_sin, const at::Tensor &rope_cos,
                      const c10::optional<at::Tensor> &state_block_table, const c10::optional<at::Tensor> &cu_seqlens,
                      const c10::optional<at::Tensor> &seqused, const c10::optional<at::Tensor> &start_pos,
                      int64_t rope_head_dim, int64_t cmp_ratio, int64_t coff, double norm_eps, int64_t rotary_mode,
                      int64_t cache_mode, int64_t state_cache_stride_dim0)
{
    const int64_t headDim = norm_weight.numel();
    at::Tensor cmp_kv;
    if (x.dim() == 3) {  // BSH: (B, Sr, head_dim)
        cmp_kv = at_npu::native::empty_with_format({x.size(0), rope_sin.size(1), headDim}, x.options(),
                                                   at_npu::native::get_npu_format(x));
    } else {  // TH: (Sr, head_dim)
        cmp_kv = at_npu::native::empty_with_format({rope_sin.size(0), headDim}, x.options(),
                                                   at_npu::native::get_npu_format(x));
    }

    EXEC_NPU_CMD<COMPRESSOR_NAME>(x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos, state_block_table,
                                  cu_seqlens, seqused, start_pos, rope_head_dim, cmp_ratio, coff, norm_eps, rotary_mode,
                                  cache_mode, state_cache_stride_dim0, cmp_kv, state_cache);

    return cmp_kv;
}
