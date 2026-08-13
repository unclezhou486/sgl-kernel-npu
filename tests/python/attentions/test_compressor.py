#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Unit tests for the compressor operator.

The compressor computes compressed KV states for DeepSeek-V4 style sparse
attention.  This test exercises the torch entry point:

- ``torch.ops.attentions.compressor`` (this repo's plugin binding), or
- ``torch.ops.custom.compressor`` (external cann-recipes-infer binding) when the
  former is unavailable.  Both wrap the same ``aclnnCompressor``.

Requirements:
  - an Ascend NPU with torch_npu
  - the compressor vendor built and discoverable via ``ASCEND_CUSTOM_OPP_PATH``
  - the attentions plugin ``libPTAExtensionOPS.so`` (build_plugin.sh output)
"""

import os
import unittest
from pathlib import Path

import torch
import torch_npu  # noqa: F401

DEVICE_ID = int(os.environ.get("DEVICE_ID", "0"))
torch_npu.npu.set_device(DEVICE_ID)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _ensure_vendor_path() -> None:
    """Point ASCEND_CUSTOM_OPP_PATH at the built vendor unless already set."""
    if os.environ.get("ASCEND_CUSTOM_OPP_PATH"):
        return
    vendor = _REPO_ROOT / "csrc" / "attentions" / "build" / "vendors" / "aie_ascendc"
    if vendor.exists():
        os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(vendor)


def _ensure_plugin_loaded() -> None:
    if hasattr(torch.ops, "attentions") and hasattr(torch.ops.attentions, "compressor"):
        return
    plugin = (
        _REPO_ROOT / "python" / "attentions" / "attentions" / "plugin"
        / "libPTAExtensionOPS.so"
    )
    if plugin.exists():
        torch.ops.load_library(str(plugin))
        return
    try:
        import attentions  # noqa: F401
    except ImportError:
        pass


def _compressor_op():
    """Return (binding, op), preferring this repo's attentions binding.

    The external ``torch.ops.custom.compressor`` binding is a GE-graph converter
    (custom_ops package) that computes ``state_cache_stride_dim0`` internally, so
    the ``state_cache_stride_dim0`` kwarg is only accepted by the attentions binding.
    """
    _ensure_plugin_loaded()
    if hasattr(torch.ops, "attentions") and hasattr(torch.ops.attentions, "compressor"):
        return "attentions", torch.ops.attentions.compressor
    try:
        import custom_ops  # noqa: F401
    except ImportError:
        pass
    if hasattr(torch.ops, "custom") and hasattr(torch.ops.custom, "compressor"):
        return "custom", torch.ops.custom.compressor
    raise RuntimeError("compressor op not registered under attentions or custom")


def _build_ring_table(start_pos, capacity, block_size, coff, cmp_ratio, batch_size):
    """Build the cache_mode=2 explicit state-location table (B, coff*ratio+capacity)."""
    history_size = coff * cmp_ratio
    banks = torch.arange(batch_size * 2, dtype=torch.int32).view(batch_size, 2)
    dummy_loc = (int(banks.max().item()) + 1) * block_size
    table = torch.full(
        (batch_size, history_size + capacity), dummy_loc, dtype=torch.int32
    )
    for b in range(batch_size):
        for col in range(history_size + capacity):
            position = start_pos[b] - history_size + col
            if position < 0:
                continue
            bank = int(banks[b, (position // block_size) % 2])
            table[b, col] = bank * block_size + position % block_size
    return table


class TestCompressor(unittest.TestCase):
    """Functional smoke tests for the compressor operator (TH layout)."""

    @classmethod
    def setUpClass(cls):
        _ensure_vendor_path()
        cls.binding, cls.op = _compressor_op()

    def _call(self, x, wkv, wgate, state_cache, ape, norm_weight, sin, cos, head_dim,
              coff, cmp_ratio, table, cu, used, starts, cache_mode, stride_dim0):
        kwargs = {
            "rope_sin": sin,
            "rope_cos": cos,
            "rope_head_dim": 64,
            "cmp_ratio": cmp_ratio,
            "state_block_table": table,
            "cu_seqlens": cu,
            "seqused": used,
            "start_pos": starts,
            "coff": coff,
            "norm_eps": 1e-6,
            "rotary_mode": 2,
            "cache_mode": cache_mode,
        }
        if self.binding == "attentions":
            # The aclnn executor path requires the explicit stride attr.
            kwargs["state_cache_stride_dim0"] = stride_dim0
        out = self.op(x, wkv, wgate, state_cache, ape, norm_weight, **kwargs)
        torch.npu.synchronize()
        return out

    def _run_ring(
        self, batch_size, capacity, hidden, head_dim, coff, cmp_ratio, start_pos
    ):
        width = coff * head_dim
        block_size = coff * cmp_ratio + capacity - 1
        block_num = 2 * batch_size + 2
        total = batch_size * capacity
        rope_rows = min(total, total // cmp_ratio + batch_size)
        dtype = torch.bfloat16

        torch.manual_seed(0)
        x = (torch.randn(total, hidden) * 0.02).to(dtype).npu()
        wkv = (torch.randn(width, hidden) * 0.02).to(dtype).npu()
        wgate = (torch.randn(width, hidden) * 0.02).to(dtype).npu()
        state_cache = torch.zeros(
            block_num, block_size, 2 * width, dtype=torch.float32
        ).npu()
        ape = (torch.randn(cmp_ratio, width) * 0.01).float().npu()
        norm_weight = (torch.randn(head_dim) * 0.02 + 1.0).float().npu()
        sin = torch.zeros(rope_rows, 64, dtype=torch.float32).npu()
        cos = torch.ones_like(sin).npu()
        table = _build_ring_table(
            start_pos, capacity, block_size, coff, cmp_ratio, batch_size
        ).npu()
        cu = (torch.arange(batch_size + 1, dtype=torch.int32) * capacity).npu()
        used = torch.full((batch_size,), capacity, dtype=torch.int32).npu()
        starts = torch.tensor(start_pos, dtype=torch.int32).npu()

        out = self._call(
            x, wkv, wgate, state_cache, ape, norm_weight, sin, cos, head_dim,
            coff, cmp_ratio, table, cu, used, starts,
            cache_mode=2, stride_dim0=block_size * (2 * width),
        )
        return out, state_cache, rope_rows, head_dim

    def test_ring_mode_th_c4a(self):
        out, state_cache, rope_rows, head_dim = self._run_ring(
            batch_size=1,
            capacity=8,
            hidden=1024,
            head_dim=512,
            coff=2,
            cmp_ratio=4,
            start_pos=[0],
        )
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.device.type, "npu")
        self.assertEqual(tuple(out.shape), (rope_rows, head_dim))
        self.assertTrue(torch.isfinite(out).all())
        if self.binding == "attentions":
            # Only the aclnn path is guaranteed to write back into the passed tensor.
            self.assertGreater(torch.abs(state_cache).max().item(), 0.0)

    def test_ring_mode_th_c4li(self):
        out, state_cache, rope_rows, head_dim = self._run_ring(
            batch_size=1,
            capacity=8,
            hidden=1024,
            head_dim=128,
            coff=2,
            cmp_ratio=4,
            start_pos=[0],
        )
        self.assertEqual(tuple(out.shape), (rope_rows, head_dim))
        self.assertTrue(torch.isfinite(out).all())
        if self.binding == "attentions":
            # Only the aclnn path is guaranteed to write back into the passed tensor.
            self.assertGreater(torch.abs(state_cache).max().item(), 0.0)

    def test_ring_mode_th_multibatch(self):
        out, state_cache, rope_rows, head_dim = self._run_ring(
            batch_size=2,
            capacity=8,
            hidden=1024,
            head_dim=128,
            coff=2,
            cmp_ratio=4,
            start_pos=[0, 8],
        )
        self.assertEqual(tuple(out.shape), (rope_rows, head_dim))
        self.assertTrue(torch.isfinite(out).all())
        if self.binding == "attentions":
            # Only the aclnn path is guaranteed to write back into the passed tensor.
            self.assertGreater(torch.abs(state_cache).max().item(), 0.0)

    def test_paged_mode_th(self):
        head_dim, coff, cmp_ratio, capacity = 128, 2, 4, 8
        width = coff * head_dim
        total = capacity
        rope_rows = min(total, total // cmp_ratio + 1)
        dtype = torch.bfloat16

        torch.manual_seed(0)
        x = (torch.randn(total, 1024) * 0.02).to(dtype).npu()
        wkv = (torch.randn(width, 1024) * 0.02).to(dtype).npu()
        wgate = (torch.randn(width, 1024) * 0.02).to(dtype).npu()
        state_cache = torch.zeros(8, 1, 2 * width, dtype=torch.float32).npu()
        ape = (torch.randn(cmp_ratio, width) * 0.01).float().npu()
        norm_weight = (torch.randn(head_dim) * 0.02 + 1.0).float().npu()
        sin = torch.zeros(rope_rows, 64, dtype=torch.float32).npu()
        cos = torch.ones_like(sin).npu()
        table = torch.arange(capacity, dtype=torch.int32).unsqueeze(0).npu()
        cu = torch.tensor([0, capacity], dtype=torch.int32).npu()
        used = torch.tensor([capacity], dtype=torch.int32).npu()
        starts = torch.zeros(1, dtype=torch.int32).npu()

        out = self._call(
            x, wkv, wgate, state_cache, ape, norm_weight, sin, cos, head_dim,
            coff, cmp_ratio, table, cu, used, starts,
            cache_mode=1, stride_dim0=2 * width,
        )
        self.assertEqual(tuple(out.shape), (rope_rows, head_dim))
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
