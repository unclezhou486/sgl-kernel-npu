# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
#
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Compressor operator tests (C128A / C4A / C4Li) on NPU.

Covers: op availability, per-variant shape/dtype, paged vs ring cache modes,
empty batch, BF16/FP16, CPU golden comparison and state-cache immutability for
C128A paged mode.
"""

import unittest

import numpy as np
import torch
import torch_npu

import attentions  # sets ASCEND_CUSTOM_OPP_PATH to the built aie_ascendc vendor
import custom_ops  # registers torch.ops.custom.* in SGLang runtime


# ── CPU reference helpers (ported from cann-ops-transformer compressor_golden) ──
def _softmax_columns(z):
    z_max = np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z - z_max)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def _rms_norm(x, weight, eps):
    var = np.mean(np.square(x), axis=-1, keepdims=True)
    x = x * np.reciprocal(np.sqrt(var + eps))
    return weight * x


def _rotary_emb(x, rope_sin, rope_cos, rotary_mode=2):
    """Apply RoPE to x (sc, rope_head_dim); rotary_mode 2 = interleave."""
    sc = x.shape[0]
    rope_head_dim = x.shape[-1]
    rope_sin = rope_sin.reshape(sc, rope_head_dim)
    rope_cos = rope_cos.reshape(sc, rope_head_dim)
    y = np.zeros_like(x)
    group = rope_head_dim // 2
    for s in range(sc):
        for i in range(group):
            if rotary_mode == 2:
                idx = 2 * i
                a = x[s][idx]
                b = x[s][idx + 1]
                y[s][idx] = a * rope_cos[s][idx] - b * rope_sin[s][idx]
                y[s][idx + 1] = a * rope_sin[s][idx + 1] + b * rope_cos[s][idx + 1]
            else:
                a = x[s][i]
                b = x[s][i + group]
                y[s][i] = a * rope_cos[s][i] - b * rope_sin[s][i]
                y[s][i + group] = a * rope_sin[s][i + group] + b * rope_cos[s][i + group]
    return y


def _read_state_page_cache(state, b_idx, start_seq_idx, end_seq_idx, block_table, d_start, d_end,
                           cache_mode=1, batch_start_pos=0, history_size=0):
    block_size = state.shape[1]
    seq_cnt = end_seq_idx - start_seq_idx
    result = np.zeros((seq_cnt, d_end - d_start), dtype=np.float32)
    if cache_mode == 2:
        state_flat = state.reshape(-1, state.shape[-1])
        for offset in range(seq_cnt):
            table_column = history_size + start_seq_idx + offset - batch_start_pos
            state_loc = int(block_table[b_idx, table_column])
            result[offset] = state_flat[state_loc, d_start:d_end]
        return result
    finish_cnt = 0
    while finish_cnt < seq_cnt:
        cur_seq_id = start_seq_idx + finish_cnt
        block_id = int(block_table[b_idx][cur_seq_id // block_size])
        block_start_seq_id = cur_seq_id % block_size
        can_read = min(block_size - block_start_seq_id, seq_cnt - finish_cnt)
        result[finish_cnt:finish_cnt + can_read, :] = state[
            block_id, block_start_seq_id:block_start_seq_id + can_read, d_start:d_end]
        finish_cnt += can_read
    return result


def _write_state_page_cache(state, update_position, sc_new_state, b_idx, start_seq_idx, end_seq_idx,
                            block_table, cache_mode=1, batch_start_pos=0, history_size=0):
    block_size = state.shape[1]
    seq_cnt = end_seq_idx - start_seq_idx
    if cache_mode == 2:
        state_flat = state.reshape(-1, state.shape[-1])
        update_flat = update_position.reshape(-1, update_position.shape[-1])
        for offset in range(seq_cnt):
            table_column = history_size + start_seq_idx + offset - batch_start_pos
            state_loc = int(block_table[b_idx, table_column])
            state_flat[state_loc] = sc_new_state[offset]
            update_flat[state_loc] = True
        return
    finish_cnt = 0
    while finish_cnt < seq_cnt:
        cur_seq_id = start_seq_idx + finish_cnt
        block_id = int(block_table[b_idx][cur_seq_id // block_size])
        block_start_seq_id = cur_seq_id % block_size
        can_write = min(block_size - block_start_seq_id, seq_cnt - finish_cnt)
        if block_id != 0:
            state[block_id, block_start_seq_id:block_start_seq_id + can_write, :] = \
                sc_new_state[finish_cnt:finish_cnt + can_write, :]
            update_position[block_id, block_start_seq_id:block_start_seq_id + can_write, :] = True
        finish_cnt += can_write


def _cpu_compressor_c128a_paged(
    x, wkv, wgate, kv_state, score_state, ape, norm_weight, rope_sin, rope_cos,
    block_table, cu_seqlens, start_pos, seqused=None,
    rope_head_dim=64, cmp_ratio=128, coff=1, norm_eps=1e-6, rotary_mode=2):
    """CPU reference for C128A (coff=1) paged mode, TH layout. Returns (cmp_kv, update_kv, update_score)."""
    x = x.float().numpy()
    wkv = wkv.float().numpy()
    wgate = wgate.float().numpy()
    kv_state = kv_state.numpy().copy()
    score_state = score_state.numpy().copy()
    ape = ape.float().numpy()
    norm_weight = norm_weight.float().numpy()
    rope_sin = rope_sin.float().numpy()
    rope_cos = rope_cos.float().numpy()
    cu_seqlens = cu_seqlens.numpy()
    start_pos = start_pos.numpy()

    head_dim = wkv.shape[0] // coff
    B = len(start_pos)
    new_kv_state = np.matmul(x, wkv.T, dtype=np.float32)
    new_score_state = np.matmul(x, wgate.T, dtype=np.float32)

    cmp_kv = np.zeros((min(x.shape[0], x.shape[0] // cmp_ratio + B), head_dim), dtype=np.float32)
    update_kv = np.zeros_like(kv_state, dtype=bool)
    update_score = np.zeros_like(score_state, dtype=bool)

    out_sum_sc_cnt = 0
    for b_idx in range(B):
        batch_start_pos = start_pos[b_idx]
        batch_seq_used = seqused[b_idx] if seqused is not None else cu_seqlens[b_idx + 1] - cu_seqlens[b_idx]
        compress_seq_id = (batch_start_pos + batch_seq_used) // cmp_ratio * cmp_ratio

        batch_seq_idx = 0
        while batch_seq_idx < batch_seq_used:
            start_seq_idx = batch_start_pos + batch_seq_idx
            end_seq_idx = start_seq_idx // cmp_ratio * cmp_ratio + cmp_ratio
            if end_seq_idx > batch_start_pos + batch_seq_used:
                end_seq_idx = batch_start_pos + batch_seq_used

            base_offset = cu_seqlens[b_idx]
            start_offset = base_offset + (start_seq_idx - batch_start_pos)
            end_offset = base_offset + (end_seq_idx - batch_start_pos)

            start_seq_id_in_sc = start_seq_idx % cmp_ratio
            end_seq_idx_in_sc = start_seq_id_in_sc + (end_seq_idx - start_seq_idx)
            new_score_state[start_offset:end_offset, :] = np.add(
                new_score_state[start_offset:end_offset, :], ape[start_seq_id_in_sc:end_seq_idx_in_sc, :])

            save_flag = True
            compress_flag = True if start_seq_idx < compress_seq_id else False

            if save_flag:
                _write_state_page_cache(kv_state, update_kv, new_kv_state[start_offset:end_offset, :], b_idx,
                                        start_seq_idx, end_seq_idx, block_table,
                                        cache_mode=1, batch_start_pos=batch_start_pos, history_size=coff * cmp_ratio)
                _write_state_page_cache(score_state, update_score, new_score_state[start_offset:end_offset, :], b_idx,
                                        start_seq_idx, end_seq_idx, block_table,
                                        cache_mode=1, batch_start_pos=batch_start_pos, history_size=coff * cmp_ratio)

            if compress_flag:
                sc_kv_state = np.zeros((coff, cmp_ratio, head_dim), dtype=np.float32)
                sc_score_state = np.full((coff, cmp_ratio, head_dim), -float("inf"), dtype=np.float32)

                coff_id = coff - 1
                d_start = coff_id * head_dim
                d_end = (coff_id + 1) * head_dim
                cnt_from_state = 0
                if batch_start_pos == start_seq_idx:
                    cnt_from_state = batch_start_pos % cmp_ratio
                    if cnt_from_state > 0:
                        copy_start_seq_id = batch_start_pos - cnt_from_state
                        copy_end_seq_id = batch_start_pos
                        sc_kv_state[coff_id, 0:cnt_from_state, :] = _read_state_page_cache(
                            kv_state, b_idx, copy_start_seq_id, copy_end_seq_id, block_table, d_start, d_end,
                            cache_mode=1, batch_start_pos=batch_start_pos, history_size=coff * cmp_ratio)
                        sc_score_state[coff_id, 0:cnt_from_state, :] = _read_state_page_cache(
                            score_state, b_idx, copy_start_seq_id, copy_end_seq_id, block_table, d_start, d_end,
                            cache_mode=1, batch_start_pos=batch_start_pos, history_size=coff * cmp_ratio)
                sc_kv_state[coff_id, cnt_from_state:cmp_ratio, :] = new_kv_state[start_offset:end_offset, d_start:d_end]
                sc_score_state[coff_id, cnt_from_state:cmp_ratio, :] = new_score_state[start_offset:end_offset, d_start:d_end]

                sc_kv_state = sc_kv_state.reshape(coff * cmp_ratio, head_dim)
                sc_score_state = sc_score_state.reshape(coff * cmp_ratio, head_dim)
                sc_score_state = _softmax_columns(sc_score_state)
                sc_data = sc_kv_state * sc_score_state
                sc_cmp_kv = np.sum(sc_data, axis=0, keepdims=True)
                sc_cmp_kv = _rms_norm(sc_cmp_kv, norm_weight, norm_eps)
                sc_cmp_kv[:, -rope_head_dim:] = _rotary_emb(
                    sc_cmp_kv[:, -rope_head_dim:], rope_sin[out_sum_sc_cnt, :], rope_cos[out_sum_sc_cnt, :], rotary_mode)
                cmp_kv[out_sum_sc_cnt, :] = sc_cmp_kv
                out_sum_sc_cnt += 1

            # advance loop index (same as reference impl)
            batch_seq_idx = end_seq_idx - batch_start_pos

    return cmp_kv, update_kv, update_score


def _has_compressor():
    try:
        getattr(torch.ops.custom, "compressor")
        return True
    except (AttributeError, RuntimeError):
        return False


class TestCompressorSmoke(unittest.TestCase):
    """Verify Compressor op runs, returns correct shape/dtype, and matches CPU
    golden + leaves state untouched where the spec says so (C128A paged)."""

    HIDDEN = 2048
    ROPE_HEAD_DIM = 64
    ROTARY_MODE = 2
    CACHE_MODE = 1
    BLOCK_SIZE = 128

    @classmethod
    def setUpClass(cls):
        torch_npu.npu.set_device(0)
        if not _has_compressor():
            raise unittest.SkipTest("torch.ops.custom.compressor not registered; check env")

    # ── helpers ──────────────────────────────────────────────────
    def _make_tensors(self, B, S, head_dim, coff, cmp_ratio, dtype=torch.bfloat16,
                      cache_mode=1, seqused=None):
        T = B * S
        cu_seqlens = torch.arange(0, T + 1, S, dtype=torch.int32)

        x = torch.randn(T, self.HIDDEN, dtype=dtype)
        wkv = torch.randn(coff * head_dim, self.HIDDEN, dtype=dtype)
        wgate = torch.randn(coff * head_dim, self.HIDDEN, dtype=dtype)
        ape = torch.randn(cmp_ratio, coff * head_dim, dtype=torch.float32)
        norm_weight = torch.randn(head_dim, dtype=torch.float32)
        rope_sin = torch.randn(min(T, T // cmp_ratio + B), self.ROPE_HEAD_DIM, dtype=torch.float32)
        rope_cos = torch.randn(min(T, T // cmp_ratio + B), self.ROPE_HEAD_DIM, dtype=torch.float32)

        if cache_mode == 1:  # paged
            max_blocks = (S + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
            block_table = torch.arange(1, B * max_blocks + 1, dtype=torch.int32).reshape(B, max_blocks)
            state_cache = torch.randn(B * max_blocks + 1, self.BLOCK_SIZE, 2 * coff * head_dim, dtype=torch.float32)
        else:  # cache_mode == 2 explicit loc table: state_block_table is [B, coff*cmp_ratio+S]
            ring_size = coff * cmp_ratio + S - 1
            table_width = coff * cmp_ratio + S
            block_table = torch.zeros(B, table_width, dtype=torch.int32)
            for b in range(B):
                for j in range(table_width):
                    block_table[b, j] = b * ring_size + j % ring_size
            state_cache = torch.randn(B + 1, ring_size, 2 * coff * head_dim, dtype=torch.float32)

        start_pos = torch.randint(0, 100, (B,), dtype=torch.int32)

        if seqused is None:
            seqused_t = None
        else:
            seqused_t = torch.tensor(seqused, dtype=torch.int32)

        return {
            "x": x, "wkv": wkv, "wgate": wgate, "state_cache": state_cache,
            "ape": ape, "norm_weight": norm_weight,
            "rope_sin": rope_sin, "rope_cos": rope_cos,
            "state_block_table": block_table,
            "cu_seqlens": cu_seqlens, "seqused": seqused_t,
            "start_pos": start_pos,
            "rope_head_dim": self.ROPE_HEAD_DIM,
            "cmp_ratio": cmp_ratio, "coff": coff,
            "norm_eps": 1e-6, "rotary_mode": self.ROTARY_MODE,
            "cache_mode": cache_mode,
        }

    def _to_npu(self, d):
        return {k: v.to("npu:0") if isinstance(v, torch.Tensor) else v for k, v in d.items()}

    def _run(self, npu):
        return torch.ops.custom.compressor(
            npu["x"], npu["wkv"], npu["wgate"], npu["state_cache"],
            npu["ape"], npu["norm_weight"],
            rope_sin=npu["rope_sin"], rope_cos=npu["rope_cos"],
            rope_head_dim=npu["rope_head_dim"], cmp_ratio=npu["cmp_ratio"],
            state_block_table=npu["state_block_table"],
            cu_seqlens=npu["cu_seqlens"], seqused=npu["seqused"],
            start_pos=npu["start_pos"], coff=npu["coff"],
            norm_eps=npu["norm_eps"], rotary_mode=npu["rotary_mode"],
            cache_mode=npu["cache_mode"],
        )

    # ── shape / dtype smoke ──────────────────────────────────────
    def test_c128a_th(self):
        """C128A: B=2, S=4, coff=1, cmp_ratio=128."""
        p = self._make_tensors(B=2, S=4, head_dim=512, coff=1, cmp_ratio=128)
        out = self._run(self._to_npu(p))
        self.assertEqual(out.dtype, p["x"].dtype)
        self.assertEqual(out.dim(), 2)
        self.assertGreater(out.shape[0], 0)

    def test_c4a_th(self):
        """C4A: B=2, S=4, coff=2, cmp_ratio=4, head_dim=512."""
        p = self._make_tensors(B=2, S=4, head_dim=512, coff=2, cmp_ratio=4)
        out = self._run(self._to_npu(p))
        self.assertEqual(out.dtype, p["x"].dtype)
        self.assertGreater(out.numel(), 0)

    def test_c4li_th(self):
        """C4Li: B=2, S=4, coff=2, cmp_ratio=4, head_dim=128."""
        p = self._make_tensors(B=2, S=4, head_dim=128, coff=2, cmp_ratio=4)
        out = self._run(self._to_npu(p))
        self.assertEqual(out.dtype, p["x"].dtype)
        self.assertGreater(out.numel(), 0)

    def test_bf16_vs_fp16(self):
        """C128A: compare BF16 vs FP16 outputs (both should run without error)."""
        for dtype in [torch.bfloat16, torch.float16]:
            p = self._make_tensors(B=1, S=1, head_dim=512, coff=1, cmp_ratio=128, dtype=dtype)
            out = self._run(self._to_npu(p))
            self.assertEqual(out.dtype, dtype)

    def test_c128a_ring_mode(self):
        """C128A ring (cache_mode=2): B=2, S=4, bank-table [B, coff*cmp_ratio+S]."""
        p = self._make_tensors(B=2, S=4, head_dim=512, coff=1, cmp_ratio=128, cache_mode=2)
        out = self._run(self._to_npu(p))
        self.assertEqual(out.dtype, p["x"].dtype)
        self.assertGreater(out.numel(), 0)

    def test_c4a_ring_mode(self):
        """C4A ring (cache_mode=2): B=2, S=4, coff=2, cmp_ratio=4."""
        p = self._make_tensors(B=2, S=4, head_dim=512, coff=2, cmp_ratio=4, cache_mode=2)
        out = self._run(self._to_npu(p))
        self.assertEqual(out.dtype, p["x"].dtype)
        self.assertGreater(out.numel(), 0)

    def test_c128a_empty_batch(self):
        """C128A with a trailing empty batch (seqused=[S, 0]) must not crash."""
        B, S = 2, 4
        p = self._make_tensors(B=B, S=S, head_dim=512, coff=1, cmp_ratio=128, seqused=[S, 0])
        out = self._run(self._to_npu(p))
        self.assertEqual(out.dtype, p["x"].dtype)

    # ── CPU golden (C128A paged) ─────────────────────────────────
    def test_cpu_golden_c128a_paged(self):
        """C128A paged cmp_kv must match CPU reference within BF16 tolerance.

        Uses start_pos=0 (block-aligned) so the first compress group is fed
        entirely from current tokens; S = cmp_ratio yields exactly one full
        compress group with no overlap-copy from history state.
        """
        B, S = 1, 128  # one full compress group, no overlap read from state
        p = self._make_tensors(B=B, S=S, head_dim=512, coff=1, cmp_ratio=128, dtype=torch.bfloat16)
        p["start_pos"] = torch.zeros(B, dtype=torch.int32)
        npu = self._to_npu(p)

        out = self._run(npu)
        out_cpu = out.cpu().to(torch.float32)

        head_dim = p["norm_weight"].shape[0]
        kv_state = p["state_cache"][:, :, :head_dim]
        score_state = p["state_cache"][:, :, head_dim:]
        golden, _, _ = _cpu_compressor_c128a_paged(
            p["x"], p["wkv"], p["wgate"], kv_state, score_state,
            p["ape"], p["norm_weight"], p["rope_sin"], p["rope_cos"],
            p["state_block_table"], p["cu_seqlens"], p["start_pos"],
            rope_head_dim=p["rope_head_dim"], cmp_ratio=p["cmp_ratio"], coff=p["coff"],
            norm_eps=p["norm_eps"], rotary_mode=p["rotary_mode"],
        )
        golden_t = torch.from_numpy(golden).float()

        # Both outputs have the same #rows (T//cmp_ratio + B for TH). Compare only
        # rows that are actually produced (B=1,S=cmp_ratio -> first row); trailing
        # padding rows are all zeros and would make relative-error NaN.
        n = min(out_cpu.shape[0], golden_t.shape[0])
        valid = ~(golden_t[:n].abs().sum(dim=-1) == 0)
        diff = (out_cpu[:n] - golden_t[:n]).abs()
        denom = golden_t[:n].abs() + 1e-6
        rel = diff / denom
        rel = rel[valid]
        self.assertGreater(rel.numel(), 0, "No valid golden rows to compare")
        max_rel = rel.max().item()
        self.assertLess(
            max_rel, 0.2,
            f"C128A golden mismatch: max relative err {max_rel:.4f}, "
            f"npu[0,:5]={out_cpu[0,:5].tolist()} golden[0,:5]={golden_t[0,:5].tolist()}",
        )

    # ── state-cache immutability (C128A paged) ───────────────────
    def test_state_cache_untouched_rows(self):
        """Rows of state_cache never written by the op must stay bitwise identical."""
        B, S = 2, 4
        p = self._make_tensors(B=B, S=S, head_dim=512, coff=1, cmp_ratio=128)
        npu = self._to_npu(p)

        # Snapshot rows that should NOT be written:
        #   - row 0 is unused (block ids start at 1)
        #   - trailing padding rows beyond B*max_blocks
        max_blocks = (S + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
        used_rows = set(int(x) for x in p["state_block_table"].reshape(-1).tolist() if x != 0)
        expected_untouched = torch.ones(p["state_cache"].shape[0], dtype=torch.bool)
        for r in used_rows:
            expected_untouched[r] = False
        if 0 in used_rows:
            expected_untouched[0] = False

        snap = npu["state_cache"].cpu().clone()
        self._run(npu)
        after = npu["state_cache"].cpu()

        untouched_idx = torch.nonzero(expected_untouched).flatten().tolist()
        for r in untouched_idx:
            self.assertTrue(
                torch.equal(snap[r], after[r]),
                f"state_cache row {r} was modified but should be untouched",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
