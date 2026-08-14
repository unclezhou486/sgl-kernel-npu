# Compressor 算子学习指南（DeepSeek-V4 稀疏注意力 KV 压缩）

> 面向新手：先看整体，再逐层深入。每层只回答"它在干嘛"，细节等用到再看。

## 0. 一句话 + 整体视图

**一句话**：compressor 是 DeepSeek-V4 稀疏注意力里的 KV 压缩算子——把连续 `r` 个 token（r=4 或 128），用学到的权重（gate）压缩成 1 条带位置信息的 KV（省显存），历史 token 的中间结果存在 `state_cache` 里跨轮复用。

```
一句话：把 r 个 token，用学到的权重(gate)，压成 1 条带位置信息的 KV（省显存）
                      ▲
        ┌─────────────┴─────────────┐
        │                           │
第1层 软件栈                     第3层 算法
sglang 调它                    6步，本质2件事:
  torch.ops.custom.compressor     ① 投影: x → kv + score
  → aclnnCompressor             ② 压缩: softmax(score)×kv 加权求和
  → CANN 框架                    + 后处理 norm/rope
        │                           │
        └─────────────┬─────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
第2层 文件结构                  第4层 数据流
host:  def(定义)               x → kv/score → 压缩KV → cmp_kv
       proto(输出形状)          state_cache(跨轮记账)
       tiling(怎么启动)          tilingKey(选kernel模板)
device: kernel(怎么算)
```

---

## 1. 完整调用链路（含 `import custom_ops`）

### 第 0 阶段：sglang 启动，`import custom_ops` 注册入口

sglang 初始化 NPU 后端时（`sglang/.../hardware_backend/npu/utils.py`）：

```python
@_call_once
def init_npu_backend():
    assert _is_npu, "NPU backend initialization called on non-NPU device."
    try:
        import custom_ops  # noqa: F401     ← 关键：加载 custom_ops 包
        import sgl_kernel_npu  # noqa: F401
    except ImportError as e:
        logger.warning("NPU custom kernel packages unavailable: %s", e)
    import torch_npu
    ...
```

`import custom_ops` 触发 `custom_ops/__init__.py`，做三件事：

```python
from . import custom_ops_lib                          # ① 加载 C++ 扩展（.so）
from .converter import (npu_compressor, ...)          # ② 加载 GE converter（python）
...
custom_ops_module = getattr(torch.ops, 'custom', None)
for op_name in dir(custom_ops_module):                # ③ 把 torch.ops.custom.* 挂到 torch_npu
    if op_name.startswith('_'): continue
    setattr(torch_npu, op_name, getattr(custom_ops_module, op_name))
```

**① `custom_ops_lib.so`（C++）**：用 `TORCH_LIBRARY(custom, m)` 注册 compressor 的 schema：

```
compressor(Tensor x, Tensor wkv, Tensor wgate, Tensor(a!) state_cache, Tensor ape,
           Tensor norm_weight, Tensor rope_sin, Tensor rope_cos,
           int rope_head_dim, int cmp_ratio, *,
           Tensor? state_block_table=None, Tensor? cu_seqlens=None,
           Tensor? seqused=None, Tensor? start_pos=None,
           int coff=1, float norm_eps=1e-6, int rotary_mode=1, int cache_mode=1) -> (Tensor)
```

- `Tensor(a!) state_cache` = 就地更新标注（既是输入也是输出）
- 无 `state_cache_stride_dim0`（converter 内部算）
- 另提供 C++ eager 实现 `custom::compressor(...)`（内部调 `aclnnCompressor`）

**② `converter/npu_compressor.py`**：注册 GE 图转换器（torch.compile / torchair 成图用）：

```python
@register_fx_node_ge_converter(torch.ops.custom.compressor.default)
def convert_compressor(x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos,
                       rope_head_dim, cmp_ratio, *, state_block_table=None, cu_seqlens=None,
                       seqused=None, start_pos=None, coff=1, norm_eps=1e-6,
                       rotary_mode=1, cache_mode=1):
    state_cache_stride_dim0 = int(state_cache.symsize[-2]) * int(state_cache.symsize[-1])  # 内部算 stride
    out = torchair.ge.custom_op(
        "Compressor", x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos,
        state_block_table, cu_seqlens, seqused, start_pos,
        rope_head_dim, cmp_ratio, coff, norm_eps, rotary_mode, cache_mode,
        state_cache_stride_dim0)
    return (out[0])
```

### 第 1 阶段：模型前向

`layers/attention/dsv4/compressor.py` 的 `Compressor` 层（持有 `wkv_gate/ape/norm/freqs_cis` 权重）前向：`compressor(x, forward_batch)` → 派发到 NPU 后端。

### 第 2 阶段：NPU 后端组装参数并调用（`ascend_dsv4_backend.py`）

```python
def forward_compress(self, compressor, x, forward_batch):
    if (forward_batch.forward_mode.is_prefill() and not is_target_verify()):
        return self._forward_compress_native(compressor, x, forward_batch)   # 纯PyTorch参考，不调op
    # decode/verify 走 fused 路径：
    self._ensure_fused_caches(compressor)          # 拆 wkv_gate → _fused_wkv_w/_fused_wgate_w
    state_cache = pool.get_state_cache(...)        # 取持久 state
    cos, sin = get_fused_compressor_rope_cos_sin(...)   # 造 rope
    cmp_kv = torch.ops.custom.compressor(          # ★真正调用（L348）
        x, compressor._fused_wkv_w, compressor._fused_wgate_w, state_cache,
        compressor.ape, compressor._fused_norm_weight_fp32,
        rope_sin=sin, rope_cos=cos, rope_head_dim=..., cmp_ratio=ratio,
        state_block_table=page_table, cu_seqlens=..., seqused=..., start_pos=...,
        coff=coff, norm_eps=..., rotary_mode=2, cache_mode=1)
    # 之后 _compressor_epilog_npu(cmp_kv) 写进 C4/C128 KV pool
```

### 第 3 阶段：`torch.ops.custom.compressor` 分岔两条路

```
torch.ops.custom.compressor(18个参数)
   ├─【eager】 C++ custom::compressor(...)
   │     ├─ construct_compressor_output_tensor() → 分配 cmp_kv
   │     └─ aclnnCompressorGetWorkspaceSize(...) → aclnnCompressor(...) → aclnn executor
   └─【GE图】 torchair 匹配 register_fx_node_ge_converter
         └─ convert_compressor(...) → torchair.ge.custom_op("Compressor", ...) → GE 图节点
              → GE executor 执行
```

### 第 4 阶段：`aclnnCompressor`（vendor 生成的 API）

`libcust_opapi.so` 里（由 `compressor_def.cpp` 用 opbuild 生成）：

```cpp
aclnnStatus aclnnCompressorGetWorkspaceSize(..., &workspaceSize, &executor);  // 内部跑 tiling
aclnnStatus aclnnCompressor(workspace, workspaceSize, executor, stream);      // 执行
```

### 第 5 阶段：CANN 框架（GE executor）

```
InferShape（compressor_proto.cpp）→ 推导 cmp_kv 形状（BSH:(B,Sr,H) / TH:(Sr,H)）
Tiling（compressor_tiling.cpp）  → 校验 → 算 workspace → GenTilingKey → SetBlockDim(20)
tilingKey → 匹配 bin-info（Compressor_<hash>.json 的 simplifiedKey）
         → 加载 Compressor_<hash>.o（日志可见 bin path）
```

### 第 6 阶段：kernel（op_kernel/）

```
compressor<Layout,DType,Coff,...> 入口 → 解 tilingData → if constexpr 选 PERF
  → CompressorKernelPerf::Init（领任务：dIdx/curGroupIdx/loopTimes/buffer）
  → Process 循环（loopTimes 轮）：
       AIC: ComputeMm1  → x@wkvᵀ/x@wgateᵀ → kv/score → GM
       AIV: ComputeVec1 → GM读MM1结果 + state重叠 + softmax + 加权和 → GM
       AIV: ComputeVec2 → GM读半成品 + RMSNorm + RoPE → cmp_kv
    跨核 flag 同步 + 每 nSize=2 轮 SyncAll
```

### 第 7 阶段：下游

```
cmp_kv → _compressor_epilog_npu → 写 C4/C128 KV pool（Indexer 场景量化成 INT8）
→ 稀疏注意力 top-k 选择（npu_quant_lightning_indexer）+ 注意力（npu_sparse_attn_sharedkv）消费
```

---

## 2. 参数与维度符号

### 18 个参数（sglang 实际调用契约）

| # | 参数 | 含义 |
|---|---|---|
| 1 | `x` | token 隐状态 (T,H) 或 (B,S,H)，BF16/FP16 |
| 2 | `wkv` | KV 投影权重 (coff·D, H) |
| 3 | `wgate` | gate 投影权重 (coff·D, H) |
| 4 | `state_cache` | 持久 FP32 状态，就地更新 `Tensor(a!)` |
| 5 | `ape` | 绝对位置嵌入 (r, coff·D) |
| 6 | `norm_weight` | RMSNorm 权重 (D,) |
| 7 | `rope_sin` | RoPE sin 表 |
| 8 | `rope_cos` | RoPE cos 表 |
| 9 | `rope_head_dim` | RoPE 维度（64） |
| 10 | `cmp_ratio` | 压缩比（4 或 128） |
| 11 | `state_block_table` | 分页/ring 位置映射 |
| 12 | `cu_seqlens` | TH 布局各 batch 前缀和 (B+1,) |
| 13 | `seqused` | 本轮有效 token 数 (B,) |
| 14 | `start_pos` | 绝对位置 (B,) |
| 15 | `coff` | 重叠系数（1 或 2） |
| 16 | `norm_eps` | RMSNorm eps |
| 17 | `rotary_mode` | RoPE 模式（1=half, 2=interleave） |
| 18 | `cache_mode` | 1=分页, 2=ring |

> `state_cache_stride_dim0` 不在 schema 里：converter 内部算（`state_cache.shape[-2]*shape[-1]`）。

### 维度符号

| 符号 | 含义 | 推导 |
|---|---|---|
| B | batch 数 | `cu_seqlens[0]-1` 或 `x[0]` |
| S | 每 batch 序列长度 | `x[1]`（BSH） |
| T | 总 token = B×S | `x[0]`（TH） |
| H | x 特征维（MM1 的 K） | `x[-1]` |
| D | 压缩 KV 宽度（head_dim） | `norm_weight[0]`，128/512 |
| coff | 重叠系数 | 1（无重叠）/ 2（有重叠） |
| r | 压缩比 | 4 或 128 |
| Sr | 压缩后条数 | ≈ ceil(S/r) |
| rD | RoPE 维度 | 64 |
| width | coff·D | 投影/state 行宽 |

三个变体：C4A（D=512,coff=2,r=4）、C4Li（D=128,coff=2,r=4）、C128A（D=512,coff=1,r=128）。

### state 概念

```
state_cache.shape = (block_num, block_size, 2·coff·D)
   └─ 最后一维 = [ kv_state(coff·D) | score_state(coff·D) ]
                  前半投影的KV内容  后半gate打分（初始-inf）
```

- 持久、就地更新，跨轮累积（压缩组可能跨多次调用）
- block 0 = skip sentinel（kv 0 / score -inf → softmax 权重 0）

---

## 3. 算法（6 步）

```
① 投影(MM1):   kv = x@wkvᵀ, score = x@wgateᵀ        （AIC 核）
② 加ape:       score += ape                          （组内位置）
③ 写/读state:  写本轮 kv/score；凑组时读重叠历史
④ softmax:     组内对 score 做列 softmax             （AIV 核）
⑤ 加权和:      Σ softmax·kv → 1 条 D 宽压缩KV        （AIV 核）
⑥ 后处理:      RMSNorm + RoPE(后64列) → bf16         （AIV 核）
```

- 为什么 gate 替代平均：避免稀释关键 token
- 为什么持久 state：decode 逐步，组凑满才压
- 为什么重叠(coff=2)：边界上下文不丢、流式友好
- 为什么 norm/rope：稳定数值 + 内嵌位置

---

## 4. 文件结构

### 仓库侧（`csrc/attentions/csrc/ops/compressor/`）

```
op_host/
  compressor_def.cpp      定义：Input/Output/Attr + OP_ADD（注册算子长什么样）
  compressor_proto.cpp    形状：IMPL_OP_INFERSHAPE（推 cmp_kv 形状/类型）
  compressor_tiling.cpp   切分：IMPL_OP_OPTILING（定 workspace/tilingKey/blockDim）
  compressor_tiling.h / compressor_tiling_data.h
op_kernel/
  compressor.cpp          入口：模板派发（compressor<...> → 选 PERF）
  compressor_kernel_perf.h 主类：Init + Process（三段调度）
  compressor_block_cube_perf.h  MM1：AIC 核投影（L1/L0 搬数 + Mmad）
  compressor_block_vec_perf.h  Vec1/Vec2：AIV 核压缩/后处理
  compressor_comm.h       公共字典：常量/结构体/枚举（ConstInfo/RunInfo…）
  compressor_tools.h      公共工具：seq 信息 + 切片迭代器
  rms_norm.h / rope.h / soft_max.h  数学工具
```

### 四级注册

| 级 | 文件 | 注册宏 | 干什么 |
|---|---|---|---|
| 1 定义 | `compressor_def.cpp` | `OP_ADD` | 算子长什么样 |
| 2 形状 | `compressor_proto.cpp` | `IMPL_OP_INFERSHAPE` | 输出形状/类型 |
| 3 切分 | `compressor_tiling.cpp` | `IMPL_OP_OPTILING` | 怎么启动 |
| 4 内核 | `op_kernel/compressor.cpp` | `__global__ __aicore__` | 怎么算 |

---

## 5. tiling 与内存

### tiling 主流程（`RunBigKernelTiling`）

```
GetNpuInfo → 校验 → SetBaseInfo → SetPageAttentionInfo → CheckFeature
→ SetTemplateId(PERF) → SetInnerSplitInfo → SetWorkSpaceInfo
→ CalcWorkSpace（算 GM workspace 总量） → GenTilingKey（打包模板参数）
→ SetBlockDim(aicNum=20)
```

### 内存层次（Ascend A3）

| 级别 | 角色 | 谁用 |
|---|---|---|
| GM | 主内存 + workspace | 所有核 |
| L2 | 隐式 cache | 硬件 |
| L1 | cube 侧 GM→L0 中转 | AIC |
| L0_A/B | 矩阵乘左右操作数 | AIC |
| L0_C | 累加器 | AIC |
| UB | vector 计算暂存 | AIV |

AscendC 的 UB/L1/L0 都是**显式 buffer**（`InitBuffer` + `DataCopy` 手动管理）；CUDA 的 L1/L2 是隐式 cache。

### `CalcWorkSpace`（C4Li 实测 24,649,728 字节）

```
maxGroupNum = aicNum / (headDim/dBaseSize) = 20/2 = 10
workspace = libapi(16,777,216)
          + mm1KvResSize×10×4×2      // MM1 kv 结果，每组一份，fp32，双缓冲
          + mm1ScoreResSize×10×4×2   // MM1 score 结果
          + vec1TailCacheSize×4×2×2  // Vec1 尾部缓存（kv+score）
          + vec1ResSize×10×4×2       // Vec1 结果
```

---

## 6. tilingKey 与模板选择

```
template_tiling_key.h: ASCENDC_TPL_ARGS_DECL 定义 bit 布局
   X_LAYOUT(1bit) X_DTYPE(4bit) COFF(2bit) ROTARY_MODE(2bit) CACHE_MODE(2bit) TEMPLATE_ID(2bit)
GenTilingKey: GET_TPL_TILING_KEY(...) 按实际输入+attr 打包成 64 位
框架: tilingKey → 匹配 bin-info JSON 的 simplifiedKey → 加载 Compressor_<hash>.o
kernel: compressor<...> 用 if constexpr 选 PERF → CompressorKernelPerf
```

> 注意：`FastEncodeTilingKeyDirect`（CANN 头 `template_argument.h`）存的是"值在 DECL 允许列表里的**下标**"，不是原始值。

---

## 7. kernel 三段（理解核心）

```
x ──► [MM1·AIC] ──► kv/score ──► [Vec1·AIV] ──► 压缩KV半成品 ──► [Vec2·AIV] ──► cmp_kv
        (GM workspace)            (GM workspace)   (攒 nSize 轮)
```

三个词：**流水线**（三段分工）、**分块**（loopTimes 轮，M/D 二维切核）、**同步**（跨核 flag + SyncAll + 双缓冲）。

### M/D 轴

- **M 轴** = 行方向 = token 方向，切成 M 组（coreGroupNum）
- **D 轴** = 列方向 = head_dim，切成 D 块（dBasicBlockNum）
- 每个逻辑核 = (M 组, D 块)，算 mBaseSize 行 × dBaseSize 列

### `ConstInfo`（每核任务卡）

- shape：B/H/S/D/r/ropeHeadDim/normEps/reciprocalD/nSize
- 分核：usedCoreNum/dBaseSize/mBaseSize/dBasicBlockNum/coreGroupNum/dIdx/curGroupIdx/aiCoreIdx
- workspace：dbWorkspaceRatio/mm1KvResSize/mm1ScoreResSize/vec1TailCacheSize/vec1ResSize/mm1ResSize/dbSize
- **注意**：顶部 `bStart/sStart/bEnd/sEnd` 和几个 `tc*/tail*` 字段是**声明但未用**的遗留字段，可忽略。

---

## 8. 为什么不需要 attentions plugin

- sglang 只调 `torch.ops.custom.compressor`（外部 custom_ops 包），从不调 `torch.ops.attentions.compressor`
- `torch.ops.attentions.*` 在 sglang 里只用于多模态 attention 后端（la/block_sparse/rainfusion）
- kernel 迁移 + 构建 vendor（aclnnCompressor）即可被 custom.compressor 消费，无需 plugin

---

## 9. 部署（让迁移的 kernel 被实际调用）

```
kernel 迁移 → build.sh 构建 → aie_ascendc vendor（aclnnCompressor）
  → 部署（ASCEND_CUSTOM_OPP_PATH 指向 vendor，或装 .run + source set_env.bash）
  → 重启 sglang → torch.ops.custom.compressor 自动用上你的 kernel
```

- sglang / custom_ops 代码零改动（按名字解析）
- 注意同名覆盖：确保你的 vendor 优先级高于 sys/其他 vendor（`ASCEND_CUSTOM_OPP_PATH` 优先）
- 验证：kernel 的 build marker printf（`Compressor: NEW kernel build`）出现即命中

---

## 10. 可选的清理项（不影响运行，上游原样）

- `compressor_tiling.h` `NORM_EPS_NAME = "nrom_eps"` 拼写
- proto 注释误导（`(B,S,N,Hckv)` vs 实际 `(B,Sr,H)`）
- proto 输出列宽用 H、实际是 D
- `CompressorKernel` 空壳 + NORMAL 死分支
- `ConstInfo` 中未用字段
- proto 对 OPTIONAL 输入用 `GetRequiredInputShape`（可用 `GetInputShape`）

---

# 附录 A：完整调用链路详述（含 `import custom_ops`）

> 这是完整的调用链路全流程描述，从 sglang 启动到 kernel 到下游，每一阶段都附实际代码。

## A.0 完整调用链路总图

```
┌─ ① sglang 模型层 ────────────────────────────────────────────┐
│  layers/attention/dsv4/compressor.py 的 Compressor 层          │
│  （权重：wkv_gate / ape / norm / freqs_cis）                   │
│  前向 compressor(x, forward_batch)                            │
└──────────────────────────┬────────────────────────────────────┘
                           ▼
┌─ ② sglang NPU 硬件后端 ───────────────────────────────────────┐
│  hardware_backend/npu/attention/ascend_dsv4_backend.py        │
│  forward_core_compressor → forward_compress                    │
│  prefill非verify → _forward_compress_native（纯PyTorch参考）    │
│  decode/verify → fused 路径：                                  │
│    拆 wkv_gate → 取 state_cache → 造 rope →                    │
│    torch.ops.custom.compressor(18个参数)   ← L348 真正调用       │
└──────────────────────────┬────────────────────────────────────┘
                           ▼
┌─ ③ custom_ops 包（外部 cann-recipes-infer）───────────────────┐
│  import custom_ops（sglang init_npu_backend 里）               │
│    ├─ custom_ops_lib.so（C++）：注册 torch.ops.custom.compressor│
│    └─ converter/npu_compressor.py（GE 转换器）                  │
│  调用时两条路：                                                 │
│    [eager]  C++ custom::compressor() → aclnnCompressor         │
│    [GE图]   converter → torchair.ge.custom_op("Compressor")    │
└──────────────────────────┬────────────────────────────────────┘
                           ▼
┌─ ④ aclnnCompressor（vendor 生成的 API）───────────────────────┐
│  libcust_opapi.so 里                                           │
│  aclnnCompressorGetWorkspaceSize(...)  → 内部跑 tiling         │
│  aclnnCompressor(workspace, size, executor, stream) → 执行      │
└──────────────────────────┬────────────────────────────────────┘
                           ▼
┌─ ⑤ CANN 框架（GE executor）───────────────────────────────────┐
│  InferShape（compressor_proto.cpp）→ 推导 cmp_kv 形状          │
│  Tiling（compressor_tiling.cpp）→ workspace/tilingKey/blockDim │
│  tilingKey 匹配 bin-info → 选 Compressor_<hash>.o 加载         │
└──────────────────────────┬────────────────────────────────────┘
                           ▼
┌─ ⑥ kernel（op_kernel/）──────────────────────────────────────┐
│  compressor<Layout,DType,Coff,...> 入口 → 解 tilingData         │
│  CompressorKernelPerf::Init（领任务）→ Process（主循环）        │
│    MM1(AIC): x@wkvᵀ/x@wgateᵀ → kv/score                       │
│    Vec1(AIV): overlap + softmax + 加权和 → 压缩KV             │
│    Vec2(AIV): RMSNorm + RoPE → cmp_kv                         │
└──────────────────────────┬────────────────────────────────────┘
                           ▼
┌─ ⑦ 下游 ──────────────────────────────────────────────────────┐
│  sglang _compressor_epilog_npu → 写 C4/C128 KV pool            │
│  （Indexer 场景动态量化成 INT8）                                │
│  → 稀疏注意力 top-k + npu_sparse_attn_sharedkv 消费             │
└────────────────────────────────────────────────────────────────┘
```

## A.1 第 0 阶段：sglang 启动，`import custom_ops` 注册

sglang 初始化 NPU 后端时（`hardware_backend/npu/utils.py`）：

```python
@_call_once
def init_npu_backend():
    assert _is_npu, "NPU backend initialization called on non-NPU device."
    try:
        import custom_ops  # noqa: F401     ← 关键：加载 custom_ops 包
        import sgl_kernel_npu  # noqa: F401
    except ImportError as e:
        logger.warning("NPU custom kernel packages unavailable: %s", e)
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    ...
```

`import custom_ops` 触发 `custom_ops/__init__.py` 执行，做三件事：

```python
from . import custom_ops_lib                          # ① 加载 C++ 扩展（.so）
from .converter import (npu_compressor, ...)          # ② 加载 GE converter（python）
...
custom_ops_module = getattr(torch.ops, 'custom', None)
for op_name in dir(custom_ops_module):                # ③ 把 torch.ops.custom.* 挂到 torch_npu
    if op_name.startswith('_'): continue
    setattr(torch_npu, op_name, getattr(custom_ops_module, op_name))
```

**① `custom_ops_lib.so`（C++）**：用 `TORCH_LIBRARY(custom, m)` 注册了 compressor 的 schema：

```
compressor(Tensor x, Tensor wkv, Tensor wgate, Tensor(a!) state_cache, Tensor ape,
           Tensor norm_weight, Tensor rope_sin, Tensor rope_cos,
           int rope_head_dim, int cmp_ratio, *,
           Tensor? state_block_table=None, Tensor? cu_seqlens=None,
           Tensor? seqused=None, Tensor? start_pos=None,
           int coff=1, float norm_eps=1e-6, int rotary_mode=1, int cache_mode=1) -> (Tensor)
```

并为每个 op 提供 C++ eager 实现 `custom::compressor(...)`（内部会调 `aclnnCompressor`）。

**② `converter/npu_compressor.py`**：注册 GE 图转换器（torch.compile / torchair 成图用）：

```python
@register_fx_node_ge_converter(torch.ops.custom.compressor.default)
def convert_compressor(x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos,
                       rope_head_dim, cmp_ratio, *, state_block_table=None, cu_seqlens=None,
                       seqused=None, start_pos=None, coff=1, norm_eps=1e-6,
                       rotary_mode=1, cache_mode=1):
    state_cache_stride_dim0 = int(state_cache.symsize[-2]) * int(state_cache.symsize[-1])  # 内部算 stride
    out = torchair.ge.custom_op(
        "Compressor", x, wkv, wgate, state_cache, ape, norm_weight, rope_sin, rope_cos,
        state_block_table, cu_seqlens, seqused, start_pos,
        rope_head_dim, cmp_ratio, coff, norm_eps, rotary_mode, cache_mode,
        state_cache_stride_dim0)
    return (out[0])
```

> **注册完的效果**：之后任何地方都能 `torch.ops.custom.compressor(...)` 调（eager 走 C++，成图走 converter）。

## A.2 第 1 阶段：模型前向，进入 Compressor

`layers/attention/dsv4/compressor.py` 的 `Compressor` 层（持有 `wkv_gate/ape/norm/freqs_cis` 权重）前向：

```
compressor(x, forward_batch) → 派发到 NPU 后端
```

## A.3 第 2 阶段：NPU 后端组装参数并调用（`ascend_dsv4_backend.py`）

```python
def forward_core_compressor(self, x, forward_batch, layer_id, compressor):
    if forward_batch.forward_mode.is_idle(): return
    compressor(x, forward_batch)          # → 进 forward_compress

def forward_compress(self, compressor, x, forward_batch):
    if (forward_batch.forward_mode.is_prefill() and not is_target_verify()):
        return self._forward_compress_native(compressor, x, forward_batch)   # 纯PyTorch参考，不调op
    # ↓ decode/verify 走 fused 路径：
    self._ensure_fused_caches(compressor)          # 拆 wkv_gate → _fused_wkv_w/_fused_wgate_w
    state_cache = pool.get_state_cache(...)        # 从 NPUCompressStatePool 取持久 state
    cos, sin = get_fused_compressor_rope_cos_sin(compressor.freqs_cis, positions_cmp, ...)  # 造 rope
    cmp_kv = torch.ops.custom.compressor(          # ★真正调用（L348）
        x, compressor._fused_wkv_w, compressor._fused_wgate_w, state_cache,
        compressor.ape, compressor._fused_norm_weight_fp32,
        rope_sin=sin, rope_cos=cos, rope_head_dim=..., cmp_ratio=ratio,
        state_block_table=page_table, cu_seqlens=..., seqused=..., start_pos=...,
        coff=coff, norm_eps=..., rotary_mode=2, cache_mode=1)
    # 之后 _compressor_epilog_npu(cmp_kv) 写进 C4/C128 KV pool
```

## A.4 第 3 阶段：`torch.ops.custom.compressor` 分岔两条路

```
torch.ops.custom.compressor(18个参数)
   ├─【eager】 C++ custom::compressor(x, wkv, wgate, state_cache, ...)
   │     ├─ construct_compressor_output_tensor() → 分配 cmp_kv
   │     └─ aclnnCompressorGetWorkspaceSize(...) → aclnnCompressor(...) → aclnn executor 执行
   └─【GE图】 torchair 匹配 register_fx_node_ge_converter
         └─ convert_compressor(...) → torchair.ge.custom_op("Compressor", ...) → GE 图节点
              → GE executor 执行图 → 跑算子
```

（sglang 图模式走 GE 图；两条路最终都执行同一个 "Compressor" 算子。）

## A.5 第 4 阶段：`aclnnCompressor`（vendor 生成的 API）

`libcust_opapi.so` 里（由 `compressor_def.cpp` 用 opbuild 生成）：

```cpp
aclnnStatus aclnnCompressorGetWorkspaceSize(x, wkv, wgate, stateCacheRef, ape, normWeight,
    ropeSin, ropeCos, stateBlockTableOpt, cuSeqlensOpt, sequsedOpt, startPosOpt,
    ropeHeadDim, cmpRatio, coff, normEps, rotaryMode, cacheMode, stateCacheStrideDim0,
    cmpKvOut, &workspaceSize, &executor);    // 内部会跑 tiling
aclnnStatus aclnnCompressor(workspace, workspaceSize, executor, stream);   // 真正执行
```

## A.6 第 5 阶段：CANN 框架（GE executor）

```
InferShape（compressor_proto.cpp）→ 推导 cmp_kv 形状（BSH:(B,Sr,H) / TH:(Sr,H)）
Tiling（compressor_tiling.cpp）  → 校验 → 算 workspace → GenTilingKey → SetBlockDim(20)
tilingKey → 匹配 bin-info（Compressor_<hash>.json 的 simplifiedKey）
         → 加载 Compressor_<hash>.o（日志: bin path ...）
```

## A.7 第 6 阶段：kernel（op_kernel/）

```
compressor<Layout,DType,Coff,...> 入口 → 解 tilingData → if constexpr 选 PERF
  → CompressorKernelPerf::Init（领任务：dIdx/curGroupIdx/loopTimes/buffer）
  → Process 循环（loopTimes 轮）：
       AIC: ComputeMm1  → x@wkvᵀ/x@wgateᵀ → kv/score → GM
       AIV: ComputeVec1 → GM读MM1结果 + state重叠 + softmax + 加权和 → GM
       AIV: ComputeVec2 → GM读半成品 + RMSNorm + RoPE → cmp_kv
    跨核 flag 同步 + 每 nSize=2 轮 SyncAll
```

## A.8 第 7 阶段：下游

```
cmp_kv 返回 sglang → _compressor_epilog_npu 写进 C4/C128 KV pool（Indexer 场景量化成 INT8）
→ 稀疏注意力 top-k 选择（npu_quant_lightning_indexer）+ 注意力（npu_sparse_attn_sharedkv）消费
```

## A.9 每层的"一句话 + 代码位置"

| 层 | 干什么 | 代码位置 |
|---|---|---|
| ① 模型层 | 定义 Compressor 层（权重/超参） | `sglang/.../dsv4/compressor.py` |
| ② 后端 | 组装 18 参数并调用 | `ascend_dsv4_backend.py:348` |
| ③ 绑定 | 注册 `torch.ops.custom.compressor` + 转 GE | `custom_ops` 包（C++ + converter） |
| ④ API | 生成层：算 workspace + 执行 | vendor 的 `libcust_opapi.so` |
| ⑤ 框架 | 推形状 + 定切分 + 选 kernel | `compressor_proto/tiling.cpp` |
| ⑥ kernel | 三段流水线算出 cmp_kv | `op_kernel/` |
| ⑦ 下游 | 写 pool 供注意力消费 | sglang epilog |

## A.10 贯穿全程的数据流

```
token 隐状态 x
  → ①投影: kv/score（gate 打分）
  → ②压缩: softmax(score)×kv 加权和 → 压缩KV
  → ③后处理: RMSNorm + RoPE
  → cmp_kv（压缩后的 KV，写进 pool）
  ＋ state_cache（跨轮记账，Tensor(a!) 就地更新）
```

## A.11 两个执行模式的分岔（在 ③ 处）

```
torch.ops.custom.compressor(...)
   ├─ [eager]   C++ 直接调 aclnnCompressor → aclnn executor
   └─ [GE 图]   converter → GE 图节点 "Compressor" → GE executor
```

sglang 图模式走 GE 图；两者最终执行同一个算子。

## A.12 与你迁移的关系（闭环）

```
你迁移的 kernel (csrc/attentions/csrc/ops/compressor)
  → build.sh 构建 → aie_ascendc vendor（aclnnCompressor + Compressor kernel）
  → 部署（ASCEND_CUSTOM_OPP_PATH / 镜像）
  → 上面整条链里的 aclnnCompressor / GE "Compressor" 节点
    自动用上你的 kernel
```

**sglang 端零改动**——它只调 `torch.ops.custom.compressor`，底层按名字解析到部署的 vendor。

## A.13 一句话总结整条链

> **sglang 启动时 `import custom_ops` 注册好入口 → 前向时后端组装 18 参数调 `torch.ops.custom.compressor` →（eager C++ 或 GE converter 两条路）→ `aclnnCompressor` → CANN 框架（推形状/切分/选 kernel 二进制）→ kernel 三段流水算出 cmp_kv → 写 KV pool 供稀疏注意力消费**，全程靠 `state_cache` 跨轮记账、`tilingKey` 选模板。
