# Compressor 算子从 cann-ops-transformer 迁移到 sgl-kernel-npu 改动说明

> 本文记录为了将 `cann-ops-transformer` 的 Compressor 算子迁移进 `sgl-kernel-npu`，并适配 CANN 9.0.0 编译环境所做的全部改动。供本地阅读，不随代码提交。

## 背景

- **源**：`cann-ops-transformer` 分支 `feature/a3-compressor-ring-bcc6304`（commit `c28649488`），算子目录 `experimental/attention/compressor/`
- **目标**：`sgl-kernel-npu` 的 `csrc/attentions/csrc/ops/compressor/`
- **用途**：DeepSeek-V4 的 KV cache 压缩前处理算子，支持 C4A / C4Li / C128A 三种变体，以及 `cache_mode=1`（分页）和 `cache_mode=2`（ring，Atlas A3 新增）
- **编译环境**：CANN 9.0.0（`swr.cn.../lmsysorg/sglang:cann9.0.0-a3-B120` 运行时验证）

---

## 一、新增文件清单

共 19 个文件（不含测试脚本），位于 `csrc/attentions/csrc/ops/compressor/`：

### op_host/（5 个）

| 文件 | 说明 |
|---|---|
| `CMakeLists.txt` | 算子构建配置（重写，见下） |
| `compressor_def.cpp` | 算子注册定义：12 输入、2 输出、7 属性 |
| `compressor_proto.cpp` | shape/dtype 推断（适配 CANN 9.0.0，见下） |
| `compressor_tiling.h` | tiling 数据结构与参数校验声明 |
| `compressor_tiling.cpp` | tiling 实现（适配 CANN 9.0.0，见下） |

### op_kernel/（13 个）

| 文件 | 说明 |
|---|---|
| `compressor.cpp` | kernel 入口（模板 dispatch，改为 arch22 单路径） |
| `compressor_kernel.h` | 基础 kernel 类 |
| `compressor_kernel_perf.h` | 高性能（PERF 模板）kernel 类 |
| `compressor_comm.h` | 通用结构体/枚举/工具 |
| `compressor_block_cube_perf.h` | Cube 计算块 |
| `compressor_block_vec_perf.h` | Vector 计算块 |
| `compressor_tiling_data.h` | tiling 数据结构体（与 host 共享） |
| `compressor_template_tiling_key.h` | 模板 tiling key 定义 |
| `compressor_tools.h` | 工具函数 |
| `compressor_vector_comm.h` | Vector 公共组件 |
| `rms_norm.h` | RMSNorm kernel |
| `rope.h` | RoPE kernel |
| `soft_max.h` | Softmax kernel |

---

## 二、目录结构迁移（arch22 展平）

源仓库中 kernel 代码按架构分目录：

```
op_kernel/
├── arch22/    ← Atlas A3（__CCE_AICORE__ == 220）
└── arch35/    ← Atlas A5（__CCE_AICORE__ == 310）
op_host/
├── arch22/
└── arch35/
```

**目标只迁移 arch22（A3）**，不做 arch35：

- `op_kernel/arch22/*.h`（12 个）→ `op_kernel/*.h`（展平，去掉 `arch22/` 前缀）
- `op_host/arch22/compressor_tiling.{h,cpp}` → `op_host/compressor_tiling.{h,cpp}`（展平）

展平后 kernel 内所有 `#include "compressor_xxx.h"` 等兄弟引用天然有效，无需改动。

---

## 三、`op_host/compressor_tiling.h` 改动

**跨目录 include 路径修正**（源指向 arch22 子目录，目标已展平）：

```cpp
// 源（cann-ops-transformer）：
#include "../../op_kernel/arch22/compressor_template_tiling_key.h"
#include "../../op_kernel/arch22/compressor_tiling_data.h"

// 目标（sgl-kernel-npu）：
#include "../op_kernel/compressor_template_tiling_key.h"
#include "../op_kernel/compressor_tiling_data.h"
```

无其他内容改动。

---

## 四、`op_host/compressor_tiling.cpp` 改动（CANN 9.0.0 兼容）

### 4.1 移除不存在的头文件

```cpp
// 源：CANN 8.x 有此头
#include "err/ops_err.h"
// 目标：CANN 9.0.0 已移除，直接删除
```

### 4.2 新增兼容宏（CANN 9.0.0 已移除旧宏）

源仓库基于更早的 CANN SDK，使用了 `OP_CHECK_IF`、`OP_LOGI`、`OP_LOGE`、`OPS_REPORT_VECTOR_INNER_ERR` 等宏，这些在 CANN 9.0.0 中不存在。在 `#include` 之后、`namespace optiling` 之前新增：

```cpp
// CANN 9.0.0 compatibility macros for originally CANN 8.x macros
#define OP_LOGI(...)
#define OP_LOGE(...)
#define OPS_REPORT_VECTOR_INNER_ERR(op, msg)
#define OP_CHECK_IF(cond, ...) if (cond) { return ge::GRAPH_FAILED; }
```

这样源码中的全部旧宏调用自动映射为：日志宏变空操作，`OP_CHECK_IF` 变为条件判断+失败返回，避免改动大量调用点。

### 4.3 `std::is_same_v` → `std::is_same<...>::value`

CANN 9.0.0 编译标准为 C++14，不支持 C++17 的 `std::is_same_v`：

```cpp
// 源：
if (std::is_same_v<T, bool>) {
// 目标：
if (std::is_same<T, bool>::value) {
```

---

## 五、`op_host/compressor_proto.cpp` 改动（CANN 9.0.0 兼容）

源仓库的 proto 文件使用了 CANN 8.x 宏，9.0.0 中已移除，全部替换为原生 C++：

### 5.1 移除头文件

```cpp
// 源：
#include "log/ops_log.h"
// 目标：CANN 9.0.0 已移除，直接删除
```

### 5.2 `OPS_LOG_E_IF_NULL` → 原生空指针检查

```cpp
// 源：
OPS_LOG_E_IF_NULL(context, xShape, return ge::GRAPH_FAILED)
// 目标：
if (xShape == nullptr) { return ge::GRAPH_FAILED; }
```

（`GetCompressorShapeDim` 中共 13 处）

### 5.3 `OP_CHECK_IF` + `OPS_REPORT_VECTOR_INNER_ERR` → 原生检查

```cpp
// 源：
OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("Compressor", "Context is nullptr."),
           return ge::GRAPH_FAILED);
// 目标：
if (context == nullptr) { return ge::GRAPH_FAILED; }
```

（`InferDataTypeCompressor`、`InferShapeCompressor` 各 1 处）

### 5.4 `OPS_LOG_I` / `OPS_LOG_E_IF` → 删除/原生判断

```cpp
// 源：
OPS_LOG_I(context->GetNodeName(), "Enter Compressor inferDataType impl.");
OPS_LOG_E_IF((apiRet != GRAPH_SUCCESS), context, return ge::GRAPH_FAILED, "Context get input shape failed");
// 目标：
// （日志行删除）
if (apiRet != GRAPH_SUCCESS) { return ge::GRAPH_FAILED; }
```

---

## 六、`op_kernel/compressor.cpp` 改动（arch22 单路径）

源仓库 kernel 入口按架构条件编译：

```cpp
// 源：
#if (__CCE_AICORE__ == 220)
#include "arch22/compressor_kernel.h"
#include "arch22/compressor_kernel_perf.h"
#else
#include "arch35/compressor_kernel.h"
#include "arch35/compressor_kernel_full_load.h"
#endif
...
#if (__CCE_AICORE__ == 220)
    // dispatch CompressorKernelPerf / CompressorKernel
#else
    // dispatch CompressorKernelFullLoad / CompressorKernel
#endif
```

**目标只保留 arch22 分支**：

```cpp
// 目标：
#include "compressor_kernel.h"
#include "compressor_kernel_perf.h"
...
if constexpr (static_cast<TEMPLATE_ID>(TemplateId) == TEMPLATE_ID::PERF) {
    INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorKernelPerf, ...);
} else {
    INVOKE_COMPRESSOR_GENERAL_OP_IMPL(CompressorKernel, ...);
}
```

- 去掉 `#if/else` 架构分支
- include 从 `"arch22/xxx.h"` → `"xxx.h"`（已展平）
- 模板 dispatch 去掉 arch35 的 `CompressorKernelFullLoad` 路径

---

## 七、`op_host/CMakeLists.txt` 重写（对齐 sgl-kernel-npu 规范）

源仓库的 `op_host/CMakeLists.txt` 使用 cann-ops-transformer 构建体系（`add_op_to_compiled_list`、`add_modules_sources`、`add_tiling_modules`、`target_sources(${OPHOST_NAME}_tiling_obj ...)` 等），目标改为 sgl-kernel-npu 现有算子（`laser_attention`、`block_sparse_attention`）一致的写法：

```cmake
add_ops_compile_options(
        OP_NAME Compressor
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

set(compressor_depends compressor PARENT_SCOPE)
target_sources(op_host_aclnn PRIVATE
        compressor_def.cpp
)

target_sources(optiling PRIVATE
        compressor_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        compressor_tiling.cpp
    )
endif ()

target_sources(opsproto PRIVATE
        compressor_proto.cpp
)

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)
```

关键差异：
- 去掉 cann-ops-transformer 特有的 `add_op_to_compiled_list()` / `add_modules_sources()` / `add_tiling_modules()`
- `-fpermissive` 标志对齐现有算子
- 8 空格缩进对齐仓库风格

---

## 八、`csrc/attentions/build/build_ops.sh` 修改

算子编译列表加入 `compressor`：

```bash
# 源：
-n 'laser_attention;block_sparse_attention;sparse_block_estimate'
# 目标：
-n 'laser_attention;block_sparse_attention;sparse_block_estimate;compressor'
```

---

## 九、许可证头统一（全部 18 个源文件）

源仓库文件头为 **CANN Open Software License Agreement v2.0**，目标仓库统一为 **Mulan PSL v2**（与 `laser_attention` 等现有算子一致）：

```cpp
// 源：
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software... CANN Open Software License Agreement Version 2.0 ...
 */

// 目标：
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * ...
 */
```

---

## 十、兼容改动小结

| # | 文件 | 改动类型 | 说明 |
|---|---|---|---|
| 1 | `compressor_tiling.h` | 结构适配 | include 路径 `arch22/` → 展平 |
| 2 | `compressor_tiling.cpp` | **CANN 9.0.0 兼容** | 删 `err/ops_err.h`；新增 4 个兼容宏；`is_same_v` → `is_same<...>::value` |
| 3 | `compressor_proto.cpp` | **CANN 9.0.0 兼容** | 删 `log/ops_log.h`；替换 `OPS_LOG_E_IF_NULL` / `OP_CHECK_IF` / `OPS_REPORT_VECTOR_INNER_ERR` / `OPS_LOG_I` / `OPS_LOG_E_IF` 为原生 C++ |
| 4 | `compressor.cpp` | 结构适配 | 只保留 arch22 分支；去 `#if/else`；去 arch35 include |
| 5 | `op_host/CMakeLists.txt` | 构建适配 | 重写对齐 sgl-kernel-npu 现有算子 |
| 6 | `build_ops.sh` | 构建适配 | 算子列表加 `compressor` |
| 7 | 全部 18 个源文件 | 许可证 | CANN → Mulan PSL v2 |

**未改动的部分**：`compressor_def.cpp`、`compressor_proto.cpp` 的输入输出定义、kernel 12 个 `.h` 文件的算法逻辑，均与源仓库一致（仅许可证头变了）。

---

## 十一、构建与验证

- **单算子编译**：`csrc/attentions/build/build_ops.sh` → `SUCCESS`，产出 `aie_ascendc` vendor 包（`libcust_opapi.so` 含 `aclnnCompressor` / `aclnnCompressorGetWorkspaceSize`，A3/A2 kernel `.o`）
- **运行时验证**：SGLang 容器 `zyl-sgl`，`import attentions`（自动设 `ASCEND_CUSTOM_OPP_PATH` 指向 `aie_ascendc`）+ `import custom_ops` 后，`torch.ops.custom.compressor` 可用
- **测试**：`test_compressor.py` 9/9 通过（C128A/C4A/C4Li × 分页/ring、BF16/FP16、空 batch、CPU golden、state_cache 完整性）

> 注意：`cache_mode=2`（ring）是迁移后 `aie_ascendc` vendor 才支持的特性，镜像预装的旧 `custom_transformer` vendor 只支持 `cache_mode=1`。
