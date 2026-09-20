// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// The code snippet comes from [CANN].
//
// Copyright (c) [2025] [CANN]. All rights reserved.
//
// This file contains code from [CANN], which is released under
// the CANN Open Software License Agreement Version 2.0 (the "License")
// See the LICENSE file in the root directory of this source tree
// or at https://gitcode.com/cann/ops-nn/blob/master/LICENSE for details.

/*!
 * \file op_log.h
 * \brief Host-side logging for sgl-kernel-npu operators.
 *
 * Provides the `OP_LOGE` / `OP_LOGW` / `OP_LOGI` / `OP_LOGD` / `OP_CHECK`
 * error-reporting style of ops-transformer. Every message is printed to stderr
 * and, when the CANN alog headers are available, also recorded into the CANN
 * plog through `AlogRecord()`.
 *
 * Why AlogRecord: ops-transformer's `OP_LOGE` ultimately resolves to
 * `AlogRecord()` (via `D_OP_LOGE` -> `OpLogSub`), and `libascendalog.so` is what
 * sgl-kernel-npu already links (`ascendalog` in csrc/CMakeLists.txt). The
 * `AlogRecord`/`AlogCheckDebugLevel` declarations live in `<base/alog_pub.h>`,
 * which ships in the CANN package under `<arch>-linux/pkg_inc` (the build adds
 * that directory to the include path). The faithful in-repo reference is
 * csrc/deepep/ops/op_host/dfx_base.h and csrc/attentions/.../dfx_base.h.
 *
 * Note: the previous draft used the weak symbol `DlogRecord` (as
 * csrc/compressor/.../compressor_tiling.cpp does). That does NOT reach the plog:
 * `DlogRecord` belongs to libunified_dlog.so / libslog.so, which sgl-kernel-npu
 * does not link, so the weak symbol stays null and only stderr is written.
 *
 * Usage:
 *   #include "op_log.h"
 *   OP_LOGE("MyOp", "tensor x is nullptr, dim=%zu", dim);
 *   OP_CHECK(x == nullptr, OP_LOGE("MyOp", "x is nullptr"), return false);
 *
 * The first argument is the operator name; the output carries the level, the
 * module tag, the operator name and the source file:line so the failing
 * operator can be located at a glance.
 *
 * Debug/info are compiled out by default to avoid log flooding; define
 * SGL_KERNEL_ENABLE_DEBUG_LOG / SGL_KERNEL_ENABLE_INFO_LOG to turn them on.
 */

#ifndef SGL_KERNEL_NPU_OP_LOG_H
#define SGL_KERNEL_NPU_OP_LOG_H

#include <cstdint>
#include <cstdio>

// ---------------------------------------------------------------------------
// Optional CANN alog integration (writes into the CANN plog).
//
// If the alog header is not on the include path this header still compiles and
// simply logs to stderr only.
// ---------------------------------------------------------------------------
#if defined(__has_include)
#  if __has_include(<base/alog_pub.h>)
#    include <base/alog_pub.h>
#    define SGL_KERNEL_HAS_ALOG 1
#  elif __has_include(<alog_pub.h>)
#    include <alog_pub.h>
#    define SGL_KERNEL_HAS_ALOG 1
#  endif
#endif

#ifdef SGL_KERNEL_HAS_ALOG
// DLOG_DEBUG / DLOG_INFO / DLOG_WARN / DLOG_ERROR / DLOG_TYPE_DEBUG come from
// <base/alog_pub.h>.
#else
// alog unavailable: define the levels so the macros below still compile (the
// alog call itself is compiled out in that case).
#define DLOG_DEBUG 0
#define DLOG_INFO 1
#define DLOG_WARN 2
#define DLOG_ERROR 3
#endif

// Sub-module tag prepended to every log line.
#ifndef SGL_KERNEL_LOG_TAG
#define SGL_KERNEL_LOG_TAG "sgl-kernel-npu"
#endif

// alog module id: "OP" is the operator module id, the same value
// ops-transformer / dfx_base.h pass to AlogRecord.
#ifndef SGL_KERNEL_LOG_MODULE_ID
#define SGL_KERNEL_LOG_MODULE_ID OP
#endif

// Maximum length of a single formatted message (longer messages are truncated).
#ifndef SGL_KERNEL_LOG_MAX_LEN
#define SGL_KERNEL_LOG_MAX_LEN 4096
#endif

namespace sgl_kernel_log_detail {
// Basename of __FILE__, so the log carries "foo.cpp:123" instead of the full path.
inline const char *BaseName(const char *path)
{
    const char *base = path;
    for (const char *p = path; *p != '\0'; ++p) {
        if (*p == '/') {
            base = p + 1;
        }
    }
    return base;
}
}  // namespace sgl_kernel_log_detail

// Debug / info are compiled out by default to avoid log flooding.
#ifndef SGL_KERNEL_ENABLE_DEBUG_LOG
#define OP_LOGD(opName, ...)
#else
#define OP_LOGD(opName, ...) \
    SGL_OP_LOG_IMPL(DLOG_DEBUG, "[DEBUG]", opName, __VA_ARGS__)
#endif

#ifndef SGL_KERNEL_ENABLE_INFO_LOG
#define OP_LOGI(opName, ...)
#else
#define OP_LOGI(opName, ...) \
    SGL_OP_LOG_IMPL(DLOG_INFO, "[INFO]", opName, __VA_ARGS__)
#endif

// Warn / error always print (stderr, and the plog when alog is available).
#define OP_LOGW(opName, ...) \
    SGL_OP_LOG_IMPL(DLOG_WARN, "[WARN]", opName, __VA_ARGS__)

#define OP_LOGE(opName, ...) \
    SGL_OP_LOG_IMPL(DLOG_ERROR, "[ERROR]", opName, __VA_ARGS__)

// Print to stderr only, without a plog record.
#define OP_LOGE_WITHOUT_REPORT(opName, ...) \
    SGL_OP_LOG_STDERR_IMPL("[ERROR]", opName, __VA_ARGS__)

// Conditional log-and-act: log when `cond` is true, then run `expr`.
#define OP_CHECK(cond, logFunc, expr) \
    do {                              \
        if (cond) {                   \
            logFunc;                  \
            expr;                     \
        }                             \
    } while (0)

#define OP_CHECK_IF(cond, logFunc, expr) OP_CHECK(cond, logFunc, expr)

// ---------------------------------------------------------------------------
// Internal implementation helpers (not for direct use).
// ---------------------------------------------------------------------------
#define SGL_OP_LOG_STDERR_IMPL(tag, opName, ...)                                           \
    do {                                                                                   \
        char sgl_log_buf[SGL_KERNEL_LOG_MAX_LEN];                                          \
        snprintf(sgl_log_buf, sizeof(sgl_log_buf), __VA_ARGS__);                           \
        fprintf(stderr, "%s[%s][%s][%s:%d] %s\n", tag, SGL_KERNEL_LOG_TAG, (opName),       \
                sgl_kernel_log_detail::BaseName(__FILE__), __LINE__, sgl_log_buf);         \
    } while (0)

#ifdef SGL_KERNEL_HAS_ALOG
#define SGL_OP_LOG_IMPL(level, tag, opName, ...)                                            \
    do {                                                                                    \
        char sgl_log_buf[SGL_KERNEL_LOG_MAX_LEN];                                           \
        snprintf(sgl_log_buf, sizeof(sgl_log_buf), __VA_ARGS__);                            \
        fprintf(stderr, "%s[%s][%s][%s:%d] %s\n", tag, SGL_KERNEL_LOG_TAG, (opName),        \
                sgl_kernel_log_detail::BaseName(__FILE__), __LINE__, sgl_log_buf);          \
        if (AlogCheckDebugLevel(static_cast<int>(SGL_KERNEL_LOG_MODULE_ID), (level)) == 1) { \
            AlogRecord(static_cast<int>(SGL_KERNEL_LOG_MODULE_ID), DLOG_TYPE_DEBUG, (level), \
                       "%s[%s][%s][%s:%d] %s", tag, SGL_KERNEL_LOG_TAG, (opName),            \
                       sgl_kernel_log_detail::BaseName(__FILE__), __LINE__, sgl_log_buf);    \
        }                                                                                   \
    } while (0)
#else
#define SGL_OP_LOG_IMPL(level, tag, opName, ...) SGL_OP_LOG_STDERR_IMPL(tag, opName, __VA_ARGS__)
#endif

#endif  // SGL_KERNEL_NPU_OP_LOG_H
