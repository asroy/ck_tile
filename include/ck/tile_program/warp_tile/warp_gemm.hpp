// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

#include "ck/tile_program/warp_tile/warp_gemm_impl.hpp"
#include "ck/tile_program/warp_tile/warp_gemm_attribute_mfma.hpp"
#include "ck/tile_program/warp_tile/warp_gemm_attribute_mfma_impl.hpp"

namespace ck {
namespace tile_program {
namespace warp {

using WarpGemmMfmaF16F16F32M32N32K8 =
    WarpGemmImpl<WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplF16F16F32M32N32K8>>;

using WarpGemmMfmaF16F16F32M16N16K16 =
    WarpGemmImpl<WarpGemmAtrributeMfma<WarpGemmAttributeMfmaImplF16F16F32M16N16K16>>;

using WarpGemmMfmaF16F16F32M32N32K16 =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateK<WarpGemmAttributeMfmaImplF16F16F32M32N32K8, 2>>;

using WarpGemmMfmaF16F16F32M16N16K32 =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateK<WarpGemmAttributeMfmaImplF16F16F32M16N16K16, 2>>;

using WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution = WarpGemmImpl<
    WarpGemmAtrributeMfmaTransposedCDistribution<WarpGemmAttributeMfmaImplF16F16F32M32N32K8>>;

using WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution = WarpGemmImpl<
    WarpGemmAtrributeMfmaTransposedCDistribution<WarpGemmAttributeMfmaImplF16F16F32M16N16K16>>;

using WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImplF16F16F32M32N32K8,
        2>>;

using WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution =
    WarpGemmImpl<WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution<
        WarpGemmAttributeMfmaImplF16F16F32M16N16K16,
        2>>;

template <typename AType,
          typename BType,
          typename CType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          bool TransposeC>
struct WarpGemmMfmaDispatcher;

// clang-format off
template<> struct WarpGemmMfmaDispatcher<ck::half_t, ck::half_t, float, 32, 32,  8, false> { using Type = WarpGemmMfmaF16F16F32M32N32K8; };
template<> struct WarpGemmMfmaDispatcher<ck::half_t, ck::half_t, float, 32, 32,  8, true> { using Type = WarpGemmMfmaF16F16F32M32N32K8TransposedCDistribution; };
template<> struct WarpGemmMfmaDispatcher<ck::half_t, ck::half_t, float, 32, 32, 16, false> { using Type = WarpGemmMfmaF16F16F32M32N32K16; };
template<> struct WarpGemmMfmaDispatcher<ck::half_t, ck::half_t, float, 32, 32, 16, true> { using Type = WarpGemmMfmaF16F16F32M32N32K16TransposedCDistribution; };
template<> struct WarpGemmMfmaDispatcher<ck::half_t, ck::half_t, float, 16, 16, 16, false> { using Type = WarpGemmMfmaF16F16F32M16N16K16; };
template<> struct WarpGemmMfmaDispatcher<ck::half_t, ck::half_t, float, 16, 16, 16, true> { using Type = WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution; };
template<> struct WarpGemmMfmaDispatcher<ck::half_t, ck::half_t, float, 16, 16, 32, false> { using Type = WarpGemmMfmaF16F16F32M16N16K32; };
template<> struct WarpGemmMfmaDispatcher<ck::half_t, ck::half_t, float, 16, 16, 32, true> { using Type = WarpGemmMfmaF16F16F32M16N16K32TransposedCDistribution; };
// clang-format on

template <typename AType,
          typename BType,
          typename CType,
          index_t MPerWave,
          index_t NPerWave,
          index_t KPerWave,
          bool TransposeC>
using WarpGemmMfma =
    typename WarpGemmMfmaDispatcher<AType, BType, CType, MPerWave, NPerWave, KPerWave, TransposeC>::
        Type;

} // namespace warp
} // namespace tile_program
} // namespace ck
