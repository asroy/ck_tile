// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {
namespace tile_program {

template <index_t kKDramLoadScalarPerVector_,
          index_t kVDramLoadScalarPerVector_,
          index_t kKSmemLoadScalarPerVector_,
          index_t kVSmemLoadScalarPerVector_,
          bool kM0NeedPadding_ /* padding for seqlen_q */,
          bool kN0K1NeedPadding_ /* padding for seqlen_k */,
          bool kSupportsBias_>
struct TileFmhaTraits
{
    static constexpr index_t kKDramLoadScalarPerVector = kKDramLoadScalarPerVector_;
    static constexpr index_t kVDramLoadScalarPerVector = kVDramLoadScalarPerVector_;
    static constexpr index_t kKSmemLoadScalarPerVector = kKSmemLoadScalarPerVector_;
    static constexpr index_t kVSmemLoadScalarPerVector = kVSmemLoadScalarPerVector_;
    static constexpr bool kM0NeedPadding               = kM0NeedPadding_;
    static constexpr bool kN0K1NeedPadding             = kN0K1NeedPadding_;
    static constexpr bool kSupportsBias                = kSupportsBias_;
};

} // namespace tile_program
} // namespace ck
