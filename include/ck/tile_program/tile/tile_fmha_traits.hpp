// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {
namespace tile_program {

template <bool kPadSeqLenQ_ /* padding for seqlen_q */,
          bool kPadSeqLenK_ /* padding for seqlen_k */,
          bool kPadHeadDimQV_ /* paddding for hdim_q / hdim_v */,
          bool kHasBias_,
          bool kStoreLSE_,
          index_t kBlockPerCu_ = 2 /* hint to occupancy */>
struct TileFmhaTraits
{
    static constexpr bool kPadSeqLenQ    = kPadSeqLenQ_;
    static constexpr bool kPadSeqLenK    = kPadSeqLenK_;
    static constexpr bool kPadHeadDimQV  = kPadHeadDimQV_;
    static constexpr bool kHasBias       = kHasBias_;
    static constexpr bool kStoreLSE      = kStoreLSE_;
    static constexpr index_t kBlockPerCu = kBlockPerCu_;
};

} // namespace tile_program
} // namespace ck
