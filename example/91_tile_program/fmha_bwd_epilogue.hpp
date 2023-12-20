// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tile_program/tile/store_tile.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"

template <typename AccDataType_, typename KGradDataType_, typename VGradDataType_>
struct FmhaBwdEpilogueProblem
{
    using AccDataType   = ck::remove_cvref_t<AccDataType_>;
    using KGradDataType = ck::remove_cvref_t<KGradDataType_>;
    using VGradDataType = ck::remove_cvref_t<VGradDataType_>;
};

template <typename Problem_, typename Policy_ = void>
struct FmhaBwdEpilogue
{
    using Problem       = ck::remove_cvref_t<Problem_>;
    using AccDataType   = ck::remove_cvref_t<typename Problem::AccDataType>;
    using KGradDataType = ck::remove_cvref_t<typename Problem::KGradDataType>;
    using VGradDataType = ck::remove_cvref_t<typename Problem::VGradDataType>;

    __host__ __device__ static constexpr ck::index_t GetSmemSize() { return 0; }

    template <typename KGradDramWindowTmp,
              typename VGradDramWindowTmp,
              typename KGradAccTile,
              typename VGradAccTile>
    __device__ auto operator()(KGradDramWindowTmp& dk_dram_window_tmp,
                               VGradDramWindowTmp& dv_dram_window_tmp,
                               const KGradAccTile& dk_acc_tile,
                               const VGradAccTile& dv_acc_tile)
    {
        using namespace ck;
        using namespace ck::tile_program;

        const auto dk = tile_elementwise_in(type_convert<KGradDataType, AccDataType>, dk_acc_tile);
        store_tile(dk_dram_window_tmp, dk);

        const auto dv = tile_elementwise_in(type_convert<VGradDataType, AccDataType>, dv_acc_tile);
        store_tile(dv_dram_window_tmp, dv);
    }
};
