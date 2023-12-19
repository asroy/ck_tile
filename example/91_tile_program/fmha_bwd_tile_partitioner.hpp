// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tile_program/tile/store_tile.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"

template <typename BlockFmhaBwdShape_>
struct FmhaBwdTilePartitioner
{
    using BlockFmhaBwdShape = ck::remove_cvref_t<BlockFmhaBwdShape_>;

    static constexpr ck::index_t kN0 = BlockFmhaBwdShape::kN0;

    __host__ static constexpr auto
    GridSize(ck::index_t batch_size_, ck::index_t nhead_, ck::index_t seqlen_k_)
    {
        // TODO: this may need tuning
        return dim3(seqlen_k_ / kN0, batch_size_, nhead_);
    }

    __device__ auto operator()(ck::index_t /*seqlen_k*/)
    {
        using namespace ck;

        const index_t i_block = blockIdx.x;
        const index_t i_batch = blockIdx.y;
        const index_t i_nhead = blockIdx.z;

        return ck::make_tuple(i_block, i_nhead, i_batch);
    }
};

template <ck::index_t kBlockSize>
struct FmhaBwdOGradDotOTilePartitioner
{
    __host__ static constexpr auto
    GridSize(ck::index_t batch_size_, ck::index_t nhead_, ck::index_t seqlen_q_)
    {
        // TODO: this may need tuning
        return dim3(seqlen_q_ / kBlockSize, batch_size_, nhead_);
    }

    __device__ auto operator()(ck::index_t /*seqlen_q*/)
    {
        using namespace ck;

        const index_t i_block = blockIdx.x;
        const index_t i_batch = blockIdx.y;
        const index_t i_nhead = blockIdx.z;

        return ck::make_tuple(i_block, i_nhead, i_batch);
    }
};
