// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tile_program/block_tensor_distribution.hpp"

namespace ck {
namespace tile_program {
namespace block {

// FIXME:
template <typename CDataType, typename ATile, typename BTile>
__host__ __device__ auto block_tile_gemm(const ATile& /* a_block_tile */,
                                         const BTile& /* b_block_tile */)
{
#if 1
    // decide A/B/C-block-distribution, A/B-wave-distribution, MRepeat, NRepeat, etc
    constexpr auto a_block_distr = make_static_block_tensor_distribution(
        make_tuple(Sequence<2, 2, 32, 1>{}, Sequence<4, 2, 4>{}),
        Sequence<0, 1>{},
        Sequence<1, 1>{},
        Sequence<0, 1>{},
        Sequence<3, 3>{},
        Sequence<0, 1>{},
        Sequence<0, 1, 0, 0, 1>{},
        Sequence<0, 0, 2, 4, 4>{});

    // construct A/B-block-window from A/B-block-distribution
    //
    // construct several A/B/C-wave-tensor
    //
    // zero out C-wave-tensor
    //
    // for loop:
    //   load from A/B-block-window into A/B-wave-tensor
    //   wave_gemm
    //
    // construct C-block-tensor from C-block-distribution
    //
    // copy C-wave-tensor into C-block-tensor
    //
    // return C-block-tensor
#else
    constexpr auto c_block_distr = make_static_block_tensor_distribution(
        make_tuple(Sequence<2, 2, 4, 2, 4>{}, Sequence<2, 2, 32, 1>{}),
        Sequence<0, 1>{},
        Sequence<1, 1>{},
        Sequence<0, 1>{},
        Sequence<3, 3>{},
        Sequence<0, 1>{},
        Sequence<0, 1, 0, 0, 1>{},
        Sequence<0, 0, 2, 4, 4>{});

    return make_static_block_distributed_tensor<CDataType>(c_block_distr);
#endif
}

// FIXME:
template <typename CTile, typename ATile, typename BTile>
__host__ __device__ void block_tile_gemm(CTile& /* c_block_tile */,
                                         const ATile& /* a_block_tile */,
                                         const BTile& /* b_block_tile */)
{
}

} // namespace block
} // namespace tile_program
} // namespace ck
