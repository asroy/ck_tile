// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"

namespace ck {
namespace tile_program {
namespace block {

// Default policy for BlockGemmARegBSmemCRegV1
// Default policy class should not be templated, put template on member functions instead
struct BlockGemmARegBSmemCRegV1M16N16K16Policy
{
    template <typename Problem>
    __host__ __device__ static constexpr auto GetWarpGemmMWarpNWarp()
    {
        using namespace ck::tile_program::warp;

        return make_tuple(WarpGemmMfmaF16F16F32M16N16K16TransposedCDistribution{}, 4, 1);
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
