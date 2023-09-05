// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {
namespace tile_program {

// FIXME: host dummy function for tile program
template <typename BottomTensorView,
          typename ThreadBufferOffsets,
          typename DataType,
          typename TileDistribution>
__host__ decltype(auto)
store_tile(const TileWindowWithCoordinates<BottomTensorView, ThreadBufferOffsets>& tile_window,
           const StaticDistributedTensor<DataType, TileDistribution>&)
{
    return tile_window;
}

template <typename BottomTensorView,
          typename ThreadBufferOffsets,
          typename DataType_,
          typename TileDistribution>
__device__ decltype(auto)
store_tile(const TileWindowWithCoordinates<BottomTensorView, ThreadBufferOffsets>& tile_window,
           const StaticDistributedTensor<DataType_, TileDistribution>& dstr_tensor)
{
    using DataType = remove_cvref_t<typename BottomTensorView::DataType>;
    static_assert(is_same_v<remove_cvref_t<DataType_>, DataType>, "wrong!");

    constexpr auto NumAccess       = std::decay_t<decltype(tile_window)>::NumAccess;
    constexpr auto ScalarPerVector = std::decay_t<decltype(tile_window)>::ScalarPerVector;

    constexpr ThreadBufferOffsets thread_buffer_offsets{};

    // loop over thread tensor space [y0, y1, ...]
    static_for<0, NumAccess, 1>{}([&](auto iAccess) {
        // read from distributed tensor
        using vector_type_t = vector_type_maker_t<DataType, ScalarPerVector>;

        vector_type_t vec;

        static_for<0, ScalarPerVector, 1>{}([&](auto iScalar) {
            constexpr index_t offset = thread_buffer_offsets[iAccess][iScalar];

            vec.template AsType<DataType>()(iScalar) =
                dstr_tensor.GetThreadBuffer().template At<offset>();
        });

        using vector_t = typename vector_type_t::type;

        const vector_t vec_value = vec.template AsType<vector_t>().template At<0>();

        // write into bottom tensor
        tile_window.GetBottomTensorView().template SetVectorizedElements<vector_t>(
            tile_window.coordinates_[iAccess], vec_value);
    });

    return tile_window;
}

} // namespace tile_program
} // namespace ck
