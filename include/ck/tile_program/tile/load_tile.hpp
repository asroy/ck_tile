// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_window.hpp"
#include "ck/tile_program/tile/static_distributed_tensor.hpp"

namespace ck {
namespace tile_program {

// detail used by tile-programming APIs(), not supposed to be used directly
namespace detail {

// "Y dimension": Y dimensions inside TileWindowWithStaticDistribution
// input:
//   y_slice_origin: starting slice origin of Y dimension
//   y_slice_lengths: slice lengths of Y dimensionr
// output:
//   A StaticBuffer holding slice of thread data, and data layout is hardcoded to be in the order of
//   [Y0, Y1, Y2, ...]
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t HintNumAccessPerCoord_,
          typename YIndex,
          index_t... YSliceLengths>
__device__ auto load_sliced_thread_data_from_tile_window(
    TileWindowWithStaticDistribution<BottomTensorView_,
                                     WindowLengths_,
                                     TileDistribution_,
                                     HintNumAccessPerCoord_>& tile_window,
    const YIndex& ys_slice_origin,
    Sequence<YSliceLengths...> y_slice_lengths)
{
    return tile_window.LoadSlicedThreadData(ys_slice_origin, y_slice_lengths);
}

} // namespace detail

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          index_t HintNumAccessPerCoord_>
__device__ auto load_tile(TileWindowWithStaticDistribution<BottomTensorView_,
                                                           WindowLengths_,
                                                           TileDistribution_,
                                                           HintNumAccessPerCoord_>& tile_window)
{
    using DataType         = remove_cvref_t<typename BottomTensorView_::DataType>;
    using BottomTensorView = remove_cvref_t<BottomTensorView_>;
    using WindowLengths    = remove_cvref_t<WindowLengths_>;
    using TileDstr         = remove_cvref_t<TileDistribution_>;
    using TileWindow       = TileWindowWithStaticDistribution<BottomTensorView,
                                                        WindowLengths,
                                                        TileDstr,
                                                        HintNumAccessPerCoord_>;

    static_assert(is_known_at_compile_time<WindowLengths>::value,
                  "wrong! lengths should be static");
    static_assert(TileWindow::HasStaticTileDistribution(), "wrong!");

    constexpr auto tile_dstr = TileDstr{};

    constexpr index_t NDimY = tile_dstr.GetYs2DDescriptor().GetNumOfDimension();

    auto dstr_tensor = make_static_distributed_tensor<DataType>(tile_dstr);

    dstr_tensor.GetThreadBuffer() = detail::load_sliced_thread_data_from_tile_window(
        tile_window, MultiIndex<NDimY>{0}, to_sequence(tile_dstr.GetYs2DDescriptor().GetLengths()));

    return dstr_tensor;
}

} // namespace tile_program
} // namespace ck
