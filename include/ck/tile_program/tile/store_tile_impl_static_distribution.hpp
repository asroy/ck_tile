// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_space_filling_curve.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_window_impl_coordinates.hpp"

namespace ck {
namespace tile_program {

// FIXME: host dummy function for tile program
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename DataType_>
__host__ decltype(auto)
store_tile(TileWindowWithStaticDistribution<BottomTensorView_, WindowLengths_, TileDistribution_>&
               tile_window,
           const StaticDistributedTensor<DataType_, TileDistribution_>&)
{
    return tile_window;
}

template <typename BottomTensorView_,
          typename WindowLengths_,
          typename TileDistribution_,
          typename DataType_>
__device__ auto
store_tile(TileWindowWithStaticDistribution<BottomTensorView_, WindowLengths_, TileDistribution_>&
               tile_window,
           const StaticDistributedTensor<DataType_, TileDistribution_>& dstr_tensor)
{
    using DataType         = remove_cvref_t<typename BottomTensorView_::DataType>;
    using BottomTensorView = remove_cvref_t<BottomTensorView_>;
    using WindowLengths    = remove_cvref_t<WindowLengths_>;
    using TileDstr         = remove_cvref_t<TileDistribution_>;
    using TileWindow = TileWindowWithStaticDistribution<BottomTensorView, WindowLengths, TileDstr>;

    static_assert(is_same_v<remove_cvref_t<DataType_>, DataType>, "wrong!");
    static_assert(TileWindow::HasStaticTileDistribution(), "wrong!");

    constexpr auto tile_dstr = TileDstr{};

    constexpr auto thread_tensor_lengths_ys =
        to_sequence(tile_dstr.GetYs2DDescriptor().GetLengths());

    constexpr index_t NDimP = TileDstr::GetNumOfDimensionP();
    constexpr index_t NDimY = TileDstr::GetNumOfDimensionY();

    constexpr auto tmp = []() {
        const auto [ys_vector_lengths, ys_vector_strides] =
            TileWindow::GetWindowAdaptorYsSafeVectorLengthStrides();

        index_t VectorDimY      = 0;
        index_t ScalarPerVector = 1;

        for(index_t i = 0; i < NDimY; ++i)
        {
            if(ys_vector_strides[i] == 1 && ys_vector_lengths[i] > ScalarPerVector)
            {
                ScalarPerVector = ys_vector_lengths[i];
                VectorDimY      = i;
            }
        }

        return make_tuple(VectorDimY, ScalarPerVector);
    }();

    constexpr index_t VectorDimY      = tmp.template At<0>();
    constexpr index_t ScalarPerVector = tmp.template At<1>();

    // FIXME:
    using DimAccessOrder = typename arithmetic_sequence_gen<0, NDimY, 1>::type;

    constexpr auto scalars_per_access_arr = generate_array(
        [&](auto i) { return (i == VectorDimY) ? ScalarPerVector : 1; }, Number<NDimY>{});

    constexpr auto scalars_per_access = TO_SEQUENCE(scalars_per_access_arr, NDimY);

    using vector_type_t = vector_type_maker_t<DataType, ScalarPerVector>;
    using vector_t      = typename vector_type_t::type;

    using SFC_Ys = SpaceFillingCurve<decltype(thread_tensor_lengths_ys),
                                     DimAccessOrder,
                                     decltype(scalars_per_access)>;

    constexpr index_t NumAccess = SFC_Ys::GetNumOfAccess();

    static_assert(NumAccess > 0, "wrong! NumAccess should be larger than 0");

    constexpr auto thread_buffer_offsets = generate_tuple(
        [&](auto iAccess) {
            constexpr auto idx_ys_start = SFC_Ys::GetIndex(iAccess);

            return generate_tuple(
                [&](auto iScalar) {
                    constexpr auto idx_ys = generate_array(
                        [&](auto j) {
                            return j == VectorDimY ? (idx_ys_start[j] + iScalar) : idx_ys_start[j];
                        },
                        Number<NDimY>{});

                    constexpr index_t offset =
                        tile_dstr.GetYs2DDescriptor().CalculateOffset(idx_ys);
                    return Number<offset>{};
                },
                Number<ScalarPerVector>{});
        },
        Number<NumAccess>{});

    TileWindowWithCoordinates<BottomTensorView_, decltype(thread_buffer_offsets)>
        converted_tile_window(tile_window.GetBottomTensorView());

    // loop over thread tensor space [y0, y1, ...]
    static_for<0, NumAccess, 1>{}([&](auto iAccess) {
        vector_type_t vec;

        static_for<0, ScalarPerVector, 1>{}([&](auto iScalar) {
            constexpr index_t offset = thread_buffer_offsets[iAccess][iScalar];

            vec.template AsType<DataType>()(iScalar) =
                dstr_tensor.GetThreadBuffer().template At<offset>();
        });

        const vector_t vec_value = vec.template AsType<vector_t>().template At<0>();

        // write into bottom tensor
        tile_window.GetBottomTensorView().template SetVectorizedElements<vector_t>(
            converted_tile_window.SetBottomTensorThreadCoordinate(
                iAccess, tile_window.GetBottomTensorThreadCoordinate()),
            vec_value);

        // move thread coordinate
        if constexpr(iAccess.value != NumAccess - 1)
        {
            constexpr auto idx_diff_ys = SFC_Ys::GetForwardStep(iAccess);

            constexpr auto idx_diff_ps_ys = container_concat(Array<index_t, NDimP>{0}, idx_diff_ys);

            tile_window.MoveWindowAdaptorAndBottomTensorThreadCoordinate(idx_diff_ps_ys);
        }
    });

    // move thread coordinate back to origin
    {
        constexpr auto idx_diff_ys = SFC_Ys::GetStepBetween(Number<NumAccess - 1>{}, Number<0>{});

        constexpr auto idx_diff_ps_ys = container_concat(Array<index_t, NDimP>{0}, idx_diff_ys);

        tile_window.MoveWindowAdaptorAndBottomTensorThreadCoordinate(idx_diff_ps_ys);
    }

    return converted_tile_window;
}

} // namespace tile_program
} // namespace ck
