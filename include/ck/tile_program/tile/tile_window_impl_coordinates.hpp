// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_adaptor_coordinate.hpp"

namespace ck {
namespace tile_program {

template <typename BottomTensorView_, typename ThreadBufferOffsets_>
struct TileWindowWithCoordinates
{
    using ThreadBufferOffsets =
        remove_cvref_t<ThreadBufferOffsets_>; // [NumAccess x ScalarPerVector] tuple

    static constexpr index_t NumAccess = ThreadBufferOffsets::Size();
    static_assert(0 < NumAccess);

    static constexpr index_t ScalarPerVector = tuple_element_t<0, ThreadBufferOffsets>::Size();
    static_assert(0 < ScalarPerVector);

    using BottomTensorView = remove_reference_t<BottomTensorView_>;

    using BottomTensorDesc = typename BottomTensorView::TensorDesc;

    using DataType = typename BottomTensorView::DataType;

    static constexpr index_t NDimBottomTensor = BottomTensorDesc::GetNumOfDimension();

    using BottomTensorIndex = Array<index_t, NDimBottomTensor>;

    using BottomTensorCoord =
        decltype(make_tensor_coordinate(BottomTensorDesc{}, BottomTensorIndex{}));

    __host__ __device__ constexpr TileWindowWithCoordinates() = default;

    __host__
        __device__ constexpr TileWindowWithCoordinates(const BottomTensorView& bottom_tensor_view)
        : bottom_tensor_view_{bottom_tensor_view}
    {
    }

    __host__ __device__ static constexpr index_t GetNumOfDimension() { return NDimBottomTensor; }

    __host__ __device__ constexpr auto GetBottomTensorView() const { return bottom_tensor_view_; }

    __host__ __device__ constexpr decltype(auto)
    GetBottomTensorThreadCoordinate(index_t iAccess) const
    {
        return thread_coordinates_[iAccess];
    }

    __host__ __device__ constexpr decltype(auto)
    SetBottomTensorThreadCoordinate(index_t iAccess, const BottomTensorCoord& coordinate)
    {
        return thread_coordinates_(iAccess) = coordinate;
    }

    private:
    // this is the bottom tensor view
    // [x0', x1', ...] ==> [offset]
    BottomTensorView bottom_tensor_view_;

    Array<BottomTensorCoord, NumAccess> thread_coordinates_;
};

} // namespace tile_program
} // namespace ck
