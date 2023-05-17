// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

namespace ck {
namespace tile_program {
namespace block {

namespace detail {

template <index_t NDimMax>
__host__ __device__ constexpr auto make_sequential_index(index_t ibegin, index_t iend)
{
    Array<index_t, NDimMax> arr{0};

    for(index_t i = 0; i < iend - ibegin; ++i)
    {
        arr(i) = ibegin + i;
    }

    return arr;
}

// this returns a constexpr encoding of BlockTensorDistribution
template <typename... XsUnMergeUpLengthss, // Tuple<Sequence<...>, ...>
          index_t... DimsWid2XsMajor,
          index_t... DimsWid2XsMinor,
          index_t... DimsLid2XsMajor,
          index_t... DimsLid2XsMinor,
          index_t... DimsYs2XsMajor,
          index_t... DimsYs2XsMinor,
          index_t... YsOrder>
__host__ __device__ constexpr auto make_block_tensor_distribution_encoding(
    //
    Tuple<XsUnMergeUpLengthss...>,
    //
    Sequence<DimsWid2XsMajor...>,
    Sequence<DimsWid2XsMinor...>,
    //
    Sequence<DimsLid2XsMajor...>,
    Sequence<DimsLid2XsMinor...>,
    //
    Sequence<DimsYs2XsMajor...>,
    Sequence<DimsYs2XsMinor...>,
    //
    Sequence<YsOrder...>)
{
    STATIC_ASSERT(sizeof...(DimsWid2XsMajor) == sizeof...(DimsWid2XsMinor) &&
                      sizeof...(DimsLid2XsMajor) == sizeof...(DimsLid2XsMinor) &&
                      sizeof...(DimsYs2XsMajor) == sizeof...(DimsYs2XsMinor) &&
                      sizeof...(DimsYs2XsMajor) == sizeof...(YsOrder),
                  "");

    constexpr index_t kMaxNumTransforms = 10;
    constexpr index_t kMaxMetaDataSize  = 128;
    constexpr index_t kMaxNumDim        = 10;

    using Name     = IndexTransformEnum;
    using MetaData = MetaDataBuffer<kMaxMetaDataSize>;
    using NumDim   = index_t;
    using Dims     = Array<index_t, kMaxNumDim>;
    using Lengths  = Array<index_t, kMaxNumDim>;

    // Adaptor: [wid, lid, y0, y1, ...] to [x0, x1, ...]
    // Adaptor: [y0, y1, ...] to [did]
    constexpr index_t ndim_x_major = sizeof...(XsUnMergeUpLengthss);

    // Dim Ids: [idim_x_major, idim_x_minor] to [idim_hidden]
    Array<Array<index_t, kMaxNumDim>, ndim_x_major> dims_x_major_x_minor_to_hidden_ids;
    Array<Array<index_t, kMaxNumDim>, ndim_x_major> dims_x_major_x_minor_to_hidden_lengths;

    auto trans = Array<Tuple<Name, MetaData, NumDim, Dims, NumDim, Dims>, kMaxNumTransforms>{};

    index_t num_tran       = 0;
    index_t hidden_dim_cnt = ndim_x_major;

    // these are the unmerge transforms
    static_for<0, ndim_x_major, 1>{}([&trans,
                                      &num_tran,
                                      &hidden_dim_cnt,
                                      &dims_x_major_x_minor_to_hidden_ids,
                                      &dims_x_major_x_minor_to_hidden_lengths](auto idim_x_major) {
        constexpr auto x_minor_lengths = type_pack_element<idim_x_major, XsUnMergeUpLengthss...>{};

        constexpr index_t ndim_x_minor = x_minor_lengths.Size();

        trans(num_tran++) = {
            IndexTransformEnum::UnMerge,
            MetaData{to_array<index_t, ndim_x_minor>(x_minor_lengths)},
            NumDim{1},
            Dims{idim_x_major},
            NumDim{ndim_x_minor},
            make_sequential_index<kMaxNumDim>(hidden_dim_cnt, hidden_dim_cnt + ndim_x_minor)};

        for(index_t i = 0; i < ndim_x_minor; ++i)
        {
            dims_x_major_x_minor_to_hidden_ids(idim_x_major)(i)     = hidden_dim_cnt;
            dims_x_major_x_minor_to_hidden_lengths(idim_x_major)(i) = x_minor_lengths[i];

            hidden_dim_cnt++;
        }
    });

    // transform: wid
    index_t hidden_dim_id_wid = hidden_dim_cnt++;

    {
        constexpr index_t ndim_low          = sizeof...(DimsWid2XsMajor);
        constexpr auto dims_wid_to_xs_major = Sequence<DimsWid2XsMajor...>{};
        constexpr auto dims_wid_to_xs_minor = Sequence<DimsWid2XsMinor...>{};

        Dims low_dims;
        Lengths low_lengths;

        for(index_t i = 0; i < ndim_low; ++i)
        {
            index_t x_major = dims_wid_to_xs_major[i];
            index_t x_minor = dims_wid_to_xs_minor[i];
            low_dims(i)     = dims_x_major_x_minor_to_hidden_ids[x_major][x_minor];
            low_lengths(i)  = dims_x_major_x_minor_to_hidden_lengths[x_major][x_minor];
        }

        trans(num_tran++) = {IndexTransformEnum::Merge,
                             MetaData{to_array<index_t, ndim_low>(low_lengths)},
                             NumDim{ndim_low},
                             low_dims,
                             NumDim{1},
                             Dims{hidden_dim_id_wid}};
    }

    // transform: lid
    index_t hidden_dim_id_lid = hidden_dim_cnt++;

    {
        constexpr index_t ndim_low          = sizeof...(DimsLid2XsMinor);
        constexpr auto dims_lid_to_xs_major = Sequence<DimsLid2XsMajor...>{};
        constexpr auto dims_lid_to_xs_minor = Sequence<DimsLid2XsMinor...>{};

        Dims low_dims;
        Lengths low_lengths;

        for(index_t i = 0; i < ndim_low; ++i)
        {
            index_t x_major = dims_lid_to_xs_major[i];
            index_t x_minor = dims_lid_to_xs_minor[i];
            low_dims(i)     = dims_x_major_x_minor_to_hidden_ids[x_major][x_minor];
            low_lengths(i)  = dims_x_major_x_minor_to_hidden_lengths[x_major][x_minor];
        }

        trans(num_tran++) = {IndexTransformEnum::Merge,
                             MetaData{to_array<index_t, ndim_low>(low_lengths)},
                             NumDim{ndim_low},
                             low_dims,
                             NumDim{1},
                             Dims{hidden_dim_id_lid}};
    }

    // bottom dims [x0, x1, x2, ...]
    constexpr index_t ndim_bottom = ndim_x_major;

    constexpr auto bottom_dim_ids = make_sequential_index<kMaxNumDim>(0, ndim_bottom);

    // top dims [wid, lid, y0, y1, ...]
    constexpr auto dims_ys_to_xs_major = Sequence<DimsYs2XsMajor...>{};
    constexpr auto dims_ys_to_xs_minor = Sequence<DimsYs2XsMinor...>{};

    constexpr index_t ndim_y   = sizeof...(DimsYs2XsMajor);
    constexpr index_t ndim_top = 2 + ndim_y;

    auto top_dim_ids = Dims{hidden_dim_id_wid, hidden_dim_id_lid};

    {
        for(index_t i = 0; i < ndim_y; ++i)
        {
            index_t x_major    = dims_ys_to_xs_major[i];
            index_t x_minor    = dims_ys_to_xs_minor[i];
            top_dim_ids(2 + i) = dims_x_major_x_minor_to_hidden_ids[x_major][x_minor];
        }
    }

    // adaptor: [y0, y1, ...] to [did]
    Lengths y_lengths;
    index_t did_length = 1;

    for(index_t i = 0; i < ndim_y; ++i)
    {
        constexpr auto ys_order = Sequence<YsOrder...>{};

        index_t x_major  = dims_ys_to_xs_major[ys_order[i]];
        index_t x_minor  = dims_ys_to_xs_minor[ys_order[i]];
        index_t y_length = dims_x_major_x_minor_to_hidden_lengths[x_major][x_minor];
        y_lengths(i)     = y_length;
        did_length *= y_length;
    }

    auto tran = make_tuple(IndexTransformEnum::UnMerge,
                           MetaData{to_array<index_t, ndim_y>(y_lengths)},
                           NumDim{1},
                           Dims{0},
                           NumDim{ndim_y},
                           Dims{1 + YsOrder...});

    return make_tuple(
        make_tuple(trans, num_tran, bottom_dim_ids, ndim_bottom, top_dim_ids, ndim_top),
        make_tuple(make_tuple(tran), 1, Dims{0}, 1, Dims{1 + YsOrder...}, ndim_y),
        did_length);
}

} // namespace detail

template <typename WidLidYs2XsAdaptor_, typename Ys2DidDescriptor_>
struct BlockTensorDistribution
{
    using WidLidYs2XsAdaptor = remove_cvref_t<WidLidYs2XsAdaptor_>;
    using Ys2DidDescriptor   = remove_cvref_t<Ys2DidDescriptor_>;

    WidLidYs2XsAdaptor wid_lid_ys_to_xs_;
    Ys2DidDescriptor ys_to_did_;

    __host__ __device__ constexpr const auto& GetWidLidYs2XsAdaptor() const
    {
        return wid_lid_ys_to_xs_;
    }

    __host__ __device__ constexpr const auto& GetYs2DidDescriptor() const { return ys_to_did_; }

    __host__ __device__ constexpr const auto& GetWidLidYsLengths() const
    {
        return wid_lid_ys_to_xs_;
    }

    __device__ auto CalculateThreadWidLidYsOrigin() const
    {
        constexpr index_t ndim = WidLidYs2XsAdaptor::GetNumOfTopDimension();

        Array<index_t, ndim> idx{get_warp_id(), get_lane_id()};

        for(index_t i = 2; i < ndim; ++i)
        {
            idx(i) = 0;
        }

        return idx;
    }

    __host__ __device__ static constexpr bool IsKnownAtCompileTime()
    {
        return WidLidYs2XsAdaptor::IsKnownAtCompileTime() &&
               Ys2DidDescriptor::IsKnownAtCompileTime();
    }

    __host__ __device__ void Print() const
    {
        printf("{");
        printf("BlockTensorDistribution, ");
        wid_lid_ys_to_xs_.Print();
        ys_to_did_.Print();
        printf("}");
    }
};

// this returns a constexpr BlockTensorDistribution
template <typename... XsUnMergeUpLengthss, // Tuple<Sequence<...>, ...>
          index_t... DimsWid2XsMajor,
          index_t... DimsWid2XsMinor,
          index_t... DimsLid2XsMajor,
          index_t... DimsLid2XsMinor,
          index_t... DimsYs2XsMajor,
          index_t... DimsYs2XsMinor,
          index_t... YsOrder>
__host__ __device__ constexpr auto make_block_tensor_distribution(
    //
    Tuple<XsUnMergeUpLengthss...>,
    //
    Sequence<DimsWid2XsMajor...>,
    Sequence<DimsWid2XsMinor...>,
    //
    Sequence<DimsLid2XsMajor...>,
    Sequence<DimsLid2XsMinor...>,
    //
    Sequence<DimsYs2XsMajor...>,
    Sequence<DimsYs2XsMinor...>,
    //
    Sequence<YsOrder...>)
{
    constexpr auto encode =
        detail::make_block_tensor_distribution_encoding(Tuple<XsUnMergeUpLengthss...>{},
                                                        Sequence<DimsWid2XsMajor...>{},
                                                        Sequence<DimsWid2XsMinor...>{},
                                                        Sequence<DimsLid2XsMajor...>{},
                                                        Sequence<DimsLid2XsMinor...>{},
                                                        Sequence<DimsYs2XsMajor...>{},
                                                        Sequence<DimsYs2XsMinor...>{},
                                                        Sequence<YsOrder...>{});

    constexpr auto encoded_wid_lid_ys_to_xs_adaptor = encode.template At<0>();
    constexpr auto encoded_ys_to_did_adaptor        = encode.template At<1>();
    constexpr index_t did_length                    = encode.template At<2>();

    constexpr auto wid_lid_ys_to_xs_adaptor =
        CONSTRUCT_TENSOR_ADAPTOR_FROM_ENCODING(encoded_wid_lid_ys_to_xs_adaptor);

    constexpr auto ys_to_did_adaptor =
        CONSTRUCT_TENSOR_ADAPTOR_FROM_ENCODING(encoded_ys_to_did_adaptor);

    constexpr auto ys_to_did_descriptor =
        make_tensor_descriptor_from_adaptor(ys_to_did_adaptor, did_length);

    return BlockTensorDistribution<remove_cvref_t<decltype(wid_lid_ys_to_xs_adaptor)>,
                                   remove_cvref_t<decltype(ys_to_did_descriptor)>>{
        wid_lid_ys_to_xs_adaptor, ys_to_did_descriptor};
}

// this returns a static BlockTensorDistribution
template <typename... XsUnMergeUpLengthss, // Tuple<Sequence<...>, ...>
          index_t... DimsWid2XsMajor,
          index_t... DimsWid2XsMinor,
          index_t... DimsLid2XsMajor,
          index_t... DimsLid2XsMinor,
          index_t... DimsYs2XsMajor,
          index_t... DimsYs2XsMinor,
          index_t... YsOrder>
__host__ __device__ constexpr auto make_static_block_tensor_distribution(
    //
    Tuple<XsUnMergeUpLengthss...>,
    //
    Sequence<DimsWid2XsMajor...>,
    Sequence<DimsWid2XsMinor...>,
    //
    Sequence<DimsLid2XsMajor...>,
    Sequence<DimsLid2XsMinor...>,
    //
    Sequence<DimsYs2XsMajor...>,
    Sequence<DimsYs2XsMinor...>,
    //
    Sequence<YsOrder...>)
{
    constexpr auto encode =
        detail::make_block_tensor_distribution_encoding(Tuple<XsUnMergeUpLengthss...>{},
                                                        Sequence<DimsWid2XsMajor...>{},
                                                        Sequence<DimsWid2XsMinor...>{},
                                                        Sequence<DimsLid2XsMajor...>{},
                                                        Sequence<DimsLid2XsMinor...>{},
                                                        Sequence<DimsYs2XsMajor...>{},
                                                        Sequence<DimsYs2XsMinor...>{},
                                                        Sequence<YsOrder...>{});

    constexpr auto encoded_wid_lid_ys_to_xs_adaptor = encode.template At<0>();
    constexpr auto encoded_ys_to_did_adaptor        = encode.template At<1>();
    constexpr index_t did_length                    = encode.template At<2>();

    constexpr auto wid_lid_ys_to_xs_adaptor =
        CONSTRUCT_STATIC_TENSOR_ADAPTOR_FROM_ENCODING(encoded_wid_lid_ys_to_xs_adaptor);

    constexpr auto ys_to_did_adaptor =
        CONSTRUCT_STATIC_TENSOR_ADAPTOR_FROM_ENCODING(encoded_ys_to_did_adaptor);

    constexpr auto ys_to_did_descriptor =
        make_tensor_descriptor_from_adaptor(ys_to_did_adaptor, Number<did_length>{});

    return BlockTensorDistribution<remove_cvref_t<decltype(wid_lid_ys_to_xs_adaptor)>,
                                   remove_cvref_t<decltype(ys_to_did_descriptor)>>{
        wid_lid_ys_to_xs_adaptor, ys_to_did_descriptor};
}

} // namespace block
} // namespace tile_program
} // namespace ck