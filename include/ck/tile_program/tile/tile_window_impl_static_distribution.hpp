// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"
#include "ck/tensor_description/tensor_adaptor_coordinate.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/static_tile_distribution_helper.hpp"

namespace ck {
namespace tile_program {
template <typename BottomTensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t HintNumCoords_ = 1>
struct TileWindowWithStaticDistribution
{
    using BottomTensorView = remove_reference_t<BottomTensorView_>;
    using WindowLengths    = remove_cvref_t<WindowLengths_>;
    using TileDstr         = remove_cvref_t<StaticTileDistribution_>;

    using WindowAdaptor    = typename TileDstr::PsYs2XsAdaptor;
    using BottomTensorDesc = typename BottomTensorView::TensorDesc;

    using DataType = typename BottomTensorView::DataType;

    static constexpr index_t NDimWindowAdaptorTop = WindowAdaptor::GetNumOfTopDimension();
    static constexpr index_t NDimBottomTensor     = BottomTensorDesc::GetNumOfDimension();

    // TODO: check WindowLengths and StaticTileDistribution are consistent

    static_assert(is_known_at_compile_time<WindowLengths>::value,
                  "wrong! lengths should be static");
    static_assert(TileDstr::IsStatic(), "wrong!");

    static_assert(NDimBottomTensor == WindowAdaptor::GetNumOfBottomDimension(),
                  "wrong! inconsistent # of diemsnions");

    using AdaptorTopIndex   = Array<index_t, NDimWindowAdaptorTop>;
    using BottomTensorIndex = Array<index_t, NDimBottomTensor>;

    using WindowAdaptorCoord =
        decltype(make_tensor_adaptor_coordinate(WindowAdaptor{}, AdaptorTopIndex{}));

    using BottomTensorCoord =
        decltype(make_tensor_coordinate(BottomTensorDesc{}, BottomTensorIndex{}));

    struct TraitsBase
    {
        protected:
        static constexpr index_t NDimP = TileDstr::GetNumOfDimensionP();
        static constexpr index_t NDimY = TileDstr::GetNumOfDimensionY();

        using DataType = remove_cvref_t<typename BottomTensorView_::DataType>;

        template <index_t ScalarPerVector>
        using vector_type_t = vector_type_maker_t<DataType, ScalarPerVector>;

        template <index_t VectorDimY, index_t ScalarPerVector>
        static constexpr auto GetScalarsPerAccess()
        {
            constexpr auto scalars_per_access_arr = generate_array(
                [&](auto i) { return (i == VectorDimY) ? ScalarPerVector : 1; }, Number<NDimY>{});

            /// TODO: add non-automatic storage argument support to macro TO_SEQUENCE()
            constexpr auto NDimY_ = NDimY;

            return TO_SEQUENCE(scalars_per_access_arr, NDimY_);
        }

        template <typename TensorLengths, index_t VectorDimY, index_t ScalarPerVector>
        using SpaceFillingCurve =
            ck::SpaceFillingCurve<TensorLengths,
                                  typename arithmetic_sequence_gen<0, NDimY, 1>::type,
                                  decltype(GetScalarsPerAccess<VectorDimY, ScalarPerVector>())>;
    };

    template <typename YIndex, index_t... YSliceLengths>
    struct LoadTraits : TraitsBase
    {
        using TraitsBase::NDimP;
        using TraitsBase::NDimY;

        static_assert(NDimY == YIndex::Size() && NDimY == sizeof...(YSliceLengths),
                      "wrong! inconsistent # of dimension");

        using typename TraitsBase::DataType;

        using ThreadBuffer =
            StaticBuffer<AddressSpaceEnum::Vgpr,
                         DataType,
                         container_reduce(Sequence<YSliceLengths...>{}, math::multiplies{}, 1),
                         true>;

        private:
        static constexpr auto GetVectorDimYScalarPerVector()
        {
            auto [ys_vector_lengths, ys_vector_strides] =
                TileWindowWithStaticDistribution::GetWindowAdaptorYsSafeVectorLengthStrides();

            index_t VectorDimY_      = 0;
            index_t ScalarPerVector_ = 1;

            for(index_t i = 0; i < NDimY; ++i)
            {
                if(ys_vector_strides[i] == 1 && ys_vector_lengths[i] > ScalarPerVector_)
                {
                    ScalarPerVector_ =
                        math::gcd(ys_vector_lengths[i], Sequence<YSliceLengths...>{}[i]);
                    VectorDimY_ = i;
                }
            }

            return make_tuple(VectorDimY_, ScalarPerVector_);
        }

        public:
        static constexpr index_t VectorDimY      = GetVectorDimYScalarPerVector().template At<0>();
        static constexpr index_t ScalarPerVector = GetVectorDimYScalarPerVector().template At<1>();

        using vector_type_t = typename TraitsBase::template vector_type_t<ScalarPerVector>;
        using vector_t      = typename vector_type_t::type;

        using SpaceFillingCurve = typename TraitsBase::
            template SpaceFillingCurve<Sequence<YSliceLengths...>, VectorDimY, ScalarPerVector>;

        static constexpr index_t NumAccess = SpaceFillingCurve::GetNumOfAccess();

        static_assert(0 < NumAccess, "Wrong! NumAccess should be larger than 0");
    };

    struct StoreTraits : TraitsBase
    {
        using TraitsBase::NDimP;
        using TraitsBase::NDimY;

        using typename TraitsBase::DataType;

        private:
        static constexpr auto GetVectorDimYScalarPerVector()
        {
            const auto [ys_vector_lengths, ys_vector_strides] =
                TileWindowWithStaticDistribution::GetWindowAdaptorYsSafeVectorLengthStrides();

            index_t VectorDimY_      = 0;
            index_t ScalarPerVector_ = 1;

            for(index_t i = 0; i < NDimY; ++i)
            {
                if(ys_vector_strides[i] == 1 && ys_vector_lengths[i] > ScalarPerVector_)
                {
                    ScalarPerVector_ = ys_vector_lengths[i];
                    VectorDimY_      = i;
                }
            }

            return make_tuple(VectorDimY_, ScalarPerVector_);
        }

        public:
        static constexpr index_t VectorDimY      = GetVectorDimYScalarPerVector().template At<0>();
        static constexpr index_t ScalarPerVector = GetVectorDimYScalarPerVector().template At<1>();

        using vector_type_t = typename TraitsBase::template vector_type_t<ScalarPerVector>;
        using vector_t      = typename vector_type_t::type;

        private:
        static constexpr auto GetSpaceFillingCurve()
        {
            constexpr auto tile_dstr = TileDstr{};

            constexpr auto thread_tensor_lengths_ys =
                to_sequence(tile_dstr.GetYs2DDescriptor().GetLengths());

            return
                typename TraitsBase::template SpaceFillingCurve<decltype(thread_tensor_lengths_ys),
                                                                VectorDimY,
                                                                ScalarPerVector>{};
        }

        public:
        using SpaceFillingCurve = decltype(GetSpaceFillingCurve());

        static constexpr index_t NumAccess = SpaceFillingCurve::GetNumOfAccess();

        static_assert(0 < NumAccess, "Wrong! NumAccess should be larger than 0");
    };

    static constexpr index_t NumCoords =
        (0 < HintNumCoords_ ? math::min(StoreTraits::NumAccess, HintNumCoords_)
                            : StoreTraits::NumAccess);

    static constexpr index_t NumAccessPerCoord = StoreTraits::NumAccess / NumCoords;

    __device__ constexpr TileWindowWithStaticDistribution() = default;

    __device__ constexpr TileWindowWithStaticDistribution(
        const BottomTensorView& bottom_tensor_view,
        const WindowLengths& window_lengths,
        const BottomTensorIndex& window_origin,
        const TileDstr& tile_distribution)
        : bottom_tensor_view_{bottom_tensor_view},
          window_lengths_{window_lengths},
          window_origin_{window_origin},
          bottom_tensor_thread_coord_{},
          tile_dstr_{tile_distribution},
          window_adaptor_thread_coord_{}
    {
#if 0 // debug
      // only support warp-tile and block-tile
        static_assert(TileDstr::NDimP == 1 or TileDstr::NDimP == 2, "wrong!");

        if constexpr(TileDstr::NDimP == 1)
        {
            window_adaptor_thread_coord_ = make_tensor_adaptor_coordinate(
                tile_distribution.GetPsYs2XsAdaptor(), AdaptorTopIndex{get_lane_id(), 0});
        }
        else if constexpr(TileDstr::NDimP == 2)
        {
            window_adaptor_thread_coord_ =
                make_tensor_adaptor_coordinate(tile_distribution.GetPsYs2XsAdaptor(),
                                               AdaptorTopIndex{get_warp_id(), get_lane_id(), 0});
        }
#elif 0
        // only support warp-tile and block-tile
        static_assert(TileDstr::NDimP == 1 or TileDstr::NDimP == 2, "wrong!");

        if constexpr(TileDstr::NDimP == 1)
        {
            window_adaptor_thread_coord_ = make_tensor_adaptor_coordinate(
                tile_distribution.GetPsYs2XsAdaptor(),
                container_concat(Array<index_t, 1>{get_lane_id()},
                                 Array<index_t, TileDstr::NDimY>{0}));
        }
        else if constexpr(TileDstr::NDimP == 2)
        {
            window_adaptor_thread_coord_ = make_tensor_adaptor_coordinate(
                tile_distribution.GetPsYs2XsAdaptor(),
                container_concat(Array<index_t, 2>{get_warp_id(), get_lane_id()},
                                 Array<index_t, TileDstr::NDimY>{0}));
        }
#else
        window_adaptor_thread_coord_ = make_tensor_adaptor_coordinate(
            tile_distribution.GetPsYs2XsAdaptor(),
            container_concat(detail::get_partition_index(tile_distribution),
                             Array<index_t, TileDstr::NDimY>{0}));
#endif

        BottomTensorIndex bottom_tensor_thread_origin_idx;

        for(index_t i = 0; i < NDimBottomTensor; ++i)
        {
            bottom_tensor_thread_origin_idx(i) =
                window_origin[i] + window_adaptor_thread_coord_.GetBottomIndex()[i];
        }

        bottom_tensor_thread_coord_ = make_tensor_coordinate(
            bottom_tensor_view_.GetTensorDescriptor(), bottom_tensor_thread_origin_idx);

        // pre-compute NumCoords (WindowAdaptorCoord, BottomTensorCoord) bundles to speed up
        // future Store() calls (might allocate more registers)
        {
            using Traits = StoreTraits;

            static_assert(Traits::NumAccess % NumCoords == 0,
                          "# of access is not divisible by HintNumCoords_");

            using SFC_Ys = typename Traits::SpaceFillingCurve;

            auto window_adaptor_thread_coord = window_adaptor_thread_coord_;
            auto bottom_tensor_thread_coord  = bottom_tensor_thread_coord_;

            // loop over thread tensor space [y0, y1, ...]
            static_for<0, NumCoords, 1>{}([&](auto iCoord) {
                // store current thread coordinates
                pre_computed_coords_(iCoord) =
                    tie(window_adaptor_thread_coord, bottom_tensor_thread_coord);

                static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                    constexpr auto iAccess = Number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                    // move thread coordinate
                    if constexpr(iAccess != Traits::NumAccess - 1)
                    {
                        constexpr auto idx_diff_ys = SFC_Ys::GetForwardStep(iAccess);

                        constexpr auto idx_diff_ps_ys =
                            container_concat(Array<index_t, Traits::NDimP>{0}, idx_diff_ys);

                        MoveWindowAdaptorAndBottomTensorThreadCoordinate(
                            window_adaptor_thread_coord,
                            bottom_tensor_thread_coord,
                            idx_diff_ps_ys);
                    }
                });
            });
        }
    }

    __device__ static constexpr index_t GetNumOfDimension() { return NDimBottomTensor; }

    __device__ static constexpr bool HasStaticTileDistribution() { return TileDstr::IsStatic(); }

    __device__ constexpr auto GetWindowLengths() const { return window_lengths_; }

    __device__ constexpr auto GetTileDistribution() const { return tile_dstr_; }

    __device__ constexpr auto GetBottomTensorView() const { return bottom_tensor_view_; }

    __device__ constexpr auto GetWindowOrigin() const { return window_origin_; }

    __device__ constexpr auto GetBottomTensorThreadCoordinate() const
    {
        return bottom_tensor_thread_coord_;
    }

    // move thread's window adaptor coordiante
    // [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...]
    __device__ void MoveWindowAdaptorThreadCoordinate(const AdaptorTopIndex& idx_diff_adaptor)
    {
        move_tensor_adaptor_coordinate(
            tile_dstr_.GetPsYs2XsAdaptor(), window_adaptor_thread_coord_, idx_diff_adaptor);
    }

    // move thread's botom tensor coordiante
    // [x0', x1', ... ] ==> [offset]
    __device__ void MoveBottomTensorThreadCoordinate(const BottomTensorIndex& idx_diff_tensor)
    {
        move_tensor_coordinate(bottom_tensor_view_.GetTensorDescriptor(),
                               bottom_tensor_thread_coord_,
                               idx_diff_tensor);
    }

    // move thread's window adaptor coordinate and bottom tensor coordinate
    // [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...] ==> [x0', x1', ...] ==> [offset]
    __device__ void
    MoveWindowAdaptorAndBottomTensorThreadCoordinate(const AdaptorTopIndex& idx_diff_adaptor_top)
    {
        MoveWindowAdaptorAndBottomTensorThreadCoordinate(
            window_adaptor_thread_coord_, bottom_tensor_thread_coord_, idx_diff_adaptor_top);
    }

    // move thread's window adaptor coordinate and bottom tensor coordinate
    // [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...] ==> [x0', x1', ...] ==> [offset]
    __device__ void MoveWindowAdaptorAndBottomTensorThreadCoordinate(
        WindowAdaptorCoord& window_adaptor_thread_coord,
        BottomTensorCoord& bottom_tensor_thread_coord,
        const AdaptorTopIndex& idx_diff_adaptor_top) const
    {
        Array<index_t, NDimBottomTensor> idx_diff_adaptor_bottom;

        move_tensor_adaptor_coordinate(tile_dstr_.GetPsYs2XsAdaptor(),
                                       window_adaptor_thread_coord,
                                       idx_diff_adaptor_top,
                                       idx_diff_adaptor_bottom);

        move_tensor_coordinate(bottom_tensor_view_.GetTensorDescriptor(),
                               bottom_tensor_thread_coord,
                               idx_diff_adaptor_bottom);
    }

    // return vector dimension among [y0, y1, ...]
    __device__ static constexpr auto GetWindowAdaptorYsSafeVectorLengthStrides()
    {
        // bottom tensor top dimension vector lengths and strides
        const auto [bottom_tensor_top_dim_vector_lengths, bottom_tensor_top_dim_vector_strides] =
            BottomTensorDesc::GetTopDimensionSafeVectorLengthStrides();

        // window vector lengths/strides
        const auto window_adaptor_bottom_dim_vector_lengths = bottom_tensor_top_dim_vector_lengths;
        const auto window_adaptor_bottom_dim_vector_strides = bottom_tensor_top_dim_vector_strides;

        // window adaptor [p0, p1, ..., y0, y1, ...]
        Array<index_t, WindowAdaptor::GetNumOfHiddenDimension()> window_adaptor_vector_lengths{-1};
        Array<index_t, WindowAdaptor::GetNumOfHiddenDimension()> window_adaptor_vector_strides{-1};

        constexpr auto window_adaptor_bottom_dims = WindowAdaptor::GetBottomDimensionHiddenIds();

        set_container_subset(window_adaptor_vector_lengths,
                             window_adaptor_bottom_dims,
                             window_adaptor_bottom_dim_vector_lengths);
        set_container_subset(window_adaptor_vector_strides,
                             window_adaptor_bottom_dims,
                             window_adaptor_bottom_dim_vector_strides);

        const auto [window_adaptor_ps_ys_vector_lengths, window_adaptor_ps_ys_vector_strides] =
            WindowAdaptor{}.GetTopDimensionSafeVectorLengthStrides(window_adaptor_vector_lengths,
                                                                   window_adaptor_vector_strides);

        // [y0, y1, ...]
        constexpr auto y_dims = typename arithmetic_sequence_gen<TileDstr::GetNumOfDimensionP(),
                                                                 NDimWindowAdaptorTop,
                                                                 1>::type{};

        return make_tuple(get_container_subset(window_adaptor_ps_ys_vector_lengths, y_dims),
                          get_container_subset(window_adaptor_ps_ys_vector_strides, y_dims));
    }

    template <typename YIndex, index_t... YSliceLengths>
    __device__ auto LoadSlicedThreadData(const YIndex& ys_slice_origin, Sequence<YSliceLengths...>)
    {
        using Traits = LoadTraits<YIndex, YSliceLengths...>;

        constexpr index_t NDimP = Traits::NDimP;
        constexpr index_t NDimY = Traits::NDimY;

        typename Traits::ThreadBuffer thread_buf;

        constexpr index_t VectorDimY      = Traits::VectorDimY;
        constexpr index_t ScalarPerVector = Traits::ScalarPerVector;

        using vector_type_t = typename Traits::vector_type_t;
        using vector_t      = typename vector_type_t::type;

        using SFC_Ys = typename Traits::SpaceFillingCurve;

        constexpr index_t num_access = Traits::NumAccess;

        // move to slice origin
        const auto ps_ys_slice_origin = container_concat(Array<index_t, NDimP>{0}, ys_slice_origin);

        MoveWindowAdaptorAndBottomTensorThreadCoordinate(ps_ys_slice_origin);

        constexpr auto tile_dstr = TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, num_access, 1>{}([&](auto iAccess) {
            // read from bottom tensor
            const vector_t vec_value =
                GetBottomTensorView().template GetVectorizedElements<vector_t>(
                    GetBottomTensorThreadCoordinate());

            const vector_type_t vec{vec_value};

            // data index [y0, y1, ...]
            constexpr auto idx_ys_start = SFC_Ys::GetIndex(iAccess);

            // write into distributed tensor
            static_for<0, ScalarPerVector, 1>{}([&](auto j) {
                constexpr auto idx_ys = generate_array(
                    [&](auto jj) {
                        return jj == VectorDimY ? (idx_ys_start[jj] + j) : idx_ys_start[jj];
                    },
                    Number<NDimY>{});

                constexpr index_t d = tile_dstr.GetYs2DDescriptor().CalculateOffset(idx_ys);

                thread_buf.template At<d>() = vec.template AsType<typename Traits::DataType>()[j];
            });

            // move thread coordinate
            if constexpr(iAccess.value != num_access - 1)
            {
                constexpr auto idx_diff_ys = SFC_Ys::GetForwardStep(iAccess);

                constexpr auto idx_diff_ps_ys =
                    container_concat(Array<index_t, NDimP>{0}, idx_diff_ys);

                MoveWindowAdaptorAndBottomTensorThreadCoordinate(idx_diff_ps_ys);
            }
        });

        // move thread coordinate back to origin
        {
            constexpr auto idx_diff_ys =
                SFC_Ys::GetStepBetween(Number<num_access - 1>{}, Number<0>{});

            constexpr auto idx_diff_ps_ys = container_concat(Array<index_t, NDimP>{0}, idx_diff_ys);

            MoveWindowAdaptorAndBottomTensorThreadCoordinate(idx_diff_ps_ys);
        }

        // move back to origin
        MoveWindowAdaptorAndBottomTensorThreadCoordinate(MultiIndex<NDimP + NDimY>{0} -
                                                         ps_ys_slice_origin);

        return thread_buf;
    }

    template <typename DataType_>
    __device__ void Store(const StaticDistributedTensor<DataType_, TileDstr>& dstr_tensor) const
    {
        static_assert(is_same_v<DataType_, DataType>,
                      "Unmatched DataType of tile window & distributed tensor");

        using Traits = StoreTraits;

        using vector_type_t = typename Traits::vector_type_t;
        using vector_t      = typename vector_type_t::type;

        using SFC_Ys = typename Traits::SpaceFillingCurve;

        constexpr auto tile_dstr = TileDstr{};

        // loop over thread tensor space [y0, y1, ...]
        static_for<0, NumCoords, 1>{}([&](auto iCoord) {
            /// TODO: use structure binding (to be captured later) if compiled in C++20
            auto window_adaptor_thread_coord = pre_computed_coords_[iCoord].template At<0>();
            auto bottom_tensor_thread_coord  = pre_computed_coords_[iCoord].template At<1>();

            static_for<0, NumAccessPerCoord, 1>{}([&](auto iCoordAccess) {
                constexpr auto iAccess = Number<iCoord * NumAccessPerCoord + iCoordAccess>{};

                // data index [y0, y1, ...]
                constexpr auto idx_ys_start = SFC_Ys::GetIndex(iAccess);

                // read from distributed tensor
                vector_type_t vec;

                static_for<0, Traits::ScalarPerVector, 1>{}([&](auto j) {
                    constexpr auto idx_ys = generate_array(
                        [&](auto jj) {
                            return jj == Traits::VectorDimY ? (idx_ys_start[jj] + j)
                                                            : idx_ys_start[jj];
                        },
                        Number<Traits::NDimY>{});

                    constexpr index_t d = tile_dstr.GetYs2DDescriptor().CalculateOffset(idx_ys);

                    vec.template AsType<typename Traits::DataType>()(j) =
                        dstr_tensor.GetThreadBuffer().template At<d>();
                });

                const vector_t vec_value = vec.template AsType<vector_t>().template At<0>();

                // write into bottom tensor
                GetBottomTensorView().template SetVectorizedElements<vector_t>(
                    bottom_tensor_thread_coord, vec_value);

                // move thread coordinate
                if constexpr(iCoordAccess != (NumAccessPerCoord - 1))
                {
                    constexpr auto idx_diff_ys = SFC_Ys::GetForwardStep(iAccess);

                    constexpr auto idx_diff_ps_ys =
                        container_concat(Array<index_t, Traits::NDimP>{0}, idx_diff_ys);

                    MoveWindowAdaptorAndBottomTensorThreadCoordinate(
                        window_adaptor_thread_coord, bottom_tensor_thread_coord, idx_diff_ps_ys);
                }
            });
        });
    }

    // this is the bottom tensor view
    // [x0', x1', ...] ==> [offset]
    BottomTensorView bottom_tensor_view_;

    //
    WindowLengths window_lengths_;

    // origin ([x0', x1', ...]) of window on bottom tensor
    BottomTensorIndex window_origin_;

    /// TODO: remove bottom_tensor_thread_coord_ after
    /// LoadSlicedThreadData() share almost same implementation with Store()
    // per-thread coordinate for bottom tensor
    BottomTensorCoord bottom_tensor_thread_coord_;

    // Tile tensor distribution, which contains:
    //   1. adaptor for window: [p0, p1, ..., y0, y1, ...] ==> [x0, x1, ...]
    //   2. thread descriptor for thread tensor in register: [y0, y1, ...] ==> [d]
    TileDstr tile_dstr_;

    /// TODO: remove window_adaptor_thread_coord_ after
    /// LoadSlicedThreadData() share almost same implementation with Store()
    //    thread window coordinate
    WindowAdaptorCoord window_adaptor_thread_coord_;

    Array<Tuple<WindowAdaptorCoord, BottomTensorCoord>, NumCoords> pre_computed_coords_;
};

// TODO: use strategy
template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t HintNumCoords_ = 1>
__device__ constexpr auto
make_tile_window(const TensorView_& tensor_view,
                 const WindowLengths_& window_lengths,
                 const MultiIndex<TensorView_::GetNumOfDimension()>& origin,
                 const StaticTileDistribution_& tile_distribution,
                 Number<HintNumCoords_> = {})
{
    return TileWindowWithStaticDistribution<remove_cvref_t<TensorView_>,
                                            remove_cvref_t<WindowLengths_>,
                                            remove_cvref_t<StaticTileDistribution_>,
                                            HintNumCoords_>{
        tensor_view, window_lengths, origin, tile_distribution};
}

template <typename TensorView_,
          typename WindowLengths_,
          typename StaticTileDistribution_,
          index_t HintNumCoords_>
__device__ void move_tile_window(
    TileWindowWithStaticDistribution<TensorView_,
                                     WindowLengths_,
                                     StaticTileDistribution_,
                                     HintNumCoords_>& window,
    const MultiIndex<TileWindowWithStaticDistribution<TensorView_,
                                                      WindowLengths_,
                                                      StaticTileDistribution_,
                                                      HintNumCoords_>::GetNumOfDimension()>& step)
{
    window.window_origin_ += step;

    window.MoveBottomTensorThreadCoordinate(step);
}

} // namespace tile_program
} // namespace ck
