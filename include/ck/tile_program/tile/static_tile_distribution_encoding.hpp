
// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

namespace ck {
namespace tile_program {

template <typename RsLengths_,    // Sequence<...>
          typename HsLengthss_,   // Tuple<Sequence<...>, ...>
          typename Ps2RHssMajor_, // Tuple<Sequence<...>, ...>
          typename Ps2RHssMinor_, // Tuple<Sequence<...>, ...>
          typename Ys2RHsMajor_,  // Sequence<...>
          typename Ys2RHsMinor_   // Sequence<...>
          >
struct StaticTileDistributionEncoding
{
    using RsLengths    = remove_cvref_t<RsLengths_>;
    using HsLengthss   = remove_cvref_t<HsLengthss_>;
    using Ps2RHssMajor = remove_cvref_t<Ps2RHssMajor_>;
    using Ps2RHssMinor = remove_cvref_t<Ps2RHssMinor_>;
    using Ys2RHsMajor  = remove_cvref_t<Ys2RHsMajor_>;
    using Ys2RHsMinor  = remove_cvref_t<Ys2RHsMinor_>;

    static_assert(Ps2RHssMajor::Size() == Ps2RHssMinor::Size(), "wrong!");
    static_assert(Ys2RHsMajor::Size() == Ys2RHsMinor::Size(), "wrong!");

    static constexpr index_t NDimX = HsLengthss::Size();
    static constexpr index_t NDimP = Ps2RHssMajor::Size();
    static constexpr index_t NDimY = Ys2RHsMajor::Size();

    static constexpr auto rs_lengths_       = RsLengths{};
    static constexpr auto hs_lengthss_      = HsLengthss{};
    static constexpr auto ps_to_rhss_major_ = Ps2RHssMajor{};
    static constexpr auto ps_to_rhss_minor_ = Ps2RHssMinor{};
    static constexpr auto ys_to_rhs_major_  = Ys2RHsMajor{};
    static constexpr auto ys_to_rhs_minor_  = Ys2RHsMinor{};

    // redundant but useful info
    struct Detail
    {
        static constexpr index_t ndim_rh_major_    = NDimX + 1;
        static constexpr index_t ndim_range_major_ = NDimX;

        // Array
        static constexpr auto ndims_rhs_minor_ = generate_array(
            [](auto i) {
                if constexpr(i.value == 0)
                {
                    return rs_lengths_.Size();
                }
                else
                {
                    return hs_lengthss_[i - Number<1>{}].Size();
                }
            },
            Number<NDimX + 1>{});

        //
        static constexpr index_t max_ndim_rh_minor_ =
            container_reduce(ndims_rhs_minor_, math::maximize<index_t>{}, 0);

        // Array of Array
        static constexpr auto rhs_major_minor_to_ys_ = [] {
            Array<Array<index_t, max_ndim_rh_minor_>, NDimX + 1> rhs_major_minor_to_ys_tmp{{-1}};

            static_for<0, NDimY, 1>{}([&](auto i) {
                constexpr index_t rh_major = ys_to_rhs_major_[i];
                constexpr index_t rh_minor = ys_to_rhs_minor_[i];

                rhs_major_minor_to_ys_tmp(rh_major)(rh_minor) = i;
            });

            return rhs_major_minor_to_ys_tmp;
        }();

        // Array
        static constexpr auto ndims_range_minor_ = [] {
            Array<index_t, NDimX> ndims_range_minor{0};

            for(index_t i = 0; i < NDimY; i++)
            {
                const index_t range_major = ys_to_rhs_major_[i] - 1;

                ndims_range_minor(range_major)++;
            }

            return ndims_range_minor;
        }();

        //
        static constexpr index_t max_ndim_range_minor_ =
            container_reduce(ndims_range_minor_, math::maximize<index_t>{}, 0);

        // Array
        static constexpr auto rhs_major_minor_to_range_minor_ = [] {
            Array<Array<index_t, max_ndim_rh_minor_>, ndim_rh_major_>
                rhs_major_minor_to_range_minor{{-1}};

            static_for<0, ndim_rh_major_, 1>{}([&](auto rh_major) {
                constexpr index_t ndim_rh_minor = ndims_rhs_minor_[rh_major];

                index_t cnt_ndim_range_minor = 0;

                static_for<0, ndim_rh_minor, 1>{}([&](auto rh_minor) {
                    constexpr index_t idim_y = rhs_major_minor_to_ys_[rh_major][rh_minor];

                    if(idim_y >= 0)
                    {
                        rhs_major_minor_to_range_minor(rh_major)(rh_minor) = cnt_ndim_range_minor;

                        cnt_ndim_range_minor++;
                    }
                });
            });

            return rhs_major_minor_to_range_minor;
        }();

        // Array
        static constexpr auto ys_to_range_major_ =
            generate_array([](auto i) { return ys_to_rhs_major_[i] - 1; }, Number<NDimY>{});

        //
        static constexpr auto ys_to_range_minor_ = generate_array(
            [](auto i) {
                return rhs_major_minor_to_range_minor_[ys_to_rhs_major_[i]][ys_to_rhs_minor_[i]];
            },
            Number<NDimY>{});

        //
        static constexpr auto distributed_ranges_lengthss_ = [] {
            Array<Array<index_t, max_ndim_range_minor_>, ndim_range_major_>
                distributed_ranges_lengthss{{-1}};

            static_for<0, NDimY, 1>{}([&](auto i) {
                const index_t rh_major = ys_to_rhs_major_[i];
                const index_t rh_minor = ys_to_rhs_minor_[i];

                const index_t h_length = hs_lengthss_[Number<rh_major - 1>{}][rh_minor];

                const index_t range_major = rh_major - 1;
                const index_t range_minor = rhs_major_minor_to_range_minor_[rh_major][rh_minor];

                distributed_ranges_lengthss(range_major)(range_minor) = h_length;
            });

            return distributed_ranges_lengthss;
        }();

        static constexpr auto ndims_distributed_ranges_minor_ = [] {
            Array<index_t, ndim_range_major_> ndims_distributed_ranges_minor{0};

            static_for<0, NDimY, 1>{}([&](auto i) {
                const index_t range_major = ys_to_rhs_major_[i] - 1;

                ndims_distributed_ranges_minor(range_major)++;
            });

            return ndims_distributed_ranges_minor;
        }();

        __host__ __device__ void Print() const
        {
            printf("StaticTileDistributionEncoding::Detail{");
            //
            printf("ndim_rh_major_: ");
            print(ndim_rh_major_);
            printf(", ");
            //
            printf("ndim_range_major_: ");
            print(ndim_range_major_);
            printf(", ");
            //
            printf("ndims_rhs_minor_: ");
            print(ndims_rhs_minor_);
            printf(", ");
            //
            printf("ndim_rh_major_: ");
            print(ndim_rh_major_);
            printf(", ");
            //
            printf("max_ndim_rh_minor_: ");
            print(max_ndim_rh_minor_);
            printf(", ");
            //
            printf("rhs_major_minor_to_ys_: ");
            print(rhs_major_minor_to_ys_);
            printf(", ");
            //
            printf("ndims_range_minor_: ");
            print(ndims_range_minor_);
            printf(", ");
            //
            printf("max_ndim_range_minor_: ");
            print(max_ndim_range_minor_);
            printf(", ");
            //
            printf("ys_to_range_major_: ");
            print(ys_to_range_major_);
            printf(", ");
            //
            printf("ys_to_range_minor_: ");
            print(ys_to_range_minor_);
            printf(", ");
            //
            printf("distributed_ranges_lengthss_: ");
            print(distributed_ranges_lengthss_);
            printf(", ");
            //
            printf("ndims_distributed_ranges_minor_: ");
            print(ndims_distributed_ranges_minor_);
            //
            printf("}");
        }
    };

    __host__ __device__ void Print() const
    {
        printf("StaticTileDistributionEncoding{");
        //
        printf("NDimX: %d, NDimP: %d, NDimY: %d, ", NDimX, NDimP, NDimY);
        //
        printf("rs_lengths_: ");
        print(rs_lengths_);
        printf(", ");
        //
        printf("hs_lengthss_: ");
        print(hs_lengthss_);
        printf(", ");
        //
        printf("ps_to_rhss_major_: ");
        print(ps_to_rhss_major_);
        printf(", ");
        //
        printf("ps_to_rhss_minor_: ");
        print(ps_to_rhss_minor_);
        printf(", ");
        //
        printf("ys_to_rhs_major_: ");
        print(ys_to_rhs_major_);
        printf(", ");
        //
        printf("ys_to_rhs_minor_: ");
        print(ys_to_rhs_minor_);
        printf(", ");
        //
        printf("Detail: ");
        print(Detail{});
        //
        printf("}");
    }
};

} // namespace tile_program
} // namespace ck
