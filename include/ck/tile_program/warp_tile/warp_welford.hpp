// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"

#include "ck/tile_program/tile/static_distributed_tensor.hpp"
#include "ck/tile_program/tile/static_tile_distribution_helper.hpp"
#include "ck/tile_program/tile/distributed_tile_sweep.hpp"

namespace ck {
namespace tile_program {
namespace warp {

template <typename ComputeDataType_, bool GetActualVariance = true>
struct WarpMergeWelford
{
    using ComputeDataType = remove_cvref_t<ComputeDataType_>;

    template <typename T>
    __device__ static inline void
    Merge(T& mean_a, T& var_a, int& count_a, T mean_b, T var_b, int count_b)
    {
        int count            = count_a + count_b;
        T count_b_over_count = count == 0 ? type_convert<T>(0) : type_convert<T>(count_b) / count;
        T delta              = mean_b - mean_a;
        mean_a += delta * count_b_over_count;
        var_a += var_b + delta * delta * count_a * count_b_over_count;
        count_a = count;
    }

    template <typename MeanDistributedTensor_, typename VarDistributedTensor_>
    __device__ void
    operator()(MeanDistributedTensor_& mean_tensor, VarDistributedTensor_& var_tensor, int& count)
    {
        using Dstr             = typename MeanDistributedTensor_::StaticTileDistribution;
        using DstrEncode       = typename Dstr::DstrEncode;
        using DstrEncodeDetail = typename DstrEncode::Detail;

        static_assert(is_same_v<Dstr, typename VarDistributedTensor_::StaticTileDistribution>,
                      "wrong!");

        constexpr index_t NDimP = Dstr::GetNumOfDimensionP();
        constexpr index_t NDimR = Dstr::GetNumOfDimensionR();

        constexpr index_t idim_p_lane = NDimP - 1;

        const auto ps_idx = make_array<index_t>(get_warp_id(), get_lane_id());
        const auto rs_idx = mean_tensor.GetTileDistribution().CalculateRsIndexFromPsIndex(ps_idx);

        constexpr index_t thread_buf_size = MeanDistributedTensor_::GetThreadBufferSize();
        static_assert(thread_buf_size == VarDistributedTensor_::GetThreadBufferSize());

        const int original_count = count;

        // loop over thread data
        static_for<0, thread_buf_size, 1>{}([&](auto i) {
            auto v_local_mean  = mean_tensor.GetThreadBuffer()[i];
            auto v_local_var   = var_tensor.GetThreadBuffer()[i];
            auto v_local_count = original_count;

            // cross-lane reduce for replication
            // only reduce on R dimension correspond to lane
            // (lane id maps to this R dimension)
            static_for<0, NDimR, 1>{}([&](auto idim_r) {
                // FIXME: nasty to use does_p_own_r_
                if constexpr(DstrEncodeDetail::does_p_own_r_[idim_p_lane][idim_r])
                {
                    constexpr index_t r_length = DstrEncode::rs_lengths_[idim_r];

                    constexpr index_t lid_over_rid_derivative =
                        DstrEncodeDetail::ps_over_rs_derivative_[idim_p_lane][idim_r];

                    static_assert(math::is_power_of_two_integer(r_length),
                                  "wrong! only support power of 2 reduction");

                    constexpr index_t nstage = math::integer_log2_floor(r_length);

                    // reduction sweep forward
                    static_for<0, nstage, 1>{}([&](auto istage) {
                        constexpr index_t lid_delta =
                            lid_over_rid_derivative * (1 << (nstage - istage - 1));

                        // pull data from remote lane
                        const auto v_remote_mean  = warp_shuffle_down(v_local_mean, lid_delta);
                        const auto v_remote_var   = warp_shuffle_down(v_local_var, lid_delta);
                        const auto v_remote_count = warp_shuffle_down(v_local_count, lid_delta);

                        // welford merge
                        Merge(v_local_mean,
                              v_local_var,
                              v_local_count,
                              v_remote_mean,
                              v_remote_var,
                              v_remote_count);
                    });
                }
            });

            // cross-lane broadcast for replication
            // only broadcast on R dimension correspond to lane
            // (lane id maps to this R dimension)
            static_for<0, NDimR, 1>{}([&](auto idim_r) {
                // FIXME: nasty to use does_p_own_r_
                if constexpr(DstrEncodeDetail::does_p_own_r_[idim_p_lane][idim_r])
                {
                    const index_t r_id = rs_idx[idim_r];

                    constexpr index_t r_length = DstrEncode::rs_lengths_[idim_r];

                    constexpr index_t lid_over_rid_derivative =
                        DstrEncodeDetail::ps_over_rs_derivative_[NDimP - 1][idim_r];

                    static_assert(math::is_power_of_two_integer(r_length),
                                  "wrong! only support power of 2 reduction");

                    constexpr index_t nstage = math::integer_log2_floor(r_length);

                    // broadcast sweep backward
                    static_for<0, nstage, 1>{}([&](auto istage) {
                        // do I hold reduced data?
                        const bool do_i_hold_reduced_data = r_id < (1 << istage);

                        constexpr index_t lid_delta = lid_over_rid_derivative * (1 << istage);

                        // pull data from remote lane
                        const auto v_remote_mean  = warp_shuffle_up(v_local_mean, lid_delta);
                        const auto v_remote_var   = warp_shuffle_up(v_local_var, lid_delta);
                        const auto v_remote_count = warp_shuffle_up(v_local_count, lid_delta);

                        // decide whether to update local data with remote data
                        v_local_mean  = do_i_hold_reduced_data ? v_local_mean : v_remote_mean;
                        v_local_var   = do_i_hold_reduced_data ? v_local_var : v_remote_var;
                        v_local_count = do_i_hold_reduced_data ? v_local_count : v_remote_count;
                    });
                }
            });

            mean_tensor.GetThreadBuffer()(i) = v_local_mean;

            if constexpr(GetActualVariance)
                var_tensor.GetThreadBuffer()(i) = v_local_var / v_local_count;
            else
                var_tensor.GetThreadBuffer()(i) = v_local_var;

            count = v_local_count;
        });
    }
};

} // namespace warp
} // namespace tile_program
} // namespace ck
