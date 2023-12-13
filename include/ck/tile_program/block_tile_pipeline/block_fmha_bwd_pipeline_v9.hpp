// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/load_tile.hpp"
#include "ck/tile_program/tile/store_tile.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/tile/slice_tile.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_bwd_pipeline_default_policy.hpp"
#include "ck/tile_program/block_tile/block_reduce.hpp"
#include "ck/tile_program/tile/shuffle_distributed_tensor.hpp"

namespace ck {
namespace tile_program {
namespace block {

template <typename Problem, typename Policy = BlockFmhaBwdPipelineDefaultPolicy>
struct BlockFmhaBwdPipelineV9
{
    using QDataType           = remove_cvref_t<typename Problem::QDataType>;
    using KDataType           = remove_cvref_t<typename Problem::KDataType>;
    using VDataType           = remove_cvref_t<typename Problem::VDataType>;
    using GemmDataType        = remove_cvref_t<typename Problem::GemmDataType>;
    using LSEDataType         = remove_cvref_t<typename Problem::LSEDataType>;
    using AccDataType         = remove_cvref_t<typename Problem::AccDataType>;
    using SMPLComputeDataType = remove_cvref_t<typename Problem::SMPLComputeDataType>;
    using DDataType           = remove_cvref_t<typename Problem::DDataType>;
    // using ZDataType        = remove_cvref_t<typename Problem::ZDataType>;
    using ODataType     = remove_cvref_t<typename Problem::ODataType>;
    using OGradDataType = remove_cvref_t<typename Problem::OGradDataType>;
    using QGradDataType = remove_cvref_t<typename Problem::QGradDataType>;

    using BlockFmhaShape = remove_cvref_t<typename Problem::BlockFmhaShape>;

    static constexpr index_t kBlockSize = Problem::kBlockSize;

    static constexpr index_t kM0        = BlockFmhaShape::kM0;
    static constexpr index_t kN0        = BlockFmhaShape::kN0;
    static constexpr index_t kK0        = BlockFmhaShape::kK0;
    static constexpr index_t kK1        = BlockFmhaShape::kK1;
    static constexpr index_t kK2        = BlockFmhaShape::kK2;
    static constexpr index_t kK3        = BlockFmhaShape::kK3;
    static constexpr index_t kK4        = BlockFmhaShape::kK4;
    static constexpr index_t kQKHeaddim = BlockFmhaShape::kQKHeaddim;
    static constexpr index_t kVHeaddim  = BlockFmhaShape::kVHeaddim;

    static constexpr bool kQLoadOnce      = BlockFmhaShape::kQLoadOnce;
    static constexpr bool kQTLoadOnce     = BlockFmhaShape::kQTLoadOnce;
    static constexpr bool kKLoadOnce      = BlockFmhaShape::kKLoadOnce;
    static constexpr bool kKTLoadOnce     = BlockFmhaShape::kKTLoadOnce;
    static constexpr bool kVLoadOnce      = BlockFmhaShape::kVLoadOnce;
    static constexpr bool kOGradLoadOnce  = BlockFmhaShape::kOGradLoadOnce;
    static constexpr bool kOGradTLoadOnce = BlockFmhaShape::kOGradTLoadOnce;

    __host__ __device__ static constexpr ck::index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    template <typename QDramBlockWindowTmp,
              typename QTDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename KTDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename OGradDramBlockWindowTmp,
              typename OGradTDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename DDramBlockWindowTmp,
              typename QGradDramBlockWindowTmp>
    __host__ __device__ auto operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
                                        const QTDramBlockWindowTmp& qt_dram_block_window_tmp,
                                        const KDramBlockWindowTmp& k_dram_block_window_tmp,
                                        const KTDramBlockWindowTmp& kt_dram_block_window_tmp,
                                        const VDramBlockWindowTmp& v_dram_block_window_tmp,
                                        const OGradDramBlockWindowTmp& do_dram_block_window_tmp,
                                        const OGradTDramBlockWindowTmp& dot_dram_block_window_tmp,
                                        const LSEDramBlockWindowTmp& lse_dram_block_window_tmp,
                                        const DDramBlockWindowTmp& d_dram_block_window_tmp,
                                        const QGradDramBlockWindowTmp& dq_dram_block_window_tmp,
                                        float scale,
                                        index_t num_total_loop,
                                        void* smem_ptr) const
    {
        static_assert(
            is_same_v<QDataType, remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                is_same_v<QDataType, remove_cvref_t<typename QTDramBlockWindowTmp::DataType>> &&
                is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                is_same_v<KDataType, remove_cvref_t<typename KTDramBlockWindowTmp::DataType>> &&
                is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>> &&
                is_same_v<OGradDataType,
                          remove_cvref_t<typename OGradDramBlockWindowTmp::DataType>> &&
                is_same_v<OGradDataType,
                          remove_cvref_t<typename OGradTDramBlockWindowTmp::DataType>> &&
                is_same_v<LSEDataType, remove_cvref_t<typename LSEDramBlockWindowTmp::DataType>> &&
                is_same_v<DDataType, remove_cvref_t<typename DDramBlockWindowTmp::DataType>> &&
                is_same_v<QGradDataType,
                          remove_cvref_t<typename QGradDramBlockWindowTmp::DataType>>,
            "wrong!");

        static_assert(kM0 == QDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kM0 == QTDramBlockWindowTmp{}.GetWindowLengths()[Number<1>{}] &&
                          kN0 == KDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kN0 == KTDramBlockWindowTmp{}.GetWindowLengths()[Number<1>{}] &&
                          kN0 == VDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kM0 == OGradDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kM0 == OGradTDramBlockWindowTmp{}.GetWindowLengths()[Number<1>{}] &&
                          kM0 == LSEDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kM0 == DDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}] &&
                          kM0 == QGradDramBlockWindowTmp{}.GetWindowLengths()[Number<0>{}],
                      "wrong!");

        // Q tile in LDS
        QDataType* q_lds_ptr = static_cast<QDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>() +
            Policy::template GetSmemSizeKT<Problem>()));
        auto q_lds           = make_tensor_view<AddressSpaceEnum::Lds>(
            q_lds_ptr, Policy::template MakeQLdsBlockDescriptor<Problem>());
        auto q_lds_window =
            make_tile_window(q_lds, make_tuple(Number<kM0>{}, Number<kK0>{}), {0, 0});

        // QT tile in LDS
        QDataType* qt_lds_ptr = static_cast<QDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>() +
            Policy::template GetSmemSizeKT<Problem>()));
        auto qt_lds           = make_tensor_view<AddressSpaceEnum::Lds>(
            qt_lds_ptr, Policy::template MakeQTLdsBlockDescriptor<Problem>());
        auto qt_lds_window =
            make_tile_window(qt_lds, make_tuple(Number<kQKHeaddim>{}, Number<kK3>{}), {0, 0});

        // K tile in LDS
        auto k_lds = make_tensor_view<AddressSpaceEnum::Lds>(
            reinterpret_cast<KDataType*>(smem_ptr),
            Policy::template MakeKLdsBlockDescriptor<Problem>());
        auto k_lds_window =
            make_tile_window(k_lds, make_tuple(Number<kN0>{}, Number<kK0>{}), {0, 0});

        // KT tile in LDS
        KDataType* kt_lds_ptr = static_cast<KDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>()));
        auto kt_lds           = make_tensor_view<AddressSpaceEnum::Lds>(
            kt_lds_ptr, Policy::template MakeKTLdsBlockDescriptor<Problem>());
        auto kt_lds_window =
            make_tile_window(kt_lds, make_tuple(Number<kQKHeaddim>{}, Number<kK4>{}), {0, 0});

        // OGrad tile in LDS
        OGradDataType* do_lds_ptr = static_cast<OGradDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>() +
            Policy::template GetSmemSizeKT<Problem>()));
        auto do_lds               = make_tensor_view<AddressSpaceEnum::Lds>(
            do_lds_ptr, Policy::template MakeOGradLdsBlockDescriptor<Problem>());
        auto do_lds_window =
            make_tile_window(do_lds, make_tuple(Number<kM0>{}, Number<kK2>{}), {0, 0});

        // OGradT tile in LDS
        OGradDataType* dot_lds_ptr = static_cast<OGradDataType*>(static_cast<void*>(
            static_cast<char*>(smem_ptr) + Policy::template GetSmemSizeK<Problem>() +
            Policy::template GetSmemSizeKT<Problem>()));
        auto dot_lds               = make_tensor_view<AddressSpaceEnum::Lds>(
            dot_lds_ptr, Policy::template MakeOGradTLdsBlockDescriptor<Problem>());
        auto dot_lds_window =
            make_tile_window(dot_lds, make_tuple(Number<kVHeaddim>{}, Number<kK1>{}), {0, 0});

        // Block GEMM
        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPTOGradTBlockGemm<Problem>();
        constexpr auto gemm_2 = Policy::template GetOGradVBlockGemm<Problem>();
        constexpr auto gemm_3 = Policy::template GetSGradTQTBlockGemm<Problem>();
        constexpr auto gemm_4 = Policy::template GetSGradKTBlockGemm<Problem>();

        auto v_dram_window = make_tile_window(
            v_dram_block_window_tmp.GetBottomTensorView(),
            v_dram_block_window_tmp.GetWindowLengths(),
            v_dram_block_window_tmp.GetWindowOrigin(),
            Policy::template MakeVDramRegStatTileDistribution<Problem, decltype(gemm_2)>());

        auto v = load_tile(v_dram_window); // persistent v register tile

        auto lse_dram_window = make_tile_window(
            lse_dram_block_window_tmp.GetBottomTensorView(),
            lse_dram_block_window_tmp.GetWindowLengths(),
            lse_dram_block_window_tmp.GetWindowOrigin(),
            Policy::template MakeLSEDDramTileDistribution<Problem, decltype(gemm_0)>());

        auto d_dram_window = make_tile_window(
            d_dram_block_window_tmp.GetBottomTensorView(),
            d_dram_block_window_tmp.GetWindowLengths(),
            d_dram_block_window_tmp.GetWindowOrigin(),
            Policy::template MakeLSEDDramTileDistribution<Problem, decltype(gemm_0)>());

        auto st_acc = decltype(gemm_0(q_lds_window, k_lds_window)){};

        // infer Sacc, S, P, M, L, Oacc type
        using SBlockTileType =
            decltype(tile_elementwise_in(type_convert<SMPLComputeDataType, SaccDataType>, s_acc));

        using PBlockTileType =
            decltype(tile_elementwise_in(type_convert<PDataType, SaccDataType>, s_acc));

        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, Sequence<1>{}, f_max, SMPLComputeDataType{0}));

        using OaccBlockTileType = decltype(
            gemm_1(get_slice_tile(PBlockTileType{}, Sequence<0, 0>{}, Sequence<kM0, kK1>{}),
                   v_lds_window));

        // init Oacc, M, L
        auto o_acc = OaccBlockTileType{};
        auto m     = MLBlockTileType{};
        auto l     = MLBlockTileType{};

        tile_elementwise_inout([](auto& e) { e = 0; }, o_acc);
        tile_elementwise_inout([](auto& e) { e = NumericLimits<SMPLComputeDataType>::Lowest(); },
                               m);
        tile_elementwise_inout([](auto& e) { e = 0; }, l);

        auto k_dram_block_window = k_dram_block_window_tmp;
        auto v_dram_window =
            make_tile_window(v_dram_block_window_tmp.GetBottomTensorView(),
                             v_dram_block_window_tmp.GetWindowLengths(),
                             v_dram_block_window_tmp.GetWindowOrigin(),
                             Policy::template MakeVDramTileDistribution<Problem>());

        auto q_tile           = tile_elementwise_in(q_element_func, q);
        index_t i_total_loops = 0;
        do
        {
            // STAGE 1, QK gemm
            auto k_dram_window = make_tile_window(
                k_dram_block_window.GetBottomTensorView(),
                k_dram_block_window.GetWindowLengths(),
                k_dram_block_window.GetWindowOrigin(),
                Policy::template MakeKDramTileDistribution<Problem>()); // K DRAM tile window for
                                                                        // load

            auto k_block_tile = load_tile(k_dram_window);
            {
                move_tile_window(k_dram_window, {0, kK0});

                tile_elementwise_inout([](auto& c) { c = 0; }, s_acc); // Initialize C

                store_tile(k_lds_window,
                           tile_elementwise_in(k_element_func, k_block_tile)); // LDS write 0
                k_block_tile = load_tile(k_dram_window);                       // global read 1
            }

            // index_t i_k0_loops = num_sub_loop_qk - 2;
            constexpr index_t k0_loops = kK0BlockLength / kK0;

            if constexpr(k0_loops > 2)
            {
                static_for<0, k0_loops - 2, 1>{}([&](auto i_k0) {
                    block_sync_lds();
                    gemm_0(s_acc,
                           get_slice_tile(q_tile,
                                          Sequence<0, i_k0 * kK0>{},
                                          Sequence<kM0, (i_k0 + 1) * kK0>{}),
                           k_lds_window);
                    block_sync_lds();
                    move_tile_window(k_dram_window, {0, kK0});

                    store_tile(
                        k_lds_window,
                        tile_elementwise_in(k_element_func, k_block_tile)); // LDS write i + 1
                    k_block_tile = load_tile(k_dram_window);                // global read i + 2
                });
            }

            const auto v_prefetch = load_tile(v_dram_window); // prefetch load v tile
            {                                                 // tail
                block_sync_lds();
                gemm_0(s_acc,
                       get_slice_tile(q_tile,
                                      Sequence<0, (k0_loops - 2) * kK0>{},
                                      Sequence<kM0, (k0_loops - 1) * kK0>{}),
                       k_lds_window);
                block_sync_lds();

                store_tile(k_lds_window, tile_elementwise_in(k_element_func, k_block_tile));
                block_sync_lds();

                gemm_0(s_acc,
                       get_slice_tile(q_tile,
                                      Sequence<0, (k0_loops - 1) * kK0>{},
                                      Sequence<kM0, k0_loops * kK0>{}),
                       k_lds_window);
            }

            // STAGE 2, scale softmax
            tile_elementwise_inout([&scale](auto& x) { x = x * scale; }, s_acc);

            const auto s =
                tile_elementwise_in(type_convert<SMPLComputeDataType, SaccDataType>, s_acc); // S{j}
            auto m_local = block_tile_reduce<SMPLComputeDataType>(
                s,
                Sequence<1>{},
                f_max,
                NumericLimits<SMPLComputeDataType>::Lowest()); // m_local = rowmax(S{j})
            block_tile_reduce_sync(m_local, f_max);

            const auto m_old = m; // m{j-1}
            tile_elementwise_inout(
                [](auto& e0, auto e1, auto e2) { e0 = max(e1, e2); }, m, m_old, m_local); // m{j}

            auto p_compute = make_static_distributed_tensor<SMPLComputeDataType>(
                s.GetTileDistribution()); // Pcompute{j}

            constexpr auto p_spans = decltype(p_compute)::GetDistributedSpans();
            sweep_tile_span(p_spans[Number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                sweep_tile_span(p_spans[Number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    p_compute(i_j_idx)     = math::exp(s[i_j_idx] - m[i_idx]);
                });
            });

            auto rowsum_p = block_tile_reduce<SMPLComputeDataType>(
                p_compute, Sequence<1>{}, f_sum, SMPLComputeDataType{0}); // rowsum(Pcompute{j})

            block_tile_reduce_sync(rowsum_p, f_sum);
            // l{j}, Oacc{j}
            constexpr auto o_spans = decltype(o_acc)::GetDistributedSpans();
            sweep_tile_span(o_spans[Number<0>{}], [&](auto idx0) {
                constexpr auto i_idx = make_tuple(idx0);
                const auto tmp       = math::exp(m_old[i_idx] - m[i_idx]);
                l(i_idx)             = tmp * l[i_idx] + rowsum_p[i_idx];
                sweep_tile_span(o_spans[Number<1>{}], [&](auto idx1) {
                    constexpr auto i_j_idx = make_tuple(idx0, idx1);
                    // FIXME: this use different equation from FA v2 paper,
                    // but produce correc result.
                    // Is the equation wrong?
                    o_acc(i_j_idx) *= tmp;
                });
            });

            block_sync_lds();
            if constexpr(ck::is_same_v<VLayout, ck::tensor_layout::gemm::RowMajor>)
            {
                auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                    Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                shuffle_distributed_tensor(v_shuffle_tmp, v_prefetch);
                store_tile(
                    v_lds_window,
                    tile_elementwise_in(v_element_func, v_shuffle_tmp)); // store the prefetch
            }
            else
            {
                store_tile(v_lds_window,
                           tile_elementwise_in(v_element_func, v_prefetch)); // store the prefetch
            }
            move_tile_window(v_dram_window, {0, kK1});

            const auto p =
                tile_elementwise_in(type_convert<PDataType, SMPLComputeDataType>, p_compute);

            // STAGE 3, KV gemm
            constexpr index_t k1_loops = kN0 / kK1;
            if constexpr(k1_loops > 1)
            {
                static_for<0, k1_loops - 1, 1>{}([&](auto i_k1) {
                    const auto v = load_tile(v_dram_window); // load next v
                    block_sync_lds();
                    gemm_1(o_acc,
                           get_slice_tile(
                               p, Sequence<0, i_k1 * kK1>{}, Sequence<kM0, (i_k1 + 1) * kK1>{}),
                           v_lds_window);
                    block_sync_lds();
                    if constexpr(ck::is_same_v<VLayout, ck::tensor_layout::gemm::RowMajor>)
                    {
                        auto v_shuffle_tmp = make_static_distributed_tensor<VDataType>(
                            Policy::template MakeShuffledVRegBlockDescriptor<Problem>());
                        shuffle_distributed_tensor(v_shuffle_tmp, v);
                        store_tile(v_lds_window,
                                   tile_elementwise_in(v_element_func,
                                                       v_shuffle_tmp)); // store the prefetch
                    }
                    else
                    {
                        store_tile(v_lds_window,
                                   tile_elementwise_in(v_element_func, v)); // store next v
                    }
                    move_tile_window(v_dram_window, {0, kK1});
                });
            }
            // move K tile windows
            move_tile_window(k_dram_block_window, {kN0, 0});
            i_total_loops++;
            // tail
            {
                block_sync_lds();
                gemm_1(o_acc,
                       get_slice_tile(p, Sequence<0, (k1_loops - 1) * kK1>{}, Sequence<kM0, kN0>{}),
                       v_lds_window);
                block_sync_lds();
            }
        } while(i_total_loops < num_total_loop);

        // finally, O
        constexpr auto o_spans = decltype(o_acc)::GetDistributedSpans();

        sweep_tile_span(o_spans[Number<0>{}], [&](auto idx0) {
            constexpr auto i_idx = make_tuple(idx0);
            const auto tmp       = 1 / l[i_idx];
            sweep_tile_span(o_spans[Number<1>{}], [&](auto idx1) {
                constexpr auto i_j_idx = make_tuple(idx0, idx1);
                o_acc(i_j_idx) *= tmp;
            });
        });

        return o_acc;
    }

    template <typename QDramBlockWindowTmp,
              typename QTDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename KTDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename OGradDramBlockWindowTmp,
              typename OGradTDramBlockWindowTmp,
              typename LSEDramBlockWindowTmp,
              typename DDramBlockWindowTmp,
              typename QGradDramBlockWindowTmp>
    __host__ __device__ auto operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
                                        const QTDramBlockWindowTmp& qt_dram_block_window_tmp,
                                        const KDramBlockWindowTmp& k_dram_block_window_tmp,
                                        const KTDramBlockWindowTmp& kt_dram_block_window_tmp,
                                        const VDramBlockWindowTmp& v_dram_block_window_tmp,
                                        const OGradDramBlockWindowTmp& do_dram_block_window_tmp,
                                        const OGradTDramBlockWindowTmp& dot_dram_block_window_tmp,
                                        const LSEDramBlockWindowTmp& lse_dram_block_window_tmp,
                                        const DDramBlockWindowTmp& d_dram_block_window_tmp,
                                        const QGradDramBlockWindowTmp& dq_dram_block_window_tmp,
                                        float scale,
                                        index_t num_total_loop,
                                        void* smem_ptr) const
    {
        return operator()(q_dram_block_window_tmp,
                          qt_dram_block_window_tmp,
                          k_dram_block_window_tmp,
                          kt_dram_block_window_tmp,
                          v_dram_block_window_tmp,
                          do_dram_block_window_tmp,
                          dot_dram_block_window_tmp,
                          lse_dram_block_window_tmp,
                          d_dram_block_window_tmp,
                          dq_dram_block_window_tmp,
                          scale,
                          num_total_loop,
                          smem_ptr);
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
