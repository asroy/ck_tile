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
#include "ck/tile_program/warp_tile/warp_gemm_dispatcher.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_problem.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_bsmem_creg_v1.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_bsmem_creg_v1_custom_policy.hpp"
#include "ck/tile_program/block_tile/block_gemm_asmem_breg_creg_v1.hpp"
#include "ck/tile_program/block_tile/block_gemm_asmem_breg_creg_v1_custom_policy.hpp"
#include "ck/tile_program/block_tile/block_gemm_asmem_bsmem_creg_v1.hpp"
#include "ck/tile_program/block_tile/block_gemm_asmem_bsmem_creg_v1_custom_policy.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

namespace ck {
namespace tile_program {
namespace block {

struct BlockFmhaBwdPipelineDefaultPolicy
{
    template <typename Problem>
    __host__ __device__ static constexpr auto GetSmemKPackQ()
    {
        // TODO: this is for 3d layout
        using QDataType = remove_cvref_t<typename Problem::QDataType>;
        return 16 / sizeof(QDataType);
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetSmemKPackK()
    {
        // TODO: this is for 3d layout
        using KDataType = remove_cvref_t<typename Problem::KDataType>;
        return 16 / sizeof(KDataType);
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetSmemKPackV()
    {
        // TODO: this is for 3d layout
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        return 16 / sizeof(VDataType);
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetSmemKPackOGrad()
    {
        // TODO: this is for 3d layout
        using OGradDataType = remove_cvref_t<typename Problem::OGradDataType>;
        return 16 / sizeof(OGradDataType);
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetSmemKPackSGrad()
    {
        // TODO: this is for 3d layout
        using GemmDataType = remove_cvref_t<typename Problem::GemmDataType>;
        return 16 / sizeof(GemmDataType);
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetTransposedVectorloadQ()
    {
        return 4; // TODO: fix me
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetTransposedVectorloadK()
    {
        return 4; // TODO: fix me
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetTransposedVectorloadOGrad()
    {
        return 4; // TODO: fix me
    }

    template <typename Problem, typename BlockGemm>
    __host__ __device__ static constexpr auto MakeVRegBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kVHeaddim;

        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template At<0>())>;

        constexpr index_t MWarp = config.template At<1>();
        constexpr index_t NWarp = config.template At<2>();

        constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WG::kN);
        constexpr index_t KIterPerWarp = kKPerBlock / WG::kK;

        constexpr auto v_block_outer_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<MWarp>,
            Tuple<Sequence<NIterPerWarp, NWarp>, Sequence<KIterPerWarp>>,
            Tuple<Sequence<0, 1>>,
            Tuple<Sequence<0, 1>>,
            Sequence<1, 2>,
            Sequence<0, 0>>{};

        constexpr auto v_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            v_block_outer_dstr_encoding, typename WG::BWarpDstrEncoding{});

        constexpr auto v_block_dstr = make_static_tile_distribution(v_block_dstr_encode);

        return v_block_dstr;
    }

    // 3d + padding
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeQLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock =
            Problem::BlockFmhaShape::kK0; // Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPack = GetSmemKPackQ<Problem>();

        constexpr auto q_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kKPack>{}, Number<kMPerBlock>{}, Number<kKPack>{}),
            make_tuple(Number<(kMPerBlock + 1) * kKPack>{}, Number<kKPack>{}, Number<1>{}),
            Number<8>{},
            Number<1>{});

        constexpr auto q_lds_block_desc = transform_tensor_descriptor(
            q_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return q_lds_block_desc;
    }

    // 3d + padding
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock =
            Problem::BlockFmhaShape::kK0; // Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPack = GetSmemKPackK<Problem>();

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kKPack>{}, Number<kNPerBlock>{}, Number<kKPack>{}),
            make_tuple(Number<(kNPerBlock + 1) * kKPack>{}, Number<kKPack>{}, Number<1>{}),
            Number<8>{},
            Number<1>{});

        constexpr auto k_lds_block_desc = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return k_lds_block_desc;
    }

    // 3d + padding
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeVLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK2;
        constexpr index_t kPad       = 1;
        constexpr index_t kKPack     = GetSmemKPackV<Problem>();

        constexpr auto v_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kKPack>{}, Number<kNPerBlock>{}, Number<kKPack>{}),
            make_tuple(Number<(kNPerBlock + kPad) * kKPack>{}, Number<kKPack>{}, Number<1>{}),
            Number<kKPack>{},
            Number<1>{});

        constexpr auto v_lds_block_desc = transform_tensor_descriptor(
            v_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(kNPerBlock),
                make_merge_transform(make_tuple(Number<kKPerBlock / kKPack>{}, Number<kKPack>{}))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return v_lds_block_desc;
    }

    // 3d + padding
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeOGradLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock =
            Problem::BlockFmhaShape::kK2; // Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kKPack = GetSmemKPackOGrad<Problem>();

        constexpr auto do_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kKPack>{}, Number<kMPerBlock>{}, Number<kKPack>{}),
            make_tuple(Number<(kMPerBlock + 1) * kKPack>{}, Number<kKPack>{}, Number<1>{}),
            Number<8>{},
            Number<1>{});

        constexpr auto do_lds_block_desc = transform_tensor_descriptor(
            do_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return do_lds_block_desc;
    }

    // 3d + padding
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeSGradLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPack     = GetSmemKPackSGrad<Problem>();

        constexpr auto ds_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kKPack>{}, Number<kMPerBlock>{}, Number<kKPack>{}),
            make_tuple(Number<(kMPerBlock + 1) * kKPack>{}, Number<kKPack>{}, Number<1>{}),
            Number<8>{},
            Number<1>{});

        constexpr auto ds_lds_block_desc = transform_tensor_descriptor(
            ds_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kMPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / kKPack, kKPack))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return ds_lds_block_desc;
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeQTLdsBlockDescriptor()
    {
        using QDataType                = remove_cvref_t<typename Problem::QDataType>;
        constexpr index_t Banks        = 32; // TODO: need change based on arch
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(QDataType);
        constexpr index_t kKPack       = GetSmemKPackQ<Problem>();
        static_assert(PixelsPerRow % kKPack == 0);
        constexpr index_t NPerRow    = PixelsPerRow / kKPack;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock =
            Problem::BlockFmhaShape::kK3; // Problem::BlockFmhaShape::kM0;
        static_assert(kNPerBlock % NPerRow == 0);
        static_assert(kKPerBlock % kKPack == 0);

        constexpr auto qt_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kKPack>{},
                       Number<kNPerBlock / NPerRow>{},
                       Number<NPerRow>{},
                       Number<kKPack>{}),
            make_tuple(Number<(kNPerBlock / NPerRow) * (PixelsPerRow + kKPack)>{},
                       Number<PixelsPerRow + kKPack>{},
                       Number<kKPack>{},
                       Number<1>{}),
            Number<kKPack>{},
            Number<1>{});

        constexpr auto qt_lds_block_desc = transform_tensor_descriptor(
            qt_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(Number<kNPerBlock / NPerRow>{}, Number<NPerRow>{})),
                make_merge_transform(make_tuple(Number<kKPerBlock / kKPack>{}, Number<kKPack>{}))),
            make_tuple(Sequence<1, 2>{}, Sequence<0, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return qt_lds_block_desc;
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeKTLdsBlockDescriptor()
    {
        using KDataType                = remove_cvref_t<typename Problem::KDataType>;
        constexpr index_t Banks        = 32; // TODO: need change based on arch
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(KDataType);
        constexpr index_t kKPack       = GetSmemKPackK<Problem>();
        static_assert(PixelsPerRow % kKPack == 0);
        constexpr index_t NPerRow    = PixelsPerRow / kKPack;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock =
            Problem::BlockFmhaShape::kK4; // Problem::BlockFmhaShape::kN0;
        static_assert(kNPerBlock % NPerRow == 0);
        static_assert(kKPerBlock % kKPack == 0);

        constexpr auto kt_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kKPack>{},
                       Number<kNPerBlock / NPerRow>{},
                       Number<NPerRow>{},
                       Number<kKPack>{}),
            make_tuple(Number<(kNPerBlock / NPerRow) * (PixelsPerRow + kKPack)>{},
                       Number<PixelsPerRow + kKPack>{},
                       Number<kKPack>{},
                       Number<1>{}),
            Number<kKPack>{},
            Number<1>{});

        constexpr auto kt_lds_block_desc = transform_tensor_descriptor(
            kt_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(Number<kNPerBlock / NPerRow>{}, Number<NPerRow>{})),
                make_merge_transform(make_tuple(Number<kKPerBlock / kKPack>{}, Number<kKPack>{}))),
            make_tuple(Sequence<1, 2>{}, Sequence<0, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return kt_lds_block_desc;
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeOGradTLdsBlockDescriptor()
    {
        using QGradDataType            = remove_cvref_t<typename Problem::QGradDataType>;
        constexpr index_t Banks        = 32; // TODO: need change based on arch
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(QGradDataType);
        constexpr index_t kKPack       = GetSmemKPackOGrad<Problem>();
        static_assert(PixelsPerRow % kKPack == 0);
        constexpr index_t NPerRow    = PixelsPerRow / kKPack;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kKPerBlock =
            Problem::BlockFmhaShape::kK1; // Problem::BlockFmhaShape::kM0;
        static_assert(kNPerBlock % NPerRow == 0);
        static_assert(kKPerBlock % kKPack == 0);

        constexpr auto dot_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kKPack>{},
                       Number<kNPerBlock / NPerRow>{},
                       Number<NPerRow>{},
                       Number<kKPack>{}),
            make_tuple(Number<(kNPerBlock / NPerRow) * (PixelsPerRow + kKPack)>{},
                       Number<PixelsPerRow + kKPack>{},
                       Number<kKPack>{},
                       Number<1>{}),
            Number<kKPack>{},
            Number<1>{});

        constexpr auto dot_lds_block_desc = transform_tensor_descriptor(
            dot_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(Number<kNPerBlock / NPerRow>{}, Number<NPerRow>{})),
                make_merge_transform(make_tuple(Number<kKPerBlock / kKPack>{}, Number<kKPack>{}))),
            make_tuple(Sequence<1, 2>{}, Sequence<0, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return dot_lds_block_desc;
    }

    // template <typename Problem>
    // __host__ __device__ static constexpr ck::index_t GetSmemSize()
    // {
    //     constexpr index_t smem_size_gemm_0 =
    //         GetSmemSizeQ<Problem>() + sizeof(typename Problem::KDataType) *
    //                                       MakeKLdsBlockDescriptor<Problem>().GetElementSpaceSize();
    //     constexpr index_t smem_size_gemm_1 =
    //         MakeVLdsBlockDescriptor<Problem>().GetElementSpaceSize() *
    //         sizeof(typename Problem::VDataType);

    //     // TODO: consider shuffle requirement
    //     return math::max(smem_size_gemm_0, smem_size_gemm_1);
    // }

    template <typename Problem>
    __host__ __device__ static constexpr ck::index_t GetSmemSize()
    {
        constexpr index_t smem_size_gemm_q =
            sizeof(typename Problem::QDataType) *
            MakeQLdsBlockDescriptor<Problem>().GetElementSpaceSize();
        constexpr index_t smem_size_gemm_qt =
            sizeof(typename Problem::QDataType) *
            MakeQTLdsBlockDescriptor<Problem>().GetElementSpaceSize();
        constexpr index_t smem_size_gemm_k =
            sizeof(typename Problem::KDataType) *
            MakeKLdsBlockDescriptor<Problem>().GetElementSpaceSize();
        constexpr index_t smem_size_gemm_kt =
            sizeof(typename Problem::KDataType) *
            MakeKTLdsBlockDescriptor<Problem>().GetElementSpaceSize();
        constexpr index_t smem_size_gemm_do =
            sizeof(typename Problem::OGradDataType) *
            MakeOGradLdsBlockDescriptor<Problem>().GetElementSpaceSize();
        constexpr index_t smem_size_gemm_dot =
            sizeof(typename Problem::OGradDataType) *
            MakeOGradTLdsBlockDescriptor<Problem>().GetElementSpaceSize();
        constexpr index_t smem_size_gemm_v =
            sizeof(typename Problem::VDataType) *
            MakeVLdsBlockDescriptor<Problem>().GetElementSpaceSize();
        constexpr index_t smem_size_gemm_ds =
            sizeof(typename Problem::GemmDataType) *
            MakeSGradLdsBlockDescriptor<Problem>().GetElementSpaceSize();

        // TODO: consider shuffle requirement
        return smem_size_gemm_q + smem_size_gemm_qt + smem_size_gemm_k + smem_size_gemm_kt +
               smem_size_gemm_do + smem_size_gemm_dot + smem_size_gemm_v + smem_size_gemm_ds;
    }

    template <typename Problem, typename BlockGemm>
    __host__ __device__ static constexpr auto MakeVDramRegStatTileDistribution()
    {
        constexpr auto config   = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG                = remove_cvref_t<decltype(config.template At<0>())>;
        constexpr index_t NWarp = config.template At<2>();

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kVHeaddim;

        constexpr index_t K2 = WG::kK / WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K1 = WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K0 = kKPerBlock / (K1 * K2);

        constexpr index_t N2 = WG::WarpGemmAttribute::Impl::kANLane;
        constexpr index_t N1 = NWarp;
        constexpr index_t N0 = kNPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1, N2>, Sequence<K0, K1, K2>>,
                                           Tuple<Sequence<1>, Sequence<2, 1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Sequence<2, 1, 2>,
                                           Sequence<0, 0, 2>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeVDramRegTempTileDistribution()
    {
        using VDataType = remove_cvref_t<typename Problem::VDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK2;

        constexpr index_t K1 = 16 / sizeof(VDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
        // coalesce reading for each blocks
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = kNPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1, N2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<1>, Sequence<2, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<0, 1>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeQDramTileDistribution()
    {
        using QDataType = remove_cvref_t<typename Problem::QDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock =
            Problem::BlockFmhaShape::kK0; // Problem::BlockFmhaShape::kVHeaddim;

        constexpr index_t K1 = 16 / sizeof(QDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t M2 = get_warp_size() / K0;
        // coalesce reading for each blocks
        constexpr index_t M1 = kBlockSize / get_warp_size();
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<M0, M1, M2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<1>, Sequence<2, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<0, 1>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeKDramTileDistribution()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock =
            Problem::BlockFmhaShape::kK0; // Problem::BlockFmhaShape::kQKHeaddim;

        constexpr index_t K1 = 16 / sizeof(KDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
        // coalesce reading for each blocks
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = kNPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1, N2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<1>, Sequence<2, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<0, 1>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeOGradDramTileDistribution()
    {
        using OGradDataType = remove_cvref_t<typename Problem::OGradDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock =
            Problem::BlockFmhaShape::kK0; // Problem::BlockFmhaShape::kVHeaddim;

        constexpr index_t K1 = 16 / sizeof(OGradDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t M2 = get_warp_size() / K0;
        // coalesce reading for each blocks
        constexpr index_t M1 = kBlockSize / get_warp_size();
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<M0, M1, M2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<1>, Sequence<2, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<0, 1>>{});
    }

    template <typename Problem>
    __device__ static constexpr auto MakeQTDramTileDistribution()
    {
        using QDataType = remove_cvref_t<typename Problem::QDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock =
            Problem::BlockFmhaShape::kK3; // Problem::BlockFmhaShape::kM0;

        constexpr index_t N1 = GetTransposedVectorloadQ<Problem>();
        constexpr index_t N0 = kNPerBlock / N1; // P

        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        static_assert(total_pixels % N1 == 0); // TODO: this is not always true?
        constexpr index_t K3     = total_pixels / N1;
        constexpr index_t kKPack = GetSmemKPackQ<Problem>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        constexpr index_t K1 = get_warp_size() / (K2 * N0);
        constexpr index_t K0 = kBlockSize / get_warp_size();
        static_assert(kKPerBlock == K0 * K1 * K2 * K3);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2, K3>>,
                                           Tuple<Sequence<2>, Sequence<2, 1, 2>>,
                                           Tuple<Sequence<0>, Sequence<1, 0, 2>>,
                                           Sequence<2, 1>,
                                           Sequence<3, 1>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeShuffledQTRegBlockDescriptor()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock =
            Problem::BlockFmhaShape::kK3; // Problem::BlockFmhaShape::kM0;

        constexpr index_t N1           = GetTransposedVectorloadQ<Problem>();
        constexpr index_t N0           = kNPerBlock / N1;
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        static_assert(total_pixels % N1 == 0); // TODO: this is not always true?
        constexpr index_t K3     = total_pixels / N1;
        constexpr index_t kKPack = GetSmemKPackQ<Problem>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        constexpr index_t K1 = get_warp_size() / (K2 * N0);
        constexpr index_t K0 = kBlockSize / get_warp_size();

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2, K3>>,
                                           Tuple<Sequence<2>, Sequence<2, 1, 2>>,
                                           Tuple<Sequence<0>, Sequence<1, 0, 2>>,
                                           Sequence<1, 2>,
                                           Sequence<1, 3>>{});
    }

    template <typename Problem>
    __device__ static constexpr auto MakeKTDramTileDistribution()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock =
            Problem::BlockFmhaShape::kK4; // Problem::BlockFmhaShape::kN0;

        constexpr index_t N1 = GetTransposedVectorloadK<Problem>();
        constexpr index_t N0 = kNPerBlock / N1; // P

        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        static_assert(total_pixels % N1 == 0); // TODO: this is not always true?
        constexpr index_t K3     = total_pixels / N1;
        constexpr index_t kKPack = GetSmemKPackK<Problem>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        constexpr index_t K1 = get_warp_size() / (K2 * N0);
        constexpr index_t K0 = kBlockSize / get_warp_size();
        static_assert(kKPerBlock == K0 * K1 * K2 * K3);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2, K3>>,
                                           Tuple<Sequence<2>, Sequence<2, 1, 2>>,
                                           Tuple<Sequence<0>, Sequence<1, 0, 2>>,
                                           Sequence<2, 1>,
                                           Sequence<3, 1>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeShuffledKTRegBlockDescriptor()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kQKHeaddim;
        constexpr index_t kKPerBlock =
            Problem::BlockFmhaShape::kK4; // Problem::BlockFmhaShape::kN0;

        constexpr index_t N1           = GetTransposedVectorloadK<Problem>();
        constexpr index_t N0           = kNPerBlock / N1;
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        static_assert(total_pixels % N1 == 0); // TODO: this is not always true?
        constexpr index_t K3     = total_pixels / N1;
        constexpr index_t kKPack = GetSmemKPackK<Problem>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        constexpr index_t K1 = get_warp_size() / (K2 * N0);
        constexpr index_t K0 = kBlockSize / get_warp_size();

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2, K3>>,
                                           Tuple<Sequence<2>, Sequence<2, 1, 2>>,
                                           Tuple<Sequence<0>, Sequence<1, 0, 2>>,
                                           Sequence<1, 2>,
                                           Sequence<1, 3>>{});
    }

    template <typename Problem>
    __device__ static constexpr auto MakeOGradTDramTileDistribution()
    {
        using OGradDataType = remove_cvref_t<typename Problem::OGradDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kKPerBlock =
            Problem::BlockFmhaShape::kK1; // Problem::BlockFmhaShape::kM0;

        constexpr index_t N1 = GetTransposedVectorloadOGrad<Problem>();
        constexpr index_t N0 = kNPerBlock / N1; // P

        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        static_assert(total_pixels % N1 == 0); // TODO: this is not always true?
        constexpr index_t K3     = total_pixels / N1;
        constexpr index_t kKPack = GetSmemKPackOGrad<Problem>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        constexpr index_t K1 = get_warp_size() / (K2 * N0);
        constexpr index_t K0 = kBlockSize / get_warp_size();
        static_assert(kKPerBlock == K0 * K1 * K2 * K3);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2, K3>>,
                                           Tuple<Sequence<2>, Sequence<2, 1, 2>>,
                                           Tuple<Sequence<0>, Sequence<1, 0, 2>>,
                                           Sequence<2, 1>,
                                           Sequence<3, 1>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeShuffledOGradTRegBlockDescriptor()
    {
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kVHeaddim;
        constexpr index_t kKPerBlock =
            Problem::BlockFmhaShape::kK1; // Problem::BlockFmhaShape::kM0;

        constexpr index_t N1           = GetTransposedVectorloadOGrad<Problem>();
        constexpr index_t N0           = kNPerBlock / N1;
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        static_assert(total_pixels % N1 == 0); // TODO: this is not always true?
        constexpr index_t K3     = total_pixels / N1;
        constexpr index_t kKPack = GetSmemKPackOGrad<Problem>();
        static_assert(kKPack % K3 == 0);
        constexpr index_t K2 = kKPack / K3; // TODO: this dimention could be outside single wave
        constexpr index_t K1 = get_warp_size() / (K2 * N0);
        constexpr index_t K0 = kBlockSize / get_warp_size();

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2, K3>>,
                                           Tuple<Sequence<2>, Sequence<2, 1, 2>>,
                                           Tuple<Sequence<0>, Sequence<1, 0, 2>>,
                                           Sequence<1, 2>,
                                           Sequence<1, 3>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetQKBlockGemm()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::QDataType,
                                     typename Problem::KDataType,
                                     typename Problem::AccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::BlockFmhaShape::kM0,
                                                   Problem::BlockFmhaShape::kN0,
                                                   Problem::BlockFmhaShape::kK0>>;

        using WarpGemm =
            warp::WarpGemmImpl<warp::WarpGemmAtrributeMfmaIterateKCDistribution_SwizzleB<
                warp::WarpGemmAttributeMfmaImplF16F16F32M32N32K8,
                2>>;

        using BlockGemmPolicy =
            BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::QDataType,
                                                  typename Problem::KDataType,
                                                  typename Problem::SaccDataType,
                                                  typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                  WarpGemm>;

        return BlockGemmASmemBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
    }

    // template <typename Problem>
    // __host__ __device__ static constexpr auto GetQKTBlockGemm()
    // {
    //     using BlockGemmProblem =
    //         BlockGemmPipelineProblem<typename Problem::QDataType,
    //                                  typename Problem::KDataType,
    //                                  typename Problem::AccDataType,
    //                                  Problem::kBlockSize,
    //                                  TileGemmShape<Problem::BlockFmhaShape::kM0,
    //                                                Problem::BlockFmhaShape::kN0,
    //                                                Problem::BlockFmhaShape::kK0>>;

    //     using WarpGemm =
    //         warp::WarpGemmImpl<warp::WarpGemmAtrributeMfmaIterateKCDistribution_SwizzleB<
    //             warp::WarpGemmAttributeMfmaImplF16F16F32M32N32K8,
    //             2>>;

    //     using BlockGemmPolicy =
    //         BlockGemmARegBSmemCRegV1CustomPolicy<typename Problem::QDataType,
    //                                              typename Problem::KDataType,
    //                                              typename Problem::AccDataType,
    //                                              typename
    //                                              Problem::BlockFmhaShape::Gemm0BlockWarps,
    //                                              WarpGemm>;

    //     return BlockGemmARegBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
    // }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetPTOGradBlockGemm()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::GemmDataType,
                                     typename Problem::OGradDataType,
                                     typename Problem::AccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::BlockFmhaShape::kN0,
                                                   Problem::BlockFmhaShape::kVHeaddim,
                                                   Problem::BlockFmhaShape::kK1>>;

        using WarpGemm = ck::tile_program::warp::WarpGemmMfmaDispatcher<
            typename Problem::GemmDataType,
            typename Problem::OGradDataType,
            typename Problem::AccDataType,
            Problem::BlockFmhaShape::Gemm1WarpTile::At(Number<0>{}),
            Problem::BlockFmhaShape::Gemm1WarpTile::At(Number<1>{}),
            Problem::BlockFmhaShape::Gemm1WarpTile::At(Number<2>{}),
            true>;
        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV1CustomPolicy<typename Problem::GemmDataType,
                                                 typename Problem::OGradDataType,
                                                 typename Problem::AccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                 WarpGemm>;
        return BlockGemmARegBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetOGradVTBlockGemm()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::OGradDataType,
                                     typename Problem::VDataType,
                                     typename Problem::AccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::BlockFmhaShape::kM0,
                                                   Problem::BlockFmhaShape::kN0,
                                                   Problem::BlockFmhaShape::kK2>>;

        using WarpGemm =
            warp::WarpGemmImpl<warp::WarpGemmAtrributeMfmaIterateKCDistribution_SwizzleB<
                warp::WarpGemmAttributeMfmaImplF16F16F32M32N32K8,
                2>>;

        using BlockGemmPolicy =
            BlockGemmASmemBRegCRegV1CustomPolicy<typename Problem::OGradDataType,
                                                 typename Problem::VDataType,
                                                 typename Problem::AccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm2BlockWarps,
                                                 WarpGemm>;

        return BlockGemmASmemBRegCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
    }

    // template <typename Problem>
    // __host__ __device__ static constexpr auto GetOGradVTBlockGemm()
    // {
    //     using BlockGemmProblem =
    //         BlockGemmPipelineProblem<typename Problem::OGradDataType,
    //                                  typename Problem::VDataType,
    //                                  typename Problem::AccDataType,
    //                                  Problem::kBlockSize,
    //                                  TileGemmShape<Problem::BlockFmhaShape::kM0,
    //                                                Problem::BlockFmhaShape::kN0,
    //                                                Problem::BlockFmhaShape::kK2>>;
    //     using WarpGemm =
    //         warp::WarpGemmImpl<warp::WarpGemmAtrributeMfmaIterateKCDistribution_SwizzleB<
    //             warp::WarpGemmAttributeMfmaImplF16F16F32M32N32K8,
    //             2>>;

    //     using BlockGemmPolicy =
    //         BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::OGradDataType,
    //                                               typename Problem::VDataType,
    //                                               typename Problem::AccDataType,
    //                                               typename
    //                                               Problem::BlockFmhaShape::Gemm2BlockWarps,
    //                                               WarpGemm>;

    //     return BlockGemmASmemBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
    // }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetSGradTQBlockGemm()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::GemmDataType,
                                     typename Problem::QDataType,
                                     typename Problem::AccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::BlockFmhaShape::kN0,
                                                   Problem::BlockFmhaShape::kQKHeaddim,
                                                   Problem::BlockFmhaShape::kK3>>;

        using WarpGemm = ck::tile_program::warp::WarpGemmMfmaDispatcher<
            typename Problem::GemmDataType,
            typename Problem::QDataType,
            typename Problem::AccDataType,
            Problem::BlockFmhaShape::Gemm3WarpTile::At(Number<0>{}),
            Problem::BlockFmhaShape::Gemm3WarpTile::At(Number<1>{}),
            Problem::BlockFmhaShape::Gemm3WarpTile::At(Number<2>{}),
            true>;
        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV1CustomPolicy<typename Problem::GemmDataType,
                                                 typename Problem::QDataType,
                                                 typename Problem::AccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm3BlockWarps,
                                                 WarpGemm>;
        return BlockGemmARegBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetSGradKBlockGemm()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::GemmDataType,
                                     typename Problem::KDataType,
                                     typename Problem::AccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::BlockFmhaShape::kM0,
                                                   Problem::BlockFmhaShape::kQKHeaddim,
                                                   Problem::BlockFmhaShape::kK4>>;

        using WarpGemm = ck::tile_program::warp::WarpGemmMfmaDispatcher<
            typename Problem::GemmDataType,
            typename Problem::KDataType,
            typename Problem::AccDataType,
            Problem::BlockFmhaShape::Gemm4WarpTile::At(Number<0>{}),
            Problem::BlockFmhaShape::Gemm4WarpTile::At(Number<1>{}),
            Problem::BlockFmhaShape::Gemm4WarpTile::At(Number<2>{}),
            true>;
        using BlockGemmPolicy =
            BlockGemmASmemBSmemCRegV1CustomPolicy<typename Problem::GemmDataType,
                                                  typename Problem::KDataType,
                                                  typename Problem::AccDataType,
                                                  typename Problem::BlockFmhaShape::Gemm4BlockWarps,
                                                  WarpGemm>;
        return BlockGemmASmemBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
