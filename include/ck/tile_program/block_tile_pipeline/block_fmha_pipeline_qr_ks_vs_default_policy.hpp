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
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

namespace ck {
namespace tile_program {
namespace block {

// This pipeline is qkv all located in LDS
struct BlockFmhaPipelineQRKSVSDefaultPolicy
{
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

    template <typename Problem, typename BlockGemm>
    __host__ __device__ static constexpr auto MakeQRegBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0BlockLength;

        constexpr auto config = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();

        using WG = remove_cvref_t<decltype(config.template At<0>())>;

        constexpr index_t MWarp = config.template At<1>();
        constexpr index_t NWarp = config.template At<2>();

        constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WG::kM);
        constexpr index_t KIterPerWarp = kKPerBlock / WG::kK;

        constexpr auto q_block_outer_dstr_encoding = StaticTileDistributionEncoding<
            Sequence<NWarp>,
            Tuple<Sequence<MIterPerWarp, MWarp>, Sequence<KIterPerWarp>>,
            Tuple<Sequence<1, 0>>,
            Tuple<Sequence<1, 0>>,
            Sequence<1, 2>,
            Sequence<0, 0>>{};

        constexpr auto q_block_dstr_encode = detail::make_embed_tile_distribution_encoding(
            q_block_outer_dstr_encoding, typename WG::AWarpDstrEncoding{});

        constexpr auto q_block_dstr = make_static_tile_distribution(q_block_dstr_encode);

        return q_block_dstr;
    }

    // 3d + padding
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeKLdsBlockDescriptor()
    {
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr auto k_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / 8>{}, Number<kNPerBlock>{}, Number<8>{}),
            make_tuple(Number<(kNPerBlock + 1) * 8>{}, Number<8>{}, Number<1>{}),
            Number<8>{},
            Number<1>{});

        constexpr auto k_lds_block_desc = transform_tensor_descriptor(
            k_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                       make_merge_transform(make_tuple(kKPerBlock / 8, 8))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return k_lds_block_desc;
    }

    // 3d + padding
    template <typename Problem>
    __host__ __device__ static constexpr auto MakeVLdsBlockDescriptor()
    {
#if 1
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t kPad       = 1;
        constexpr index_t kK1        = 8;

        constexpr auto v_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / kK1>{}, Number<kNPerBlock>{}, Number<kK1>{}),
            make_tuple(Number<(kNPerBlock + kPad) * kK1>{}, Number<kK1>{}, Number<1>{}),
            Number<kK1>{},
            Number<1>{});

        constexpr auto v_lds_block_desc = transform_tensor_descriptor(
            v_lds_block_desc_0,
            make_tuple(make_pass_through_transform(kNPerBlock),
                       make_merge_transform(make_tuple(Number<kKPerBlock / kK1>{}, Number<kK1>{}))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return v_lds_block_desc;
#elif 0
        using VDataType                = remove_cvref_t<typename Problem::VDataType>;
        constexpr index_t PixelsPerRow = 32 * 4 / sizeof(VDataType);
        constexpr index_t KPack        = GetSmemKPackV<Problem>();
        static_assert(PixelsPerRow % KPack == 0);
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        constexpr index_t NumRows    = (kNPerBlock * kKPerBlock + PixelsPerRow - 1) /
                                    PixelsPerRow; // TODO: not power of 2 block size?

        constexpr auto v_lds_block_desc_0 =
            make_naive_tensor_descriptor(make_tuple(Number<NumRows>{}, Number<PixelsPerRow>{}),
                                         make_tuple(Number<PixelsPerRow + KPack>{}, Number<1>{}),
                                         Number<KPack>{},
                                         Number<1>{});

        constexpr auto v_lds_block_desc_1 = transform_tensor_descriptor(
            v_lds_block_desc_0,
            make_tuple(make_merge_transform(make_tuple(Number<NumRows>{}, Number<PixelsPerRow>{}))),
            make_tuple(Sequence<0, 1>{}),
            make_tuple(Sequence<0>{}));

        constexpr auto v_lds_block_desc_2 = transform_tensor_descriptor(
            v_lds_block_desc_1,
            make_tuple(make_unmerge_transform(
                make_tuple(Number<kKPerBlock / KPack>{}, Number<kNPerBlock>{}, Number<KPack>{}))),
            make_tuple(Sequence<0>{}),
            make_tuple(Sequence<0, 1, 2>{}));

        constexpr auto v_lds_block_desc = transform_tensor_descriptor(
            v_lds_block_desc_2,
            make_tuple(
                make_pass_through_transform(kNPerBlock),
                make_merge_transform(make_tuple(Number<kKPerBlock / KPack>{}, Number<KPack>{}))),
            make_tuple(Sequence<1>{}, Sequence<0, 2>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return v_lds_block_desc;
#else
        using VDataType                = remove_cvref_t<typename Problem::VDataType>;
        constexpr index_t Banks        = 32;
        constexpr index_t PixelsPerRow = Banks * 4 / sizeof(VDataType);
        constexpr index_t KPack        = GetSmemKPackV<Problem>();
        static_assert(PixelsPerRow % KPack == 0);
        constexpr index_t NPerRow    = PixelsPerRow / KPack;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;
        static_assert(kNPerBlock % NPerRow == 0);
        static_assert(kKPerBlock % KPack == 0);

        constexpr auto v_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(Number<kKPerBlock / KPack>{},
                       Number<kNPerBlock / NPerRow>{},
                       Number<NPerRow>{},
                       Number<KPack>{}),
            make_tuple(Number<kNPerBlock / NPerRow*(PixelsPerRow + KPack)>{},
                       Number<PixelsPerRow + KPack>{},
                       Number<KPack>{},
                       Number<1>{}),
            Number<KPack>{},
            Number<1>{});

        constexpr auto v_lds_block_desc = transform_tensor_descriptor(
            v_lds_block_desc_0,
            make_tuple(
                make_merge_transform(make_tuple(Number<kNPerBlock / NPerRow>{}, Number<NPerRow>{})),
                make_merge_transform(make_tuple(Number<kKPerBlock / KPack>{}, Number<KPack>{}))),
            make_tuple(Sequence<1, 2>{}, Sequence<0, 3>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return v_lds_block_desc;
#endif
    }

    template <typename Problem>
    __host__ __device__ static constexpr ck::index_t GetSmemSizeQ()
    {
        return 0;
    }

    template <typename Problem>
    __host__ __device__ static constexpr ck::index_t GetSmemSize()
    {
        constexpr index_t smem_size_gemm_0 =
            GetSmemSizeQ<Problem>() + sizeof(typename Problem::KDataType) *
                                          MakeKLdsBlockDescriptor<Problem>().GetElementSpaceSize();
        constexpr index_t smem_size_gemm_1 =
            MakeVLdsBlockDescriptor<Problem>().GetElementSpaceSize() *
            sizeof(typename Problem::VDataType);

        // TODO: consider shuffle requirement
        return math::max(smem_size_gemm_0, smem_size_gemm_1);
    }

    template <typename Problem, typename BlockGemm>
    __host__ __device__ static constexpr auto MakeQDramTileDistribution()
    {
        constexpr auto config   = BlockGemm::Policy::template GetWarpGemmMWarpNWarp<Problem>();
        using WG                = remove_cvref_t<decltype(config.template At<0>())>;
        constexpr index_t MWarp = config.template At<1>();

        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0BlockLength;

        constexpr index_t K2 = WG::kK / WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K1 = WG::WarpGemmAttribute::Impl::kABKLane;
        constexpr index_t K0 = kKPerBlock / (K1 * K2);

        constexpr index_t M2 = WG::WarpGemmAttribute::Impl::kAMLane;
        constexpr index_t M1 = MWarp;
        constexpr index_t M0 = kMPerBlock / (M2 * M1);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<M0, M1, M2>, Sequence<K0, K1, K2>>,
                                           Tuple<Sequence<1>, Sequence<2, 1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Sequence<2, 1, 2>,
                                           Sequence<0, 0, 2>>{});
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeKDramTileDistribution()
    {
        using KDataType = remove_cvref_t<typename Problem::KDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;

        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK0;

        constexpr index_t K1 = 16 / sizeof(KDataType);
        constexpr index_t K0 = kKPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
#if 1 // coalesce reading for each blocks
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = kNPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1, N2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<1>, Sequence<2, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<0, 1>>{});
#else // coalesce reading for each warps
        constexpr index_t N0 = kBlockSize / get_warp_size();
        constexpr index_t N1 = kNPerBlock / (N2 * N0);

        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1, N2>, Sequence<K0, K1>>,
                                           Tuple<Sequence<1>, Sequence<1, 2>>,
                                           Tuple<Sequence<0>, Sequence<2, 0>>,
                                           Sequence<1, 2>,
                                           Sequence<1, 1>>{});
#endif
    }

    template <typename Problem>
    __device__ static constexpr auto MakeVDramTileDistribution()
    {
        using VDataType = remove_cvref_t<typename Problem::VDataType>;
        using VLayout   = remove_cvref_t<typename Problem::BlockFmhaShape::VLayout>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        if constexpr(ck::is_same_v<VLayout, ck::tensor_layout::gemm::RowMajor>)
        {
            constexpr index_t N1 = 4;               // Y, TODO: vector load
            constexpr index_t N0 = kNPerBlock / N1; // P
            // constexpr index_t K2 = get_warp_size() / N0;            // P
            // constexpr index_t K1 = kBlockSize / get_warp_size();    // P
            // constexpr index_t K0 = kKPerBlock / (K1 * K2);          // Y

            constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
            static_assert(total_pixels % N1 == 0);    // TODO: this is not always true?
            constexpr index_t K3 = total_pixels / N1; //      kKPerBlock / (K0 * K1);          // Y
            constexpr index_t KPack = GetSmemKPackV<Problem>();
            static_assert(KPack % K3 == 0);
            constexpr index_t K2 = KPack / K3; // TODO: this dimention could be outside single wave
            constexpr index_t K1 = get_warp_size() / (K2 * N0);  // P
            constexpr index_t K0 = kBlockSize / get_warp_size(); // P

            static_assert(kKPerBlock == K0 * K1 * K2 * K3);

            // return make_static_tile_distribution(
            //     StaticTileDistributionEncoding<Sequence<1>,
            //                                 Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2>>,
            //                                 Tuple<Sequence<2>, Sequence<2, 1>>,
            //                                 Tuple<Sequence<1>, Sequence<2, 0>>,
            //                                 Sequence<2, 1>,
            //                                 Sequence<0, 1>>{});
            return make_static_tile_distribution(
                StaticTileDistributionEncoding<Sequence<1>,
                                               Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2, K3>>,
                                               // Tuple<Sequence<2>, Sequence<2, 2, 1>>,
                                               // Tuple<Sequence<0>, Sequence<1, 2, 0>>,
                                               Tuple<Sequence<2>, Sequence<2, 1, 2>>,
                                               Tuple<Sequence<0>, Sequence<1, 0, 2>>,
                                               Sequence<2, 1>,
                                               Sequence<3, 1>>{});
        }
        else
        {
            constexpr index_t K1 = 16 / sizeof(VDataType);
            constexpr index_t K0 = kKPerBlock / K1;
            constexpr index_t N2 = get_warp_size() / K0;
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
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto MakeShuffledVRegBlockDescriptor()
    {
        // This descriptor only used when V layout is seqlen * hdim
        using VLayout = remove_cvref_t<typename Problem::BlockFmhaShape::VLayout>;
        static_assert(ck::is_same_v<VLayout, ck::tensor_layout::gemm::RowMajor>);
        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN1;
        constexpr index_t kKPerBlock = Problem::BlockFmhaShape::kK1;

        constexpr index_t N1 = 4;               // Y, TODO: vector load
        constexpr index_t N0 = kNPerBlock / N1; // P
        // constexpr index_t K2 = get_warp_size() / N0;        // P
        // constexpr index_t K1 = kBlockSize / get_warp_size(); // P
        // constexpr index_t K0 = kKPerBlock / (K1 * K2);          // Y
        constexpr index_t total_pixels = kNPerBlock * kKPerBlock / kBlockSize;
        static_assert(total_pixels % N1 == 0);       // TODO: this is not always true?
        constexpr index_t K3    = total_pixels / N1; //      kKPerBlock / (K0 * K1);          // Y
        constexpr index_t KPack = GetSmemKPackV<Problem>();
        static_assert(KPack % K3 == 0);
        constexpr index_t K2 = KPack / K3; // TODO: this dimention could be outside single wave
        constexpr index_t K1 = get_warp_size() / (K2 * N0);  // P
        constexpr index_t K0 = kBlockSize / get_warp_size(); // P

        // return make_static_tile_distribution(
        //         StaticTileDistributionEncoding<Sequence<1>,
        //                                     Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2>>,
        //                                     Tuple<Sequence<2>, Sequence<2, 1>>,
        //                                     Tuple<Sequence<1>, Sequence<2, 0>>,
        //                                     Sequence<1, 2>,
        //                                     Sequence<1, 0>>{});
        return make_static_tile_distribution(
            StaticTileDistributionEncoding<Sequence<1>,
                                           Tuple<Sequence<N0, N1>, Sequence<K0, K1, K2, K3>>,
                                           // Tuple<Sequence<2>, Sequence<2, 2, 1>>,
                                           // Tuple<Sequence<0>, Sequence<1, 2, 0>>,
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
                                     typename Problem::SaccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::BlockFmhaShape::kM0,
                                                   Problem::BlockFmhaShape::kN0,
                                                   Problem::BlockFmhaShape::kK0>>;

        using WarpGemm = warp::WarpGemmImpl<
            warp::WarpGemmAtrributeMfmaIterateKAndTransposedCDistribution_SwizzleB<
                warp::WarpGemmAttributeMfmaImplF16F16F32M32N32K8,
                2>>;

        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV1CustomPolicy<typename Problem::QDataType,
                                                 typename Problem::KDataType,
                                                 typename Problem::SaccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm0BlockWarps,
                                                 WarpGemm>;

        return BlockGemmARegBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    __host__ __device__ static constexpr auto GetKVBlockGemm()
    {
        using BlockGemmProblem =
            BlockGemmPipelineProblem<typename Problem::PDataType,
                                     typename Problem::VDataType,
                                     typename Problem::OaccDataType,
                                     Problem::kBlockSize,
                                     TileGemmShape<Problem::BlockFmhaShape::kM0,
                                                   Problem::BlockFmhaShape::kN1,
                                                   Problem::BlockFmhaShape::kK1>>;

        using WarpGemm = ck::tile_program::warp::WarpGemmMfmaDispatcher<
            typename Problem::PDataType,
            typename Problem::VDataType,
            typename Problem::OaccDataType,
            Problem::BlockFmhaShape::Gemm1WarpTile::At(Number<0>{}),
            Problem::BlockFmhaShape::Gemm1WarpTile::At(Number<1>{}),
            Problem::BlockFmhaShape::Gemm1WarpTile::At(Number<2>{}),
            true>;
        using BlockGemmPolicy =
            BlockGemmARegBSmemCRegV1CustomPolicy<typename Problem::PDataType,
                                                 typename Problem::VDataType,
                                                 typename Problem::OaccDataType,
                                                 typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                 WarpGemm>;
        return BlockGemmARegBSmemCRegV1<BlockGemmProblem, BlockGemmPolicy>{};
    }
};

} // namespace block
} // namespace tile_program
} // namespace ck
