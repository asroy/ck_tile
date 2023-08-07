// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/tensor_adaptor.hpp"

#include "tile_program.hpp"
#include "ck/tile_program/tile/tile_distribution.hpp"
#include "ck/tile_program/tile/tile_elementwise.hpp"
#include "ck/tile_program/tile/tile_gemm_shape.hpp"
#include "ck/tile_program/warp_tile/warp_gemm.hpp"
#include "ck/tile_program/block_tile_pipeline/block_gemm_pipeline_agmem_bgmem_creg_v2.hpp"
#include "ck/tile_program/block_tile/block_gemm_areg_bsmem_creg_v1.hpp"

// C1 = A0 * B0 * B1
template <typename A0DataType,
          typename B0DataType,
          typename Acc0DataType,
          typename C0DataType,
          typename B1DataType,
          typename Acc1DataType,
          typename C1DataType,
          ck::index_t kBlockSize,
          ck::index_t kM0PerBlock,
          ck::index_t kN0PerBlock,
          ck::index_t kK0PerBlock,
          ck::index_t kN1PerBlock>
struct GemmGemm
{
    // block gemm0 pipeline
    using BlockGemm0PipelineProblem =
        ck::tile_program::block::BlockGemmPipelineAGmemBGmemCRegV2Problem<
            A0DataType,
            B0DataType,
            Acc0DataType,
            kBlockSize,
            ck::tile_program::TileGemmShape<kM0PerBlock, kN0PerBlock, kK0PerBlock>>;

    using BlockGemm0PipelinePolicy =
        ck::tile_program::block::BlockGemmPipelineAGmemBGmemCRegV2DefaultPolicy;

    using BlockGemm0Pipeline =
        ck::tile_program::block::BlockGemmPipelineAGmemBGmemCRegV2<BlockGemm0PipelineProblem,
                                                                   BlockGemm0PipelinePolicy>;

    // block gemm1
    using BlockGemm1Problem = ck::tile_program::block::BlockGemmARegBSmemCRegV1Problem<
        C0DataType,
        B1DataType,
        Acc1DataType,
        kBlockSize,
        ck::tile_program::TileGemmShape<kM0PerBlock, kN1PerBlock, kN0PerBlock>>;

    using BlockGemm1Policy = ck::tile_program::block::BlockGemmARegBSmemCRegV1DefaultPolicy;

    using BlockGemm1 =
        ck::tile_program::block::BlockGemmARegBSmemCRegV1<BlockGemm1Problem, BlockGemm1Policy>;

    __host__ __device__ static constexpr ck::index_t GetStaticLdsSize()
    {
        return BlockGemm0Pipeline::GetStaticLdsSize();
    }

    __host__ __device__ void operator()(ProgramServer& ps,
                                        const A0DataType* p_a0,
                                        const B0DataType* p_b0,
                                        const B1DataType* p_b1,
                                        C1DataType* p_c1,
                                        ck::index_t M0,
                                        ck::index_t N0,
                                        ck::index_t K0,
                                        ck::index_t N1,
                                        ck::index_t Lda0,
                                        ck::index_t Ldb0,
                                        ck::index_t Ldb1,
                                        ck::index_t Ldc1)
    {
        using namespace ck;
        using namespace ck::tile_program;
        using namespace ck::tile_program::block;

        // FIXME: assume layout A0[M0, K0], B0[N0, K0], B1[N1, N0], C1[M0, N1]
        const auto a0_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_a0, make_tuple(M0, K0), make_tuple(Lda0, 1), Number<32>{}, Number<1>{});

        const auto b0_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b0, make_tuple(N0, K0), make_tuple(Ldb0, 1), Number<32>{}, Number<1>{});

        const auto b1_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_b1, make_tuple(N1, N0), make_tuple(Ldb1, 1), Number<32>{}, Number<1>{});

        // divide problem
        const auto id_block = ps.get_block_id();

        const auto num_tile_m0 = M0 / kM0PerBlock;
        const auto num_tile_n1 = N1 / kN1PerBlock;

        const auto block2tile = ps(make_cluster_descriptor(make_tuple(num_tile_m0, num_tile_n1)));

        const auto id_tile = block2tile.CalculateBottomIndex(make_tuple(id_block));

        const auto iM0 = ps.read_first_lane(id_tile.At<0>() * kM0PerBlock);
        const auto iN1 = ps.read_first_lane(id_tile.At<1>() * kN1PerBlock);

        // A0 DRAM block window
        auto a0_dram_block_window = make_tile_window(
            a0_dram_grid, make_tuple(Number<kM0PerBlock>{}, Number<kK0PerBlock>{}), {iM0, 0});

        // B0 DRAM block window
        auto b0_dram_block_window = make_tile_window(
            b0_dram_grid, make_tuple(Number<kN0PerBlock>{}, Number<kK0PerBlock>{}), {0, 0});

        // B1 window
        auto b1_dram_block_window = make_tile_window(
            b1_dram_grid, make_tuple(Number<kN1PerBlock>{}, Number<kN0PerBlock>{}), {iN1, 0});

        // Block GEMM0 pipeline
        constexpr auto block_gemm0_pipeline = BlockGemm0Pipeline{};

        // Bock GEMM1
        constexpr auto block_gemm1 = BlockGemm1{};

        __shared__ char p_smem_char[block_gemm0_pipeline.GetStaticLdsSize()];

        // Acc1 tile
        auto acc1_block_tile = decltype(block_gemm1(
            tile_elementwise_in(
                type_convert<C0DataType, Acc0DataType>,
                block_gemm0_pipeline(a0_dram_block_window, b0_dram_block_window, 0, nullptr)),
            b1_dram_block_window)){};

        // init Acc1
        tile_elementwise_inout([](auto& acc1) { acc1 = 0; }, acc1_block_tile);

        index_t iN0 = 0;

        do
        {
            // acc0 = a0 * b0
            const auto acc0_block_tile = block_gemm0_pipeline(
                a0_dram_block_window, b0_dram_block_window, K0 / kK0PerBlock, p_smem_char);

            // type cast acc0 into c0
            const auto c0_block_tile =
                tile_elementwise_in(type_convert<C0DataType, Acc0DataType>, acc0_block_tile);

            // acc1 += c0 * b1
            block_gemm1(acc1_block_tile, c0_block_tile, b1_dram_block_window);

            // move tile windows
            move_tile_window(b0_dram_block_window, {kN0PerBlock, 0});
            move_tile_window(b1_dram_block_window, {0, kN0PerBlock});

            iN0 += kN0PerBlock;

        } while(iN0 < N0);

        // type cast acc1 into c1
        const auto c1_block_tile =
            tile_elementwise_in(type_convert<C1DataType, Acc1DataType>, acc1_block_tile);

        // store c1
        auto c1_dram_grid = make_naive_tensor_view<AddressSpaceEnum::Global>(
            p_c1, make_tuple(M0, N1), make_tuple(Ldc1, 1), Number<32>{}, Number<1>{});

        auto c1_dram_window =
            make_tile_window(c1_dram_grid,
                             make_tuple(Number<kM0PerBlock>{}, Number<kN1PerBlock>{}),
                             {iM0, iN1},
                             c1_block_tile.GetTileDistribution());

        store_tile(c1_dram_window, c1_block_tile);
    }
};
