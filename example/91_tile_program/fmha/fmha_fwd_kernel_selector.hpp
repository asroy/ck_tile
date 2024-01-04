// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "ck/tile_program/block_tile/block_masking.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_problem.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs.hpp"
#include "ck/tile_program/block_tile_pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp"
#include "ck/tile_program/tile/tile_fmha_shape.hpp"
#include "ck/tile_program/tile/tile_fmha_traits.hpp"

#include "fmha_fwd_epilogue.hpp"
#include "fmha_fwd_kernel.hpp"
#include "fmha_fwd_tile_partitioner.hpp"
#include "fmha_fwd_type_config.hpp"

// default settings for FmhaFwdKernelSelector<> type alias
using VLayout = ck::tensor_layout::gemm::RowMajor; // (bs, nhead) seqlen * hdim
// using VLayout = ck::tensor_layout::gemm::ColumnMajor; // (bs, nhead) hdim * seqlen

struct FmhaMaskType
{
    using NoMask      = ck::tile_program::block::GenericAttentionMask<false>;
    using GenericMask = ck::tile_program::block::GenericAttentionMask<true, true>;
    using CausalMask  = ck::tile_program::block::GenericAttentionMask<true, false>;
};

inline constexpr bool kM0NeedPadding   = false;
inline constexpr bool kN0K1NeedPadding = false;

template <ck::index_t HDim>
struct FmhaBlockTile;

template <>
struct FmhaBlockTile</* HDim = */ 64> : ck::Sequence<128, 64, 32, 64, 32, 64>
{
};
template <>
struct FmhaBlockTile</* HDim = */ 128> : ck::Sequence<128, 128, 32, 128, 32, 128>
{
};
using FmhaBlockWarps = ck::Sequence<4, 1, 1>;
using FmhaWarpTile   = ck::Sequence<32, 32, 16>;

template <ck::index_t HDim>
struct FmhaShape;

template <>
struct FmhaShape</* HDim = */ 64> : ck::tile_program::TileFmhaShape<FmhaBlockTile</* HDim = */ 64>,
                                                                    FmhaBlockWarps,
                                                                    FmhaWarpTile,
                                                                    FmhaBlockWarps,
                                                                    FmhaWarpTile,
                                                                    VLayout>
{
};

template <>
struct FmhaShape</* HDim = */ 128>
    : ck::tile_program::TileFmhaShape<FmhaBlockTile</* HDim = */ 128>,
                                      FmhaBlockWarps,
                                      FmhaWarpTile,
                                      FmhaBlockWarps,
                                      FmhaWarpTile,
                                      VLayout>
{
};

template <ck::index_t HDim, bool kHasBias>
using FmhaTraits = ck::tile_program::TileFmhaTraits<kM0NeedPadding,
                                                    kN0K1NeedPadding,
                                                    kHasBias,
                                                    HDim == 64 ? /* occupancy = */ 3 : 2>;

template <ck::index_t HDim, typename DataType, bool kIsGroupMode, typename FmhaMask, bool kHasBias>
using FmhaPipelineProblem = ck::tile_program::block::BlockFmhaPipelineProblem<
    typename FmhaFwdTypeConfig<DataType>::QDataType,
    typename FmhaFwdTypeConfig<DataType>::KDataType,
    typename FmhaFwdTypeConfig<DataType>::VDataType,
    typename FmhaFwdTypeConfig<DataType>::SaccDataType,
    typename FmhaFwdTypeConfig<DataType>::SMPLComputeDataType,
    typename FmhaFwdTypeConfig<DataType>::BiasDataType,
    typename FmhaFwdTypeConfig<DataType>::PDataType,
    typename FmhaFwdTypeConfig<DataType>::OaccDataType,
    typename FmhaFwdTypeConfig<DataType>::ODataType,
    /* BlockSize = */ 256,
    FmhaShape<HDim>,
    kIsGroupMode,
    FmhaMask,
    FmhaTraits<HDim, kHasBias>>;

template <ck::index_t HDim, typename DataType, bool kIsGroupMode, typename FmhaMask, bool kHasBias>
using FmhaPipeline = ck::tile_program::block::BlockFmhaPipelineQRKSVSAsync<
    FmhaPipelineProblem<HDim, DataType, kIsGroupMode, FmhaMask, kHasBias>>;

template <typename DataType>
using FmhaEpilogue =
    FmhaFwdEpilogue<FmhaFwdEpilogueProblem<typename FmhaFwdTypeConfig<DataType>::OaccDataType,
                                           typename FmhaFwdTypeConfig<DataType>::ODataType>>;

template <ck::index_t HDim, typename DataType, bool kIsGroupMode, typename FmhaMask, bool kHasBias>
using FmhaFwdKernelSelector =
    FmhaFwdKernel<FmhaFwdTilePartitioner<FmhaShape<HDim>>,
                  FmhaPipeline<HDim, DataType, kIsGroupMode, FmhaMask, kHasBias>,
                  FmhaEpilogue<DataType>>;
