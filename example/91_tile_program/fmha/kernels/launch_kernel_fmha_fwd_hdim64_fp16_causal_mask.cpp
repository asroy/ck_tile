// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/utility/data_type.hpp"

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
#include "macro.hpp"

#include "fmha_fwd_kernel_selector.inc"

#define DEFINE_FMHA_KERNEL_INVOKE_FUNC(kernel)                     \
    template float launch_kernel<PP_UNWRAP(kernel)::BlockSize().x, \
                                 PP_UNWRAP(kernel)::kBlockPerCu,   \
                                 PP_UNWRAP(kernel),                \
                                 PP_UNWRAP(kernel)::Kargs>(        \
        const StreamConfig&, PP_UNWRAP(kernel), dim3, dim3, std::size_t, PP_UNWRAP(kernel)::Kargs)

// clang-format off
// Head Dim = 64, DataType = fp16, Causal Mask
DEFINE_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::half_t, true , ck::tile_program::block::GenericAttentionMask<true, false>, true >));
DEFINE_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::half_t, true , ck::tile_program::block::GenericAttentionMask<true, false>, false>));
DEFINE_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::half_t, false, ck::tile_program::block::GenericAttentionMask<true, false>, true >));
DEFINE_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::half_t, false, ck::tile_program::block::GenericAttentionMask<true, false>, false>));
// clang-format on
