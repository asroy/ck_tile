// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/ck.hpp"
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
#include "invoke_fmha_kernel.hpp"
#include "macro.hpp"

#include "fmha_fwd_kernel_selector.inc"

#define DEFINE_FMHA_KERNEL_INVOKE_FUNC(kernel)                                       \
    template float invoke_fmha_kernel<PP_UNWRAP(kernel)>(const void* q_ptr,          \
                                                         const void* k_ptr,          \
                                                         const void* v_ptr,          \
                                                         const void* bias_ptr,       \
                                                         void* o_ptr,                \
                                                         const void* seqstart_q_ptr, \
                                                         const void* seqstart_k_ptr, \
                                                         const void* seqlen_k_ptr,   \
                                                         ck::index_t batch,          \
                                                         ck::index_t nhead,          \
                                                         ck::index_t nhead_k,        \
                                                         ck::index_t seqlen_q,       \
                                                         ck::index_t seqlen_k,       \
                                                         ck::index_t hdim_q,         \
                                                         ck::index_t hdim_v,         \
                                                         ck::index_t max_seqlen_q,   \
                                                         float scale,                \
                                                         bool i_perm,                \
                                                         bool o_perm,                \
                                                         ck::index_t mask_y,         \
                                                         ck::index_t mask_x,         \
                                                         StreamConfig stream_config)

// clang-format off
// Head Dim = 128, DataType = bf16, Generic Mask
DEFINE_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::bhalf_t, true , ck::tile_program::block::GenericAttentionMask<true, true>, true >));
DEFINE_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::bhalf_t, true , ck::tile_program::block::GenericAttentionMask<true, true>, false>));
DEFINE_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::bhalf_t, false, ck::tile_program::block::GenericAttentionMask<true, true>, true >));
DEFINE_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::bhalf_t, false, ck::tile_program::block::GenericAttentionMask<true, true>, false>));
// clang-format on
