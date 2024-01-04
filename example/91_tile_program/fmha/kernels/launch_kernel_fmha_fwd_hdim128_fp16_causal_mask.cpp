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
#include "launch_kernel_helper.hpp"
#include "macro_utils.hpp"

#include "fmha_fwd_kernel_selector.inc"

// clang-format off
// Head Dim = 128, DataType = fp16, Causal Mask
INST_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, true , ck::tile_program::block::GenericAttentionMask<true, false>, true >));
INST_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, true , ck::tile_program::block::GenericAttentionMask<true, false>, false>));
INST_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, false, ck::tile_program::block::GenericAttentionMask<true, false>, true >));
INST_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, false, ck::tile_program::block::GenericAttentionMask<true, false>, false>));
// clang-format on
