// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/utility/data_type.hpp"

#include "ck/tile_program/block_tile/block_masking.hpp"

#include "fmha_fwd_kernel_selector.hpp"
#include "launch_kernel_helper.hpp"

// clang-format off
// Head Dim = 128, DataType = fp16, Generic Mask
INST_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, true , ck::tile_program::block::GenericAttentionMask<true, true>, true >));
INST_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, true , ck::tile_program::block::GenericAttentionMask<true, true>, false>));
INST_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, false, ck::tile_program::block::GenericAttentionMask<true, true>, true >));
INST_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, false, ck::tile_program::block::GenericAttentionMask<true, true>, false>));
// clang-format on
