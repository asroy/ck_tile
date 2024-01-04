// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/utility/data_type.hpp"

#include "ck/tile_program/block_tile/block_masking.hpp"

#include "fmha_fwd_kernel_selector.hpp"
#include "launch_kernel_helper.hpp"

// clang-format off
// Head Dim = 64, DataType = fp16, Generic Mask
INST_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::half_t, true , FmhaMasks::GenericMask, true >));
INST_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::half_t, true , FmhaMasks::GenericMask, false>));
INST_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::half_t, false, FmhaMasks::GenericMask, true >));
INST_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::half_t, false, FmhaMasks::GenericMask, false>));
// clang-format on
