// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/utility/data_type.hpp"

#include "ck/tile_program/block_tile/block_masking.hpp"

#include "fmha_fwd_kernel_selector.hpp"
#include "launch_kernel_helper.hpp"

// clang-format off
// Head Dim = 64, DataType = bf16, No Mask
INST_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::bhalf_t, true , FmhaMasks::NoMask, true >));
INST_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::bhalf_t, true , FmhaMasks::NoMask, false>));
INST_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::bhalf_t, false, FmhaMasks::NoMask, true >));
INST_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::bhalf_t, false, FmhaMasks::NoMask, false>));
// clang-format on
