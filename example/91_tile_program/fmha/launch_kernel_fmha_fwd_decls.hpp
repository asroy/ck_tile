// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"

#include "fmha_fwd_kernel_selector.hpp"
#include "launch_kernel_helper.hpp"

// clang-format off
// Head Dim = 128, DataType = fp16, No Mask
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, true , FmhaMasks::NoMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, true , FmhaMasks::NoMask, false>));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, false, FmhaMasks::NoMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, false, FmhaMasks::NoMask, false>));
// Head Dim = 128, DataType = fp16, Generic Mask
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, true , FmhaMasks::GenericMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, true , FmhaMasks::GenericMask, false>));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, false, FmhaMasks::GenericMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, false, FmhaMasks::GenericMask, false>));
// Head Dim = 128, DataType = fp16, Causal Mask
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, true , FmhaMasks::CausalMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, true , FmhaMasks::CausalMask, false>));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, false, FmhaMasks::CausalMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::half_t, false, FmhaMasks::CausalMask, false>));

// Head Dim = 64, DataType = fp16, No Mask
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::half_t, true , FmhaMasks::NoMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::half_t, true , FmhaMasks::NoMask, false>));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::half_t, false, FmhaMasks::NoMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::half_t, false, FmhaMasks::NoMask, false>));
// Head Dim = 64, DataType = fp16, Generic Mask
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::half_t, true , FmhaMasks::GenericMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::half_t, true , FmhaMasks::GenericMask, false>));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::half_t, false, FmhaMasks::GenericMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::half_t, false, FmhaMasks::GenericMask, false>));
// Head Dim = 64, DataType = fp16, Causal Mask
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::half_t, true , FmhaMasks::CausalMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::half_t, true , FmhaMasks::CausalMask, false>));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::half_t, false, FmhaMasks::CausalMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::half_t, false, FmhaMasks::CausalMask, false>));

// Head Dim = 128, DataType = bf16, No Mask
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::bhalf_t, true , FmhaMasks::NoMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::bhalf_t, true , FmhaMasks::NoMask, false>));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::bhalf_t, false, FmhaMasks::NoMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::bhalf_t, false, FmhaMasks::NoMask, false>));
// Head Dim = 128, DataType = bf16, Generic Mask
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::bhalf_t, true , FmhaMasks::GenericMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::bhalf_t, true , FmhaMasks::GenericMask, false>));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::bhalf_t, false, FmhaMasks::GenericMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::bhalf_t, false, FmhaMasks::GenericMask, false>));
// Head Dim = 128, DataType = bf16, Causal Mask
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::bhalf_t, true , FmhaMasks::CausalMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::bhalf_t, true , FmhaMasks::CausalMask, false>));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::bhalf_t, false, FmhaMasks::CausalMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<128, ck::bhalf_t, false, FmhaMasks::CausalMask, false>));

// Head Dim = 64, DataType = bf16, No Mask
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::bhalf_t, true , FmhaMasks::NoMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::bhalf_t, true , FmhaMasks::NoMask, false>));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::bhalf_t, false, FmhaMasks::NoMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::bhalf_t, false, FmhaMasks::NoMask, false>));
// Head Dim = 64, DataType = bf16, Generic Mask
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::bhalf_t, true , FmhaMasks::GenericMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::bhalf_t, true , FmhaMasks::GenericMask, false>));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::bhalf_t, false, FmhaMasks::GenericMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::bhalf_t, false, FmhaMasks::GenericMask, false>));
// Head Dim = 64, DataType = bf16, Causal Mask
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::bhalf_t, true , FmhaMasks::CausalMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::bhalf_t, true , FmhaMasks::CausalMask, false>));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::bhalf_t, false, FmhaMasks::CausalMask, true >));
DECL_LAUNCH_KERNEL((FmhaFwdKernelSelector<64, ck::bhalf_t, false, FmhaMasks::CausalMask, false>));
// clang-format on
