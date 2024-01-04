// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/host_utility/kernel_launch.hpp"

#include "macro_utils.hpp"

#define DECL_LAUNCH_KERNEL(kernel_type)                                        \
    extern template float launch_kernel<PP_UNWRAP(kernel_type)::BlockSize().x, \
                                        PP_UNWRAP(kernel_type)::kBlockPerCu,   \
                                        PP_UNWRAP(kernel_type),                \
                                        PP_UNWRAP(kernel_type)::Kargs>(        \
        const StreamConfig&,                                                   \
        PP_UNWRAP(kernel_type),                                                \
        dim3,                                                                  \
        dim3,                                                                  \
        std::size_t,                                                           \
        PP_UNWRAP(kernel_type)::Kargs)

#define INST_LAUNCH_KERNEL(kernel_type)                                                 \
    template float launch_kernel<PP_UNWRAP(kernel_type)::BlockSize().x,                 \
                                 PP_UNWRAP(kernel_type)::kBlockPerCu,                   \
                                 PP_UNWRAP(kernel_type),                                \
                                 PP_UNWRAP(kernel_type)::Kargs>(const StreamConfig&,    \
                                                                PP_UNWRAP(kernel_type), \
                                                                dim3,                   \
                                                                dim3,                   \
                                                                std::size_t,            \
                                                                PP_UNWRAP(kernel_type)::Kargs)
