// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/data_type.hpp"
#include "ck/utility/static_switch.hpp"

#include "ck/tile_program/block_tile/block_masking.hpp"

#include "fmha_fwd_kernel_selector.hpp"
#include "invoke_fmha_kernel.hpp"
#include "mask.hpp"
#include "utils.hpp"

#include "launch_kernel_fmha_fwd_decls.hpp"

template <ck::index_t HDim_, typename DataType_>
struct fmha_fwd_kernel_invoker
{
    static constexpr ck::index_t HDim = HDim_;
    using DataType                    = DataType_;
    // these args are used to select kernel.
    // args that may passed as karg shoule use operator()
    mode_enum mode;
    bool use_bias;
    mask_info mask;

    fmha_fwd_kernel_invoker(mode_enum mode_, bool use_bias_, mask_info mask_)
        : mode(mode_), use_bias(use_bias_), mask(mask_)
    {
    }

    template <typename... Args>
    float operator()(Args&&... args)
    {
        float ave_time;
        BOOL_SWITCH_2(mode == mode_enum::group, kIsGroupMode, use_bias, kHasBias, [&] {
            if(mask.type == mask_enum::no_mask)
            {
                using FmhaMask = FmhaMasks::NoMask;
                using Kernel =
                    FmhaFwdKernelSelector<HDim, DataType, kIsGroupMode, FmhaMask, kHasBias>;

                ave_time = invoke_fmha_kernel<Kernel>(std::forward<Args>(args)...);
            }
            else
            {
                BOOL_SWITCH(mask.type == mask_enum::window_generic, kIsLocal, [&]() {
                    using FmhaMask = ck::tile_program::block::GenericAttentionMask<true, kIsLocal>;
                    using Kernel =
                        FmhaFwdKernelSelector<HDim, DataType, kIsGroupMode, FmhaMask, kHasBias>;

                    ave_time = invoke_fmha_kernel<Kernel>(std::forward<Args>(args)...);
                });
            }
        });
        return ave_time;
    }
};
