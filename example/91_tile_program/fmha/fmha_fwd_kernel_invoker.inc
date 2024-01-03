// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "fmha_fwd_kernel_selector.inc"

#define DECL_FMHA_KERNEL_INVOKE_FUNC(kernel)                                                \
    extern template float invoke_fmha_kernel<PP_UNWRAP(kernel)>(const void* q_ptr,          \
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
// Head Dim = 128, DataType = fp16, No Mask
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::half_t, true , ck::tile_program::block::GenericAttentionMask<false>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::half_t, true , ck::tile_program::block::GenericAttentionMask<false>, false>));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::half_t, false, ck::tile_program::block::GenericAttentionMask<false>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::half_t, false, ck::tile_program::block::GenericAttentionMask<false>, false>));
// Head Dim = 128, DataType = fp16, Generic Mask
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::half_t, true , ck::tile_program::block::GenericAttentionMask<true, true >, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::half_t, true , ck::tile_program::block::GenericAttentionMask<true, true >, false>));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::half_t, false, ck::tile_program::block::GenericAttentionMask<true, true >, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::half_t, false, ck::tile_program::block::GenericAttentionMask<true, true >, false>));
// Head Dim = 128, DataType = fp16, Causal Mask
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::half_t, true , ck::tile_program::block::GenericAttentionMask<true, false>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::half_t, true , ck::tile_program::block::GenericAttentionMask<true, false>, false>));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::half_t, false, ck::tile_program::block::GenericAttentionMask<true, false>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::half_t, false, ck::tile_program::block::GenericAttentionMask<true, false>, false>));

// Head Dim = 64, DataType = fp16, No Mask
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::half_t, true , ck::tile_program::block::GenericAttentionMask<false>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::half_t, true , ck::tile_program::block::GenericAttentionMask<false>, false>));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::half_t, false, ck::tile_program::block::GenericAttentionMask<false>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::half_t, false, ck::tile_program::block::GenericAttentionMask<false>, false>));
// Head Dim = 64, DataType = fp16, Generic Mask
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::half_t, true , ck::tile_program::block::GenericAttentionMask<true, true>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::half_t, true , ck::tile_program::block::GenericAttentionMask<true, true>, false>));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::half_t, false, ck::tile_program::block::GenericAttentionMask<true, true>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::half_t, false, ck::tile_program::block::GenericAttentionMask<true, true>, false>));
// Head Dim = 64, DataType = fp16, Causal Mask
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::half_t, true , ck::tile_program::block::GenericAttentionMask<true, false>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::half_t, true , ck::tile_program::block::GenericAttentionMask<true, false>, false>));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::half_t, false, ck::tile_program::block::GenericAttentionMask<true, false>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::half_t, false, ck::tile_program::block::GenericAttentionMask<true, false>, false>));

// Head Dim = 128, DataType = bf16, No Mask
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::bhalf_t, true , ck::tile_program::block::GenericAttentionMask<false>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::bhalf_t, true , ck::tile_program::block::GenericAttentionMask<false>, false>));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::bhalf_t, false, ck::tile_program::block::GenericAttentionMask<false>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::bhalf_t, false, ck::tile_program::block::GenericAttentionMask<false>, false>));
// Head Dim = 128, DataType = bf16, Generic Mask
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::bhalf_t, true , ck::tile_program::block::GenericAttentionMask<true, true >, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::bhalf_t, true , ck::tile_program::block::GenericAttentionMask<true, true >, false>));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::bhalf_t, false, ck::tile_program::block::GenericAttentionMask<true, true >, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::bhalf_t, false, ck::tile_program::block::GenericAttentionMask<true, true >, false>));
// Head Dim = 128, DataType = bf16, Causal Mask
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::bhalf_t, true , ck::tile_program::block::GenericAttentionMask<true, false>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::bhalf_t, true , ck::tile_program::block::GenericAttentionMask<true, false>, false>));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::bhalf_t, false, ck::tile_program::block::GenericAttentionMask<true, false>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<128, ck::bhalf_t, false, ck::tile_program::block::GenericAttentionMask<true, false>, false>));

// Head Dim = 64, DataType = bf16, No Mask
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::bhalf_t, true , ck::tile_program::block::GenericAttentionMask<false>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::bhalf_t, true , ck::tile_program::block::GenericAttentionMask<false>, false>));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::bhalf_t, false, ck::tile_program::block::GenericAttentionMask<false>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::bhalf_t, false, ck::tile_program::block::GenericAttentionMask<false>, false>));
// Head Dim = 64, DataType = bf16, Generic Mask
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::bhalf_t, true , ck::tile_program::block::GenericAttentionMask<true, true>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::bhalf_t, true , ck::tile_program::block::GenericAttentionMask<true, true>, false>));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::bhalf_t, false, ck::tile_program::block::GenericAttentionMask<true, true>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::bhalf_t, false, ck::tile_program::block::GenericAttentionMask<true, true>, false>));
// Head Dim = 64, DataType = bf16, Causal Mask
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::bhalf_t, true , ck::tile_program::block::GenericAttentionMask<true, false>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::bhalf_t, true , ck::tile_program::block::GenericAttentionMask<true, false>, false>));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::bhalf_t, false, ck::tile_program::block::GenericAttentionMask<true, false>, true >));
DECL_FMHA_KERNEL_INVOKE_FUNC((FmhaFwdKernelSelector<64, ck::bhalf_t, false, ck::tile_program::block::GenericAttentionMask<true, false>, false>));
// clang-format on

#undef DECL_FMHA_KERNEL_INVOKE_FUNC

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
                using FmhaMask = ck::tile_program::block::GenericAttentionMask<false>;
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
