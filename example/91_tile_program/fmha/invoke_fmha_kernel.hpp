// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"

#include "invoke_fmha_kernel_fwd.hpp"

template <typename FmhaKernel>
float invoke_fmha_kernel(const void* q_ptr,
                         const void* k_ptr,
                         const void* v_ptr,
                         const void* bias_ptr,
                         void* o_ptr,
                         const void* seqstart_q_ptr,
                         const void* seqstart_k_ptr,
                         const void* seqlen_k_ptr,
                         ck::index_t batch,
                         ck::index_t nhead,
                         ck::index_t nhead_k,
                         ck::index_t seqlen_q,
                         ck::index_t seqlen_k,
                         ck::index_t hdim_q,
                         ck::index_t hdim_v,
                         ck::index_t max_seqlen_q,
                         float scale,
                         bool i_perm,
                         bool o_perm,
                         ck::index_t mask_y,
                         ck::index_t mask_x,
                         StreamConfig stream_config)
{
    constexpr bool is_v_rowmajor =
        ck::is_same_v<typename FmhaKernel::VLayout, ck::tensor_layout::gemm::RowMajor>;

    assert(nhead % nhead_k == 0);
    /// NOTE: we broadcast bias from [1, 1, seqlen_q, seqlen_k] to [batch, nhead, seqlen_q,
    ///       seqlen_k] in this example, hence both the 'batch_stride_bias' & 'nhead_stride_bias'
    ///       are 0.
    // setup stride_* arguments
    const ck::index_t stride_q = (i_perm ? hdim_q : nhead * hdim_q);
    const ck::index_t stride_k = (i_perm ? hdim_q : nhead_k * hdim_q);
    const ck::index_t stride_v = [&]() {
        if constexpr(is_v_rowmajor)
            return i_perm ? hdim_v : nhead_k * hdim_v;
        else
            return i_perm ? seqlen_k : nhead_k * seqlen_k;
    }();
    const ck::index_t stride_bias = (i_perm ? seqlen_k : 1 * seqlen_k);
    const ck::index_t stride_o    = (o_perm ? hdim_v : nhead * hdim_v);
    // setup nhead_stride_* arguments
    const ck::index_t nhead_stride_q = (i_perm ? seqlen_q * hdim_q : hdim_q);
    const ck::index_t nhead_stride_k = (i_perm ? seqlen_k * hdim_q : hdim_q);
    const ck::index_t nhead_stride_v = [&]() {
        if constexpr(is_v_rowmajor)
            return i_perm ? seqlen_k * hdim_v : hdim_v;
        else
            return i_perm ? hdim_v * seqlen_k : seqlen_k;
    }();
    const ck::index_t nhead_stride_bias = (i_perm ? 0 * seqlen_q * seqlen_k : 0 * seqlen_k);
    const ck::index_t nhead_stride_o    = (o_perm ? seqlen_q * hdim_v : hdim_v);
    // setup batch_stride_* arguments
    const ck::index_t batch_stride_q    = (nhead * seqlen_q * hdim_q);
    const ck::index_t batch_stride_k    = (nhead_k * seqlen_k * hdim_q);
    const ck::index_t batch_stride_v    = (nhead_k * hdim_v * seqlen_k);
    const ck::index_t batch_stride_bias = (0 * nhead * seqlen_q * seqlen_k);
    const ck::index_t batch_stride_o    = (nhead * seqlen_q * hdim_v);

    const auto kargs = [&] {
        // create group mode kernel arguments
        if constexpr(FmhaKernel::kIsGroupMode)
        {
            return FmhaKernel::MakeKargs(q_ptr,
                                         k_ptr,
                                         v_ptr,
                                         bias_ptr,
                                         o_ptr,
                                         seqstart_q_ptr,
                                         seqstart_k_ptr,
                                         seqlen_k_ptr,
                                         hdim_q,
                                         hdim_v,
                                         nhead / nhead_k,
                                         scale,
                                         stride_q,
                                         stride_k,
                                         stride_v,
                                         stride_bias,
                                         stride_o,
                                         nhead_stride_q,
                                         nhead_stride_k,
                                         nhead_stride_v,
                                         nhead_stride_bias,
                                         nhead_stride_o,
                                         mask_y,
                                         mask_x);
        }
        else
        { // create batch mode kernel arguments
            return FmhaKernel::MakeKargs(q_ptr,
                                         k_ptr,
                                         v_ptr,
                                         bias_ptr,
                                         o_ptr,
                                         seqlen_q,
                                         seqlen_k,
                                         hdim_q,
                                         hdim_v,
                                         nhead / nhead_k,
                                         scale,
                                         stride_q,
                                         stride_k,
                                         stride_v,
                                         stride_bias,
                                         stride_o,
                                         nhead_stride_q,
                                         nhead_stride_k,
                                         nhead_stride_v,
                                         nhead_stride_bias,
                                         nhead_stride_o,
                                         batch_stride_q,
                                         batch_stride_k,
                                         batch_stride_v,
                                         batch_stride_bias,
                                         batch_stride_o,
                                         mask_y,
                                         mask_x);
        }
    }();

    const dim3 kGridSize      = FmhaKernel::GridSize(batch, nhead, max_seqlen_q, hdim_v);
    constexpr dim3 kBlockSize = FmhaKernel::BlockSize();

    constexpr ck::index_t kBlockPerCu = FmhaKernel::kBlockPerCu;

    return launch_kernel<kBlockSize.x, kBlockPerCu>(stream_config,
                                                    FmhaKernel{},
                                                    kGridSize,
                                                    kBlockSize,
                                                    0,
                                                    kargs); // BatchStrideO
}
