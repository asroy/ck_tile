// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "ck/stream_config.hpp"

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
                         StreamConfig stream_config);
