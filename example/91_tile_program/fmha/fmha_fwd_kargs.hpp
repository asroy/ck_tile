// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

template <ck::index_t I> // to avoid duplicated base class prblem, introduce an template arg
struct FmhaFwdEmptyKargs
{
};

// kargs use aggregate initializer, so no constructor will provided
// use inheritance to minimize karg size
// user need to use MakeKargs() function to create kargs.
struct FmhaFwdCommonKargs
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    void* o_ptr;

    ck::index_t seqlen_q;
    ck::index_t seqlen_k;
    ck::index_t hdim_q;
    ck::index_t hdim_v;

    // for MQA/GQA, nhead could be different. This parameter is nhead_q / nhead_k
    // if this param is larger than 1, indicate MQA/GQA case
    ck::index_t nhead_ratio_qk;
    float scale;

    ck::index_t stride_q;
    ck::index_t stride_k;
    ck::index_t stride_v;
    ck::index_t stride_o;

    ck::index_t nhead_stride_q;
    ck::index_t nhead_stride_k;
    ck::index_t nhead_stride_v;
    ck::index_t nhead_stride_o;
};

struct FmhaFwdCommonBiasKargs
{
    const void* bias_ptr  = nullptr;
    ck::index_t stride_bias       = 0;
    ck::index_t nhead_stride_bias = 0;
};

struct FmhaFwdBatchModeBiasKargs : FmhaFwdCommonBiasKargs
{
    ck::index_t batch_stride_bias = 0;
};

struct FmhaFwdMaskKargs
{
    ck::index_t mask_y, mask_x;
};

template <bool kHasBias, bool kHasMask>
struct FmhaFwdBatchModeKargs : FmhaFwdCommonKargs,
                        std::conditional_t<kHasBias, FmhaFwdBatchModeBiasKargs, FmhaFwdEmptyKargs<0>>,
                        std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<1>>
{
    ck::index_t batch_stride_q;
    ck::index_t batch_stride_k;
    ck::index_t batch_stride_v;
    ck::index_t batch_stride_o;
};

template <bool kHasBias, bool kHasMask>
struct FmhaFwdGroupModeKargs : FmhaFwdCommonKargs,
                        std::conditional_t<kHasBias, FmhaFwdCommonBiasKargs, FmhaFwdEmptyKargs<0>>,
                        std::conditional_t<kHasMask, FmhaFwdMaskKargs, FmhaFwdEmptyKargs<1>>
{
    const int32_t* seqstart_q_ptr;
    const int32_t* seqstart_k_ptr;
    const int32_t* seqlen_k_ptr;
};

template <bool kIsGroupMode, bool kHasBias, bool kHasMask>
using FmhaFwdKargs = std::conditional_t<
  kIsGroupMode,
  FmhaFwdGroupModeKargs<kHasBias, kHasMask>,
  FmhaFwdBatchModeKargs<kHasBias, kHasMask>
>;
