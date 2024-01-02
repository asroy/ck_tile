// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ostream>
#include <string>

#include "ck/ck.hpp"

enum class mask_enum
{
    no_mask = 0,
    causal_top_left,
    causal_bottom_right,
    window_generic,
};

struct mask_info
{
    mask_enum type;
    ck::index_t y, x;

    void serialize(std::ostream& os) const;

    friend std::ostream& operator<<(std::ostream& os, const mask_info& mi);
};

std::ostream& operator<<(std::ostream& os, const mask_info& mi);

mask_info decode_mask_info(std::string str, ck::index_t seqlen_q, ck::index_t seqlen_k);
