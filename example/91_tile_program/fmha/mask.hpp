// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <ostream>
#include <string>

#include "ck/ck.hpp"
#include "ck/tile_program/block_tile/block_masking.hpp"

struct mask_info
{
    using MaskType = ck::tile_program::block::GenericAttentionMaskType;

    MaskType type;
    ck::index_t left_size, right_size;
    ck::index_t y, x;

    void serialize(std::ostream& os) const
    {
        if(type == MaskType::NoMask)
            os << "n";
        else if(type == MaskType::CausalTopLeft)
            os << "tl";
        else if(type == MaskType::CausalBottomRight)
            os << "br";
        else
        {
            os << "g(" << y << "/" << x << ")";
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const mask_info& mi);
};

std::ostream& operator<<(std::ostream& os, const mask_info& mi)
{
    mi.serialize(os);
    return os;
}

mask_info decode_mask_info(std::string str, ck::index_t seqlen_q, ck::index_t seqlen_k)
{
    ck::index_t x_total = seqlen_k;
    ck::index_t y_total = seqlen_q;
    mask_info tmp;
    auto found_0 = str.find(':');
    if(found_0 != std::string::npos)
    {
        std::string t = str.substr(0, found_0);
        std::string v = str.substr(found_0 + 1);
        auto found_1  = v.find(",");
        if(found_1 == std::string::npos)
        {
            printf("not supported value %s, %s\n", v.c_str(), str.c_str());
            assert(0);
        }
        tmp.type       = mask_info::MaskType::WindowGeneric;
        tmp.left_size  = atoi(v.substr(0, found_1).c_str());
        tmp.right_size = atoi(v.substr(found_1 + 1).c_str());

        // TODO: some validation
        if(t == "t" || t == "b")
        {
            auto r = ck::make_generic_attention_mask_coordinate_from_lr_window(
                tmp.left_size, tmp.right_size, y_total, x_total, t == "t");
            tmp.y = r.At(ck::Number<0>{});
            tmp.x = r.At(ck::Number<1>{});
        }
        else if(t == "g")
        {
            tmp.y = tmp.left_size;
            tmp.x = tmp.right_size;
        }
        else
        {
            printf("not supported type %s, %s\n", t.c_str(), str.c_str());
            assert(0);
        }
    }
    else
    {
        // should be 0, 1, 2
        tmp.type = static_cast<mask_info::MaskType>(atoi(str.c_str()));
        if(tmp.type == mask_info::MaskType::NoMask)
        {
            tmp.left_size  = -1;
            tmp.right_size = -1;

            tmp.y = 0;
            tmp.x = 0;
        }
        else if(tmp.type == mask_info::MaskType::CausalTopLeft)
        {
            tmp.left_size  = -1;
            tmp.right_size = 0;

            tmp.y = seqlen_q;
            tmp.x = 1;
        }
        else if(tmp.type == mask_info::MaskType::CausalBottomRight)
        {
            tmp.left_size  = -1;
            tmp.right_size = 0;

            tmp.y = seqlen_q;
            tmp.x = seqlen_k - seqlen_q + 1;
        }
    }
    return tmp;
}
