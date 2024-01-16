// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <optional>
#include <ostream>
#include <string>
#include <stdexcept>

#include "ck/ck.hpp"
#include "ck/tile_program/block_tile/block_masking.hpp"

struct mask_info
{
    using MaskType = ck::tile_program::block::GenericAttentionMaskType;

    MaskType type;
    ck::index_t left_size, right_size;
    ck::index_t y, x; // only used on host

    void serialize(std::ostream& os) const
    {
        if(type == MaskType::CausalTopLeft)
            os << "tl";
        else if(type == MaskType::CausalBottomRight)
            os << "br";
        else // CausalMaskDisabled
        {
            os << "g(" << y << "/" << x << ")";
        }
    }

    friend std::optional<mask_info>
    decode_mask_info(std::string str, ck::index_t seqlen_q, ck::index_t seqlen_k)
    {
        ck::index_t x_total = seqlen_k;
        ck::index_t y_total = seqlen_q;
        mask_info mask;

        auto found_0 = str.find(':');
        if(found_0 != std::string::npos)
        {
            std::string t = str.substr(0, found_0);
            std::string v = str.substr(found_0 + 1);
            auto found_1  = v.find(",");
            if(found_1 == std::string::npos)
            {
                printf("not supported value %s, %s\n", v.c_str(), str.c_str());
                throw std::invalid_argument(
                    std::string("cannot construct mask_info from string: ") + str);
            }
            mask.type      = mask_info::MaskType::CausalMaskDisabled;
            ck::index_t v0 = atoi(v.substr(0, found_1).c_str());
            ck::index_t v1 = atoi(v.substr(found_1 + 1).c_str());

            // TODO: some validation
            if(t == "t" || t == "b")
            {
                mask.left_size  = v0;
                mask.right_size = v1;

                auto r = ck::make_generic_attention_mask_coordinate_from_lr_window(
                    mask.left_size, mask.right_size, y_total, x_total, t == "t");
                mask.y = r.At(ck::Number<0>{});
                mask.x = r.At(ck::Number<1>{});
            }
            else if(t == "g")
            {
                mask.left_size  = v0 - 1;
                mask.right_size = v1 - 1;

                mask.y = v0;
                mask.x = v1;
            }
            else
            {
                printf("not supported type %s, %s\n", t.c_str(), str.c_str());
                throw std::invalid_argument(
                    std::string("cannot construct mask_info from string: ") + str);
            }

            return mask;
        }
        else
        {
            // should be 0, 1, 2
            const auto chosen_type = static_cast<int>(atoi(str.c_str()));
            if(chosen_type == 0)
            {
                return std::nullopt;
            }

            mask.type = static_cast<mask_info::MaskType>(chosen_type);
            if(mask.type == mask_info::MaskType::CausalTopLeft ||
               mask.type == mask_info::MaskType::CausalBottomRight)
            {
                mask.left_size  = -1;
                mask.right_size = 0;

                auto r = ck::make_generic_attention_mask_coordinate_from_lr_window(
                    mask.left_size,
                    mask.right_size,
                    y_total,
                    x_total,
                    mask.type == mask_info::MaskType::CausalTopLeft);

                // (y, x) values in each cases:
                //   CausalTopLeft: (y, x) = (seqlen_q, 1)
                //   CausalBottomRight: (y, x) = (seqlen_q, seqlen_k - seqlen_q + 1)
                mask.y = r.At(ck::Number<0>{});
                mask.x = r.At(ck::Number<1>{});
            }
            else
            {
                printf("not supported type %d\n", chosen_type);
                throw std::invalid_argument("cannot construct mask_info from type: " +
                                            std::to_string(chosen_type));
            }

            return mask;
        }

        return std::nullopt;
    }

    friend std::ostream& operator<<(std::ostream& os, const mask_info& mi)
    {
        mi.serialize(os);
        return os;
    }

    friend std::ostream& operator<<(std::ostream& os, const std::optional<mask_info>& mi)
    {
        if(mi.has_value())
        {
            return os << *mi;
        }
        return os << "n";
    }
};

std::optional<mask_info>
decode_mask_info(std::string str, ck::index_t seqlen_q, ck::index_t seqlen_k);
