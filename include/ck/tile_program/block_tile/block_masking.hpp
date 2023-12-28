// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tile_program {
namespace block {

// clang-format off
/*  Generic Attention Mask Coordinate
    use x(horizontal axis), y(vertical axis) to describe mask.
    top-left corner is origin

    x=1/y=5(top-left)  x=4/y=5(botm-r)    x=6/y=5            x=8/y=5(no mask)
    1 * * * * * * *    1 1 1 1 * * * *    1 1 1 1 1 1 * *    1 1 1 1 1 1 1 1
    1 1 * * * * * *    1 1 1 1 1 * * *    1 1 1 1 1 1 1 *    1 1 1 1 1 1 1 1
    1 1 1 * * * * *    1 1 1 1 1 1 * *    1 1 1 1 1 1 1 1    1 1 1 1 1 1 1 1
    1 1 1 1 * * * *    1 1 1 1 1 1 1 *    1 1 1 1 1 1 1 1    1 1 1 1 1 1 1 1
    1 1 1 1 1 * * *    1 1 1 1 1 1 1 1    1 1 1 1 1 1 1 1    1 1 1 1 1 1 1 1
    l=7,-1/r=0(tl)     l=7,-1/r=0(br)

    x=1/y=2            x=4/y=2            x=6/y=2            x=8/y=2
    1 * * * * * * *    1 1 1 1 * * * *    1 1 1 1 1 1 * *    1 1 1 1 1 1 1 1
    1 1 * * * * * *    1 1 1 1 1 * * *    1 1 1 1 1 1 1 *    1 1 1 1 1 1 1 1
    * 1 1 * * * * *    * 1 1 1 1 1 * *    * 1 1 1 1 1 1 1    * 1 1 1 1 1 1 1
    * * 1 1 * * * *    * * 1 1 1 1 1 *    * * 1 1 1 1 1 1    * * 1 1 1 1 1 1
    * * * 1 1 * * *    * * * 1 1 1 1 1    * * * 1 1 1 1 1    * * * 1 1 1 1 1
    l=1/r=0(tl)        l=1/r=3(tl)        l=1/r=5(tl)        l=1/r=7(tl)
                       l=4/r=0(br)        l=4/r=2(br)        l=4/r=4(br)

                       x=4/y=-1           x=6/y=-1            x=8/y=-1
                       * * 1 1 * * * *    * * 1 1 1 1 * *    * * 1 1 1 1 1 1
                       * * * 1 1 * * *    * * * 1 1 1 1 *    * * * 1 1 1 1 1
                       * * * * 1 1 * *    * * * * 1 1 1 1    * * * * 1 1 1 1
                       * * * * * 1 1 *    * * * * * 1 1 1    * * * * * 1 1 1
                       * * * * * * 1 1    * * * * * * 1 1    * * * * * * 1 1

    x=-2/y=5           x=1/y=5(top-left)  x=0/y=5(botm-r)
    * * * * * * * *    1 * * *            * * * *
    * * * * * * * *    1 1 * *            1 * * *
    * * * * * * * *    1 1 1 *            1 1 * *
    1 * * * * * * *    1 1 1 1            1 1 1 *
    1 1 * * * * * *    1 1 1 1            1 1 1 1

    Validations:
        x + y > 1 (x + y >= 2)

    Note:
        y = seq_q, x = 1 -> top-left
        y = seq_q, x = seq_k - seq_q + 1 -> bottom-right
        y < seq_q, x < seq_k -> local-attn
        y = seq_q, x = seq_k -> no mask

*/
// clang-format on
template <bool IsMasking_ = true, bool IsLocal_ = false>
struct GenericAttentionMask
{
    static constexpr bool IsMasking = IsMasking_; // false will disable masking
    static constexpr bool IsLocal   = IsLocal_;   // if true, upper/lower area could have mask,
                                                  // else only upper-right could have mask

    __host__ __device__ GenericAttentionMask() : y(0), x(0), y_total(0), x_total(0) {}

    __host__ __device__
    GenericAttentionMask(index_t y_, index_t x_, index_t y_total_, index_t x_total_)
        : y(y_), x(x_), y_total(y_total_), x_total(x_total_)
    {
    }
    template <typename MaskCoordinates>
    __host__ __device__ GenericAttentionMask(const MaskCoordinates& mask_coord)
        : y(mask_coord.At(Number<0>{})),
          x(mask_coord.At(Number<1>{})),
          y_total(mask_coord.At(Number<2>{})),
          x_total(mask_coord.At(Number<3>{}))
    {
    }

    template <index_t YTile, index_t XTile>
    __host__ __device__ constexpr auto
    GetTileRangeAlongX(index_t i_y, Number<YTile>, Number<XTile>) const
    {
        if constexpr(!IsMasking)
        {
            return ck::make_tuple(0, x_total);
        }
        else
        {
            // get the tile start/end range assum we loop over along X tile by tile
            index_t x_start = [&]() {
                if constexpr(IsLocal)
                {
                    index_t tmp = math::max(-y + i_y + 1, 0);
                    return (tmp / XTile) * XTile; // round to tile aligned
                }
                else
                {
                    return 0;
                }
            }();

            index_t x_end = [&]() {
                index_t tmp = math::min(i_y + YTile - 1 + x, x_total);
                return ((tmp + XTile - 1) / XTile) * XTile;
            }();

            // return index:[start, end), end-start=length
            return ck::make_tuple(x_start, x_end);
        }
    }

    __host__ __device__ constexpr auto IsOutOfBound(index_t i_y, index_t i_x) const
    {
        if constexpr(!IsMasking)
        {
            return false;
        }
        else
        {
            // no need to do min/max here, since i_x will never be < 0 or >= x_total
            index_t x_start = -y + i_y + 1;
            index_t x_end   = i_y + x;

            if constexpr(IsLocal)
            {
                return i_x < x_start || i_x >= x_end;
            }
            else
            {
                return i_x >= x_end;
            }
        }
    }

    // if current tile is at the edge, which means need per-pixel mask check.
    // otherwise no need to check per-pixel
    // Attention! assume the idex passed in this function is with in range of GetTileRangeAlongX()
    template <index_t YTile, index_t XTile>
    __host__ __device__ constexpr auto
    IsEdgeTile(index_t i_y, index_t i_x, Number<YTile>, Number<XTile>) const
    {
        if constexpr(IsLocal)
        {
            // check top-right corner > x or left-borrom corner < x
            bool top_right_edge   = (i_x + XTile) > (x + i_y);
            bool bottom_left_edge = (i_y + YTile) > (y + i_x);
            return top_right_edge || bottom_left_edge;
        }
        else
        {
            // only need to check top-right corner > x
            bool top_right_edge = (i_x + XTile) > (x + i_y);
            return top_right_edge;
        }
    }

    private:
    index_t y, x;
    index_t y_total, x_total;
};

} // namespace block
} // namespace tile_program

// TODO: prefer use this function in host code
// if left_size < 0 && right_size = 0, it is normal causal mask
// local is left_size >=0 or right_size >=0
__host__ constexpr auto
make_generic_attention_mask_coordinates_from_lr_window(index_t left_size,
                                                       index_t right_size,
                                                       index_t y_total,
                                                       index_t x_total,
                                                       bool is_top_left = true)
{
    index_t x = 0, y = 0;

    if(is_top_left)
    {
        if(left_size < 0)
            left_size = y_total - 1;
        if(right_size < 0)
            right_size = x_total - 1;

        x = 1 + right_size;
        y = left_size + 1;
    }
    else
    {
        if(left_size < 0)
            left_size = x_total - 1;
        if(right_size < 0)
            right_size = y_total - 1;

        x = x_total - y_total + 1 + right_size;
        y = y_total - x_total + 1 + left_size;
    }

    return ck::make_tuple(y, x, y_total, x_total);
}
} // namespace ck
