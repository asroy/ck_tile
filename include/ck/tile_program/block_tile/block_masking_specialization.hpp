// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tile_program {
namespace block {

struct MaskDisabledPredicate
{
    __host__ __device__ constexpr bool operator()(index_t /*m*/, index_t /*n*/) const
    {
        return false;
    };

    __host__ __device__ constexpr bool
        IsTileSkippable(index_t /*m*/, index_t /*n*/, index_t /*m_tile*/, index_t /*n_tile*/) const
    {
        return false;
    }
};

struct MaskUpperTriangleFromTopLeftPredicate
{
    __host__ __device__ constexpr bool operator()(index_t m, index_t n) const { return n > m; }

    __host__ __device__ constexpr bool
    IsTileSkippable(index_t m, index_t n, index_t m_tile, index_t /*n_tile*/) const
    {
        return operator()(m + m_tile - 1, n);
    }
};

// eg: m = 3, n = 5 => offset = 2
//    so matrix(n > m + offset) = 0
//      1  2  3  4  5
//    1 *  *  *  0  0
//    2 *  *  *  *  0
//    3 *  *  *  *  *
struct MaskUpperTriangleFromBottomRightPredicate
{
    __host__ __device__ void SetDiagonalOffset(const index_t diagonal_offset)
    {
        diagonal_offset_ = diagonal_offset;
    }
    __host__ __device__ constexpr bool operator()(index_t m, index_t n) const
    {
        return n > (m - diagonal_offset_);
    }

    __host__ __device__ constexpr bool IsTileSkippable(index_t m_tile_orig,
                                                       index_t n_tile_orig,
                                                       index_t m_tile_size,
                                                       index_t /*n_tile_size*/) const
    {
        return operator()(m_tile_orig + m_tile_size - 1, n_tile_orig);
    }

    private:
    index_t diagonal_offset_;
};

// to track the points which need to be set to -inf on C0
// Note: no need to reset M padding value, because they will not be stored out.
template <typename MaskOutPredicate_>
struct C0MatrixMask_impl
{
    using MaskOutPredicate = MaskOutPredicate_;

    __host__ __device__ C0MatrixMask_impl(index_t MRaw, index_t NRaw)
        : NRaw_(NRaw), predicate_(MaskOutPredicate{})
    {
        if constexpr(std::is_same_v<MaskOutPredicate, MaskUpperTriangleFromBottomRightPredicate>)
        {
            predicate_.SetDiagonalOffset(MRaw - NRaw);
        }
    }

    __host__ __device__ constexpr bool IsNOutOfBound(/*index_t m, */ index_t n) const
    {
        return n >= NRaw_;
    }

    __host__ __device__ constexpr bool IsMaskedElement(index_t m, index_t n) const
    {
        return predicate_(m, n) || IsNOutOfBound(n);
    }

    __host__ __device__ constexpr bool
    IsTileSkippable(index_t m, index_t n, index_t m_tile, index_t n_tile) const
    {
        return predicate_.IsTileSkippable(m, n, m_tile, n_tile);
    }

    private:
    // index_t MRaw_;
    index_t NRaw_;
    MaskOutPredicate predicate_;
};

// clang-format off
/*
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
template <bool IsDisableMask_ = false, bool IsLocal_ = false>
struct GenericAttentionMask
{
    static constexpr bool IsDisableMask = IsDisableMask_;
    static constexpr bool HasMask       = !IsDisableMask;
    static constexpr bool IsLocal       = IsLocal_;

    __host__ __device__ GenericAttentionMask() : x(0), y(0), x_total(0), y_total(0) {}

    __host__ __device__
    GenericAttentionMask(index_t x_, index_t y_, index_t x_total_, index_t y_total_)
        : x(x_), y(y_), x_total(x_total_), y_total(y_total_)
    {
    }
    template <typename MaskCoordinates>
    __host__ __device__ GenericAttentionMask(const MaskCoordinates& mask_coord)
        : x(mask_coord.At(Number<0>{})),
          y(mask_coord.At(Number<1>{})),
          x_total(mask_coord.At(Number<2>{})),
          y_total(mask_coord.At(Number<3>{}))
    {
    }

    template <index_t XTile, index_t YTile>
    __host__ __device__ constexpr auto
    GetTileRangeAlongX(index_t /*i_x*/, index_t i_y, Number<XTile>, number<YTile>) const
    {
        if constexpr(IsDisableMask)
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
                    return x - 1;
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

    __host__ __device__ constexpr auto IsMasking(index_t i_x, index_t i_y) const
    {
        if constexpr(IsDisableMask)
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

    private:
    index_t x, y;
    index_t x_total, y_total;
};

} // namespace block
} // namespace tile_program

// TODO: prefer use this function in host code
// if left_size < 0 && right_size = 0, it is normal causal mask
// local is left_size >=0 or right_size >=0
__host__ constexpr auto
make_generic_attention_mask_coordinates_from_lr_window(index_t left_size,
                                                       index_t right_size,
                                                       index_t x_total,
                                                       index_t y_total,
                                                       bool is_top_left = true)
{
    index_t x, y;

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

    return ck::make_tuple(x, y, x_total, y_total);
}
} // namespace ck
