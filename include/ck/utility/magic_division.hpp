// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"
#include "integral_constant.hpp"
#include "number.hpp"
#include "type.hpp"
#include "tuple.hpp"

namespace ck {

// magic number division
// Caution:
//   1. For uint32_t as dividend: magic number division implementation being used would produce
//   correct result if the dividend is uint32_t and its value is within 31-bit value range.
//   2. For int32_t as dividendd: magic number division for int32_t dividened has not been
//   implemented, the int32_t dividend would be bit-wise interpreted as uint32_t and magic number
//   division implementation for uint32_t is then used. Therefore, dividend value need to be
//   non-negative.
// TODO:
//   1. Implement magic number divison for int32_t
//   2. Implement magic number divison for unit32_t with 32-bit value range
struct MagicDivision32BitRange
{
    // uint32_t
    __host__ __device__ static constexpr auto CalculateMagicNumbers(uint32_t divisor)
    {
        // WARNING: magic division is only valid for division inside this range.
        // assert(divisor >= 1 && divisor <= INT32_MAX)

        uint32_t shift_u32 = 0;

        while((1U << shift_u32) < divisor)
        {
            shift_u32++;
        };

        uint64_t tmp_u64        = ((1UL << shift_u32) - divisor) << 32;
        uint32_t multiplier_u32 = tmp_u64 / divisor + 1;

        return make_tuple(multiplier_u32, shift_u32);
    }

    // integral_constant<uint32_t, .>
    template <uint32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicNumbers(integral_constant<uint32_t, Divisor>)
    {
        constexpr auto tmp = CalculateMagicNumbers(uint32_t{Divisor});

        constexpr uint32_t multiplier = tmp[Number<0>{}];
        constexpr uint32_t shift      = tmp[Number<1>{}];

        return make_tuple(integral_constant<uint32_t, multiplier>{},
                          integral_constant<uint32_t, shift>{});
    }

    // integral_constant<int32_t, .>
    template <int32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicNumbers(integral_constant<int32_t, Divisor>)
    {
        return CalculateMagicNumbers(integral_constant<uint32_t, Divisor>{});
    }

    // magic division for uint32_t
    __device__ static constexpr uint32_t
    DoMagicDivision(uint32_t dividend, uint32_t multiplier, uint32_t shift)
    {
        uint32_t tmp = __umulhi(dividend, multiplier);
        return (tmp + dividend) >> shift;
    }

    __host__ static constexpr uint32_t
    DoMagicDivision(uint32_t dividend, uint32_t multiplier, uint32_t shift)
    {
        uint32_t tmp = (static_cast<uint64_t>(dividend) * multiplier) >> 32;
        return (tmp + dividend) >> shift;
    }

    // magic division for int32_t
    // HACK: use dividend_i32 as if it's uint32_t, dividend_i32 need to be
    // non-negative for result to be correct
    // TODO: figure out how to do magic number divison for int32_t as dividended
    __device__ static constexpr int32_t
    DoMagicDivision(int32_t dividend_i32, uint32_t multiplier, uint32_t shift)
    {
        uint32_t dividend_u32 = bit_cast<uint32_t>(dividend_i32);
        uint32_t tmp          = __umulhi(dividend_u32, multiplier);
        return (tmp + dividend_u32) >> shift;
    }

    __host__ static constexpr int32_t
    DoMagicDivision(int32_t dividend_i32, uint32_t multiplier, uint32_t shift)
    {
        uint32_t dividend_u32 = bit_cast<uint32_t>(dividend_i32);
        uint32_t tmp          = (static_cast<uint64_t>(dividend_u32) * multiplier) >> 32;
        return (tmp + dividend_u32) >> shift;
    }
};

// magic number division
// This version on works for divisor and dividended between [0, 1 << 16]
struct MagicDivision16BitRange
{
    // uint32_t
    __host__ __device__ static constexpr auto CalculateMagicNumbers(uint32_t divisor)
    {
        // WARNING: magic division is only valid for division inside this range.
        // assert(divisor >= 1 && divisor <= (1U << 16));

        uint32_t shift_u32 = 0;

        while((1U << shift_u32) < divisor)
        {
            shift_u32++;
        };

        uint32_t one            = 1;
        uint32_t multiplier_u32 = ((one << 16) * ((one << shift_u32) - divisor)) / divisor + 1;

        return make_tuple(multiplier_u32, shift_u32);
    }

    // integral_constant<uint32_t, .>
    template <uint32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicNumbers(integral_constant<uint32_t, Divisor>)
    {
        constexpr auto tmp = CalculateMagicNumbers(uint32_t{Divisor});

        constexpr uint32_t multiplier = tmp[Number<0>{}];
        constexpr uint32_t shift      = tmp[Number<1>{}];

        return make_tuple(integral_constant<uint32_t, multiplier>{},
                          integral_constant<uint32_t, shift>{});
    }

    // integral_constant<int32_t, .>
    template <int32_t Divisor>
    __host__ __device__ static constexpr auto
        CalculateMagicNumbers(integral_constant<int32_t, Divisor>)
    {
        return CalculateMagicNumbers(integral_constant<uint32_t, Divisor>{});
    }

    // magic division for uint32_t
    __device__ static constexpr uint32_t
    DoMagicDivision(uint32_t dividend, uint32_t multiplier, uint32_t shift)
    {
        uint32_t tmp = (dividend * multiplier) >> 16;
        return (tmp + dividend) >> shift;
    }

    __host__ static constexpr uint32_t
    DoMagicDivision(uint32_t dividend, uint32_t multiplier, uint32_t shift)
    {
        uint32_t tmp = (dividend * multiplier) >> 16;
        return (tmp + dividend) >> shift;
    }

    // magic division for int32_t
    // HACK: use dividend_i32 as if it's uint32_t, dividend_i32 need to be
    // non-negative for result to be correct
    // TODO: figure out how to do magic number divison for int32_t as dividended
    __device__ static constexpr int32_t
    DoMagicDivision(int32_t dividend_i32, uint32_t multiplier, uint32_t shift)
    {
        uint32_t dividend_u32 = bit_cast<uint32_t>(dividend_i32);
        uint32_t tmp          = (dividend_u32 * multiplier) >> 16;
        return (tmp + dividend_u32) >> shift;
    }

    __host__ static constexpr int32_t
    DoMagicDivision(int32_t dividend_i32, uint32_t multiplier, uint32_t shift)
    {
        uint32_t dividend_u32 = bit_cast<uint32_t>(dividend_i32);
        uint32_t tmp          = (dividend_u32 * multiplier) >> 16;
        return (tmp + dividend_u32) >> shift;
    }
};

using MagicDivision = MagicDivision32BitRange;

} // namespace ck
