// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <type_traits>

#include "ck/ck.hpp"
#include "ck/utility/array.hpp"
#include "ck/utility/functional.hpp"
#include "ck/utility/number.hpp"
#include "ck/utility/remove_cvref.hpp"
#include "ck/utility/tuple.hpp"

namespace ck {
namespace detail {
// helper of runtime version of At() for non-const Tuple<>
template <typename... Ts, typename Function, index_t Index>
__host__ __device__ constexpr void
runtime_get_impl(index_t target_index, Tuple<Ts...>& tuple, Function&& function, Number<Index>)
{
    static_assert(Index < sizeof...(Ts));

    if(target_index == Index)
    {
        function(tuple.template At<Index>());
    }
    else
    {
        if constexpr(Index + 1 < sizeof...(Ts))
        {
            runtime_get_impl(target_index, tuple, function, Number<Index + 1>{});
        }
    }
}

// helper of runtime version of At() for const Tuple<>
template <typename... Ts, typename Function, index_t Index>
__host__ __device__ constexpr void runtime_get_impl(index_t target_index,
                                                    const Tuple<Ts...>& tuple,
                                                    Function&& function,
                                                    Number<Index>)
{
    static_assert(Index < sizeof...(Ts));

    if(target_index == Index)
    {
        std::invoke(function, tuple.template At<Index>());
    }
    else
    {
        if constexpr(Index + 1 < sizeof...(Ts))
        {
            runtime_get_impl(target_index, tuple, function, Number<Index + 1>{});
        }
    }
}
} // namespace detail

// runtime version of At() for non-const Tuple<>
template <typename... Ts, typename Function>
__host__ __device__ constexpr void
runtime_get(index_t index, Tuple<Ts...>& tuple, Function&& function)
{
    static_assert(std::conjunction_v<std::is_invocable<Function, Ts>...>);

    assert(index < static_cast<index_t>(sizeof...(Ts)));
    detail::runtime_get_impl(index, tuple, function, Number<0>{});
}

// runtime version of At() for const Tuple<>
template <typename... Ts, typename Function>
__host__ __device__ constexpr void
runtime_get(index_t index, const Tuple<Ts...>& tuple, Function&& function)
{
    static_assert(std::conjunction_v<std::is_invocable<Function, Ts>...>);

    assert(index < static_cast<index_t>(sizeof...(Ts)));
    detail::runtime_get_impl(index, tuple, function, Number<0>{});
}

namespace detail {
// helper of runtime_get() for nested Tuple<>
template <typename Tuples, typename Function, index_t TupleIndex>
__host__ __device__ constexpr void
runtime_get_impl(const Array<index_t, remove_cvref_t<Tuples>::Size()>& indices,
                 Tuples&& tuples,
                 Function&& function,
                 Number<TupleIndex>)
{
    if constexpr(TupleIndex == remove_cvref_t<Tuples>::Size())
    {
        function();
    }
    else
    {
        runtime_get(
            indices.template At<TupleIndex>(), tuples.template At<TupleIndex>(), [&](auto&& arg) {
                runtime_get_impl(
                    indices, tuples, bind_front(std::ref(function), arg), Number<TupleIndex + 1>{});
            });
    }
}
} // namespace detail

// runtime_get() for nested Tuple<>
template <typename Tuples, typename Function>
__host__ __device__ constexpr void
runtime_get(const Array<index_t, remove_cvref_t<Tuples>::Size()>& indices,
            Tuples&& tuples,
            Function&& function)
{
    detail::runtime_get_impl(indices, tuples, function, Number<0>{});
}

} // namespace ck
