// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>

#include "ck/utility/functional.hpp"
#include "ck/utility/remove_cvref.hpp"

namespace ck {
namespace detail {
// helper of runtime version of std::get() for non-const std::tuple<>
template <typename... Ts, typename Function, std::size_t Index>
void runtime_get_impl(std::size_t target_index,
                      std::tuple<Ts...>& tuple,
                      Function&& function,
                      std::integral_constant<std::size_t, Index>)
{
    static_assert(Index < sizeof...(Ts));

    if(target_index == Index)
    {
        std::invoke(function, std::get<Index>(tuple));
    }
    else
    {
        if constexpr(Index + 1 < sizeof...(Ts))
        {
            runtime_get_impl(
                target_index, tuple, function, std::integral_constant<std::size_t, Index + 1>{});
        }
    }
}

// helper of runtime version of std::get() for const std::tuple<>
template <typename... Ts, typename Function, std::size_t Index>
void runtime_get_impl(std::size_t target_index,
                      const std::tuple<Ts...>& tuple,
                      Function&& function,
                      std::integral_constant<std::size_t, Index>)
{
    static_assert(Index < sizeof...(Ts));

    if(target_index == Index)
    {
        std::invoke(function, std::get<Index>(tuple));
    }
    else
    {
        if constexpr(Index + 1 < sizeof...(Ts))
        {
            runtime_get_impl(
                target_index, tuple, function, std::integral_constant<std::size_t, Index + 1>{});
        }
    }
}
} // namespace detail

// runtime version of std::get() for non-const std::tuple<>
template <typename... Ts, typename Function>
void runtime_get(std::size_t index, std::tuple<Ts...>& tuple, Function&& function)
{
    static_assert(std::conjunction_v<std::is_invocable<Function, Ts>...>);

    assert(index < sizeof...(Ts));
    detail::runtime_get_impl(index, tuple, function, std::integral_constant<std::size_t, 0>{});
}

// runtime version of std::get() for const std::tuple<>
template <typename... Ts, typename Function>
void runtime_get(std::size_t index, const std::tuple<Ts...>& tuple, Function&& function)
{
    static_assert(std::conjunction_v<std::is_invocable<Function, Ts>...>);

    assert(index < sizeof...(Ts));
    detail::runtime_get_impl(index, tuple, function, std::integral_constant<std::size_t, 0>{});
}

namespace detail {
// helper of runtime_get() for nested std::tuple<>
template <typename Tuples, typename Function, std::size_t TupleIndex>
void runtime_get_impl(
    const std::array<std::size_t, std::tuple_size_v<remove_cvref_t<Tuples>>>& tuple_indices,
    Tuples&& tuples,
    Function&& function,
    std::integral_constant<std::size_t, TupleIndex>)
{
    if constexpr(TupleIndex == std::tuple_size_v<remove_cvref_t<Tuples>>)
    {
        std::invoke(function);
    }
    else
    {
        runtime_get(
            std::get<TupleIndex>(tuple_indices), std::get<TupleIndex>(tuples), [&](auto&& arg) {
                runtime_get_impl(tuple_indices,
                                 tuples,
                                 bind_front(std::ref(function), arg),
                                 std::integral_constant<std::size_t, TupleIndex + 1>{});
            });
    }
}
} // namespace detail

// runtime_get() for nested std::tuple<>
template <typename Tuples, typename Function>
void runtime_get(
    const std::array<std::size_t, std::tuple_size_v<remove_cvref_t<Tuples>>>& tuple_indices,
    Tuples&& tuples,
    Function&& function)
{
    detail::runtime_get_impl(
        tuple_indices, tuples, function, std::integral_constant<std::size_t, 0>{});
}

} // namespace ck
