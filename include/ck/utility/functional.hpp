// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <functional>

#include "ck/utility/integral_constant.hpp"
#include "ck/utility/type.hpp"

namespace ck {

// TODO: right? wrong?
struct forwarder
{
    template <typename T>
    __host__ __device__ constexpr T&& operator()(T&& x) const
    {
        return static_cast<T&&>(x);
    }
};

struct swallow
{
    template <typename... Ts>
    __host__ __device__ constexpr swallow(Ts&&...)
    {
    }
};

template <typename T>
struct logical_and
{
    constexpr bool operator()(const T& x, const T& y) const { return x && y; }
};

template <typename T>
struct logical_or
{
    constexpr bool operator()(const T& x, const T& y) const { return x || y; }
};

template <typename T>
struct logical_not
{
    constexpr bool operator()(const T& x) const { return !x; }
};

// Emulate if constexpr
template <bool>
struct static_if;

template <>
struct static_if<true>
{
    using Type = static_if<true>;

    template <typename F>
    __host__ __device__ constexpr auto operator()(F f) const
    {
        // This is a trick for compiler:
        //   Pass forwarder to lambda "f" as "auto" argument, and make sure "f" will
        //   use it,
        //   this will make "f" a generic lambda, so that "f" won't be compiled
        //   until being
        //   instantiated here
        f(forwarder{});
        return Type{};
    }

    template <typename F>
    __host__ __device__ static void Else(F)
    {
    }
};

template <>
struct static_if<false>
{
    using Type = static_if<false>;

    template <typename F>
    __host__ __device__ constexpr auto operator()(F) const
    {
        return Type{};
    }

    template <typename F>
    __host__ __device__ static void Else(F f)
    {
        // This is a trick for compiler:
        //   Pass forwarder to lambda "f" as "auto" argument, and make sure "f" will
        //   use it,
        //   this will make "f" a generic lambda, so that "f" won't be compiled
        //   until being
        //   instantiated here
        f(forwarder{});
    }
};

template <bool predicate, class X, class Y>
struct conditional;

template <class X, class Y>
struct conditional<true, X, Y>
{
    using type = X;
};

template <class X, class Y>
struct conditional<false, X, Y>
{
    using type = Y;
};

template <bool predicate, class X, class Y>
using conditional_t = typename conditional<predicate, X, Y>::type;

// z = predicate ? x : y
template <bool predicate, typename X, typename Y>
constexpr auto conditional_expr(X&& x, Y&& y)
{
    if constexpr(predicate)
    {
        return std::forward<X>(x);
    }
    else
    {
        return std::forward<Y>(y);
    }
}

struct identity
{
    template <typename T>
    __host__ __device__ constexpr T&& operator()(T&& arg) const noexcept
    {
        return std::forward<T>(arg);
    }
};

namespace detail {
template <typename Function, typename FirstArg>
struct front_binder
{
    static_assert(!std::is_reference_v<Function> && !std::is_reference_v<FirstArg>);

    template <typename... Args>
    __host__ __device__ constexpr decltype(auto) operator()(Args&&... args) const
    {
        /// TODO: use std::invoke() like function to support more callable types
        return function(first_arg, std::forward<Args>(args)...);
    }

    mutable Function function;
    FirstArg& first_arg;
};
} // namespace detail

// like std::bind_front(), but keep reference to the first argument
template <typename Function, typename FirstArg>
auto bind_front_ref(Function&& function, FirstArg& first_arg)
{
    return detail::front_binder<std::remove_reference_t<Function>,
                                std::remove_reference_t<FirstArg>>{std::forward<Function>(function),
                                                                   first_arg};
}

} // namespace ck
