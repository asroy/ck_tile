// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

#include "ck/utility/remove_cvref.hpp"

namespace ck {

template <typename Pred, typename Arg, typename ArgReceiver>
__host__ auto select_arg(Pred pred,
                         Arg&& arg,
                         ArgReceiver&& arg_receiver,
                         std::optional<std::function<void()>> error_handler = std::nullopt)
    -> std::enable_if_t<std::is_invocable_r_v<bool, Pred> &&
                            std::is_invocable_v<remove_reference_t<ArgReceiver>, Arg&&>,
                        bool>
{
    if(pred())
    {
        arg_receiver(std::forward<Arg>(arg));
        return true;
    }

    if(error_handler.has_value())
    {
        (*error_handler)();
    }

    return false;
}

template <typename FirstPred,
          typename FirstArg,
          typename SecondPred,
          typename SecondArg,
          typename ArgReceiver>
__host__ auto select_arg(FirstPred first_pred,
                         FirstArg&& first_arg,
                         SecondPred&& second_pred,
                         SecondArg&& second_arg,
                         ArgReceiver&& arg_receiver,
                         std::optional<std::function<void()>> error_handler = std::nullopt)
    -> std::enable_if_t<std::is_invocable_r_v<bool, FirstPred> &&
                            std::is_invocable_v<remove_reference_t<ArgReceiver>, FirstArg&&>,
                        bool>
{
    if(first_pred())
    {
        arg_receiver(std::forward<FirstArg>(first_arg));
        return true;
    }

    return select_arg(std::forward<SecondPred>(second_pred),
                      std::forward<SecondArg>(second_arg),
                      arg_receiver,
                      std::move(error_handler));
}

template <typename FirstPred,
          typename FirstArg,
          typename SecondPred,
          typename SecondArg,
          typename ThirdPred,
          typename ThirdArg,
          typename ArgReceiver>
__host__ auto select_arg(FirstPred first_pred,
                         FirstArg&& first_arg,
                         SecondPred&& second_pred,
                         SecondArg&& second_arg,
                         ThirdPred&& third_pred,
                         ThirdArg&& third_arg,
                         ArgReceiver&& arg_receiver,
                         std::optional<std::function<void()>> error_handler = std::nullopt)
    -> std::enable_if_t<std::is_invocable_r_v<bool, FirstPred> &&
                            std::is_invocable_v<remove_reference_t<ArgReceiver>, FirstArg&&>,
                        bool>
{
    if(first_pred())
    {
        arg_receiver(std::forward<FirstArg>(first_arg));
        return true;
    }

    return select_arg(std::forward<SecondPred>(second_pred),
                      std::forward<SecondArg>(second_arg),
                      std::forward<ThirdPred>(third_pred),
                      std::forward<ThirdArg>(third_arg),
                      arg_receiver,
                      std::move(error_handler));
}

template <typename FirstPred,
          typename FirstArg,
          typename SecondPred,
          typename SecondArg,
          typename ThirdPred,
          typename ThirdArg,
          typename FourthPred,
          typename FourthArg,
          typename ArgReceiver>
__host__ auto select_arg(FirstPred first_pred,
                         FirstArg&& first_arg,
                         SecondPred&& second_pred,
                         SecondArg&& second_arg,
                         ThirdPred&& third_pred,
                         ThirdArg&& third_arg,
                         FourthPred&& fourth_pred,
                         FourthArg&& fourth_arg,
                         ArgReceiver&& arg_receiver,
                         std::optional<std::function<void()>> error_handler = std::nullopt)
    -> std::enable_if_t<std::is_invocable_r_v<bool, FirstPred> &&
                            std::is_invocable_v<remove_reference_t<ArgReceiver>, FirstArg&&>,
                        bool>
{
    if(first_pred())
    {
        arg_receiver(std::forward<FirstArg>(first_arg));
        return true;
    }

    return select_arg(std::forward<SecondPred>(second_pred),
                      std::forward<SecondArg>(second_arg),
                      std::forward<ThirdPred>(third_pred),
                      std::forward<ThirdArg>(third_arg),
                      std::forward<FourthPred>(fourth_pred),
                      std::forward<FourthArg>(fourth_arg),
                      arg_receiver,
                      std::move(error_handler));
}

} // namespace ck
