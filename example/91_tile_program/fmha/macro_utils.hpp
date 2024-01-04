// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#define PP_UNWRAP(...) PP_UNWRAP_IMPL __VA_ARGS__
#define PP_UNWRAP_IMPL(...) __VA_ARGS__
