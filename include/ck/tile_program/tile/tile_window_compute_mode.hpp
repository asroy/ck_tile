// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/number.hpp"

namespace ck {
namespace tile_program {

struct TileWindowComputeMode final
{
    using Normal                   = Number<0>;
    using PreComputeCoordsForLoad  = Number<1>;
    using PreComputeCoordsForStore = Number<2>;
};

} // namespace tile_program
} // namespace ck
