// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <cstdint>
#include <optional>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>

#include "ck/utility/span.hpp"

enum class Mode : unsigned
{
    Batch = 0,
    Group
};

std::ostream& operator<<(std::ostream& stream, Mode mode);

std::vector<int32_t> to_seqstarts(ck::span<const int32_t> seqlens);

std::vector<int32_t> generate_seqlens_q(Mode mode,
                                        unsigned count,
                                        int32_t seqlens_q_sum,
                                        std::optional<unsigned> seed = std::nullopt);

std::tuple<std::vector<int32_t>, std::vector<int32_t>> generate_seqlens_seqstarts_q(
    Mode mode, unsigned count, int32_t seqlens_q_sum, std::optional<unsigned> seed = std::nullopt);

std::vector<int32_t> generate_seqlens_k(Mode mode,
                                        unsigned count,
                                        int32_t seqlens_k_sum,
                                        ck::span<const int32_t> seqlens_q,
                                        int32_t seqlens_q_sum,
                                        std::optional<unsigned> seed = std::nullopt);

std::vector<int32_t> generate_seqstarts_k(Mode mode,
                                          unsigned count,
                                          int32_t seqlens_k_sum,
                                          ck::span<const int32_t> seqlens_q,
                                          int32_t seqlens_q_sum,
                                          std::optional<unsigned> seed = std::nullopt);

int env_get_int(const char* var_name, int default_int);
