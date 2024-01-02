// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#if 1 // DataType = fp16

using QDataType           = ck::half_t;
using KDataType           = ck::half_t;
using VDataType           = ck::half_t;
using BiasDataType        = ck::half_t;
using SaccDataType        = float;      // data type for first gemm accumulation
using SMPLComputeDataType = float;      // data type for reduction, softmax
using PDataType           = ck::half_t; // data type for A matrix of second gemm
using OaccDataType        = float;      // data type for second gemm accumulation
using ODataType           = ck::half_t;

#else // DataType = bf16

using QDataType           = ck::bhalf_t;
using KDataType           = ck::bhalf_t;
using VDataType           = ck::bhalf_t;
using BiasDataType        = ck::bhalf_t;
using SaccDataType        = float;       // data type for first gemm accumulation
using SMPLComputeDataType = float;       // data type for reduction, softmax
using PDataType           = ck::bhalf_t; // data type for A matrix of second gemm
using OaccDataType        = float;       // data type for second gemm accumulation
using ODataType           = ck::bhalf_t;

#endif
