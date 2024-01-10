// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include <array>
#include <cstring>
#include <numeric>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>

#include "ck/ck.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/tensor/tensor_view.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_description/cluster_descriptor.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/utility/common_header.hpp"

#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/fill.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"

#include "common/arg_parser.hpp"
#include "fmha_fwd.hpp"
#include "mask.hpp"
#include "reference/reference_batched_elementwise.hpp"
#include "reference/reference_batched_gemm.hpp"
#include "reference/reference_batched_masking.hpp"
#include "reference/reference_batched_softmax.hpp"
#include "utils.hpp"

auto create_args(int argc, char* argv[])
{
    ArgParser arg_parser;
    arg_parser.insert("v", "1", "weather do cpu validation or not")
        .insert("mode", "0", "kernel mode. 0:batch, 1:group")
        .insert("b", "2", "batch size")
        .insert("h", "8", "num of head, for q")
        .insert("h_k",
                "0",
                "num of head, for k/v, 0 means equal to h\n"
                "if not equal to h, then this is GQA/MQA case")
        .insert("s", "3328", "seqlen_q")
        .insert("s_k", "0", "seqlen_k, 0 means equal to s")
        .insert("d", "128", "head dim for q, k")
        .insert("d_v", "0", "head dim for v, 0 means equal to d")
        .insert("scale", "0", "scale factor. 0 means equal to 1/sqrt(seqlen)")
        .insert("iperm",
                "1",
                "permute input\n"
                "if true, will be b*h*s*d, else b*s*h*d")
        .insert("operm", "1", "permute output")
        .insert("bias", "0", "add bias or not")
        .insert("lse", "1", "0 not store lse, 1 store lse")
        .insert("prec", "fp16", "data type. fp16 or bf16")
        .insert("mask",
                "0",
                "0: no mask, 1: top-left, 2:bottom-right\n"
                "'t:l,r', top-left local-attn with left right size\n"
                "'b:l,r', bottom-r local-attn with left right size\n"
                "'g:y,x', generic attention mask coordinate with y/x size\n")
        .insert("init", "1", "init method. 0:random int, 1:random float, 2:trig float");

    bool result = arg_parser.parse(argc, argv);
    return std::make_tuple(result, arg_parser);
}

template <ck::index_t HDim_, typename DataType_>
struct fmha_fwd_kernel_invoker
{
    static constexpr ck::index_t HDim = HDim_;
    using DataType                    = DataType_;
    // these args are used to select kernel.
    // args that may passed as karg shoule use operator()
    mode_enum mode;
    bool use_bias;
    bool store_lse;
    mask_info mask;

    fmha_fwd_kernel_invoker(mode_enum mode_, bool use_bias_, bool store_lse_, mask_info mask_)
        : mode(mode_), use_bias(use_bias_), store_lse(store_lse_), mask(mask_)
    {
    }

    template <typename... Args>
    float operator()(const StreamConfig& stream, Args&&... args)
    {
        float ave_time;
        BOOL_SWITCH_3(
            mode == mode_enum::group, kIsGroupMode, use_bias, kHasBias, store_lse, kStoreLSE, [&] {
                if(mask.type == mask_enum::no_mask)
                {
                    using FmhaMask = FmhaMasks::NoMask;
                    using Kernel   = FmhaFwdKernelSelector<HDim,
                                                         DataType,
                                                         kIsGroupMode,
                                                         FmhaMask,
                                                         kHasBias,
                                                         kStoreLSE>;

                    auto [kargs, grids] =
                        fmha_fwd_create_kargs_and_grids<Kernel>(std::forward<Args>(args)...);
                    ave_time = fmha_fwd_run<Kernel>(stream, kargs, grids);
                }
                else
                {
                    BOOL_SWITCH(mask.type == mask_enum::window_generic, kIsLocal, [&]() {
                        using FmhaMask =
                            ck::tile_program::block::GenericAttentionMask<true, kIsLocal>;
                        using Kernel = FmhaFwdKernelSelector<HDim,
                                                             DataType,
                                                             kIsGroupMode,
                                                             FmhaMask,
                                                             kHasBias,
                                                             kStoreLSE>;

                        auto [kargs, grids] =
                            fmha_fwd_create_kargs_and_grids<Kernel>(std::forward<Args>(args)...);
                        ave_time = fmha_fwd_run<Kernel>(stream, kargs, grids);
                    });
                }
            });
        return ave_time;
    }
};

// different threshold for different dtype
template <typename DataType>
auto get_elimit(int /*init_method*/)
{
    double rtol = 1e-3;
    double atol = 1e-3;
    return ck::make_tuple(rtol, atol);
}

template <>
auto get_elimit<ck::bhalf_t>(int init_method)
{
    if(init_method == 0)
    {
        double rtol = 1e-2;
        double atol = 1e-2;
        return ck::make_tuple(rtol, atol);
    }
    else
    {
        double rtol = 3e-3;
        double atol = 3e-3;
        return ck::make_tuple(rtol, atol);
    }
}

template <typename DataType>
bool run(const ArgParser& arg_parser)
{
    int do_validation   = arg_parser.get_int("v");
    auto mode           = static_cast<mode_enum>(arg_parser.get_uint32("mode"));
    ck::index_t batch   = arg_parser.get_int("b");
    ck::index_t nhead   = arg_parser.get_int("h");
    ck::index_t nhead_k = arg_parser.get_int("h_k");
    if(nhead_k == 0)
        nhead_k = nhead;

    if(nhead % nhead_k != 0)
    {
        std::cerr << "nhead:" << nhead << " must be multiple of nhead_k:" << nhead_k << std::endl;
        return false;
    }

    ck::index_t seqlen_q = arg_parser.get_int("s");
    ck::index_t seqlen_k = arg_parser.get_int("s_k");
    if(seqlen_k == 0)
        seqlen_k = seqlen_q;
    ck::index_t hdim_q = arg_parser.get_int("d");
    ck::index_t hdim_v = arg_parser.get_int("d_v");
    if(hdim_v == 0)
        hdim_v = hdim_q;

    int i_perm = arg_parser.get_int("iperm"); // if true, will be batch * nhead * seqlen * hdim
    int o_perm = arg_parser.get_int("operm"); // if false, will be batch * seqlen * nhead * hdim

    float scale = arg_parser.get_float("scale");
    if(scale == .0f)
        scale = 1.0 / ck::math::sqrt(static_cast<float>(hdim_q)); // TODO: q ? v ?

    bool use_bias = arg_parser.get_uint32("bias");

    bool store_lse = arg_parser.get_uint32("lse");

    mask_info mask = decode_mask_info(arg_parser.get_str("mask"), seqlen_q, seqlen_k);

    int init_method = arg_parser.get_int("init");

    int stream_warmup = env_get_int("CK_WARMUP", 5);
    int stream_repeat = env_get_int("CK_REPEAT", 20);

    StreamConfig stream_config{nullptr, true, 0, stream_warmup, stream_repeat};

    const auto [seqlens_q, seqstart_q_host] = generate_seqlens_seqstarts_q(mode, batch, seqlen_q);
    const std::vector<int32_t> seqstart_k_host =
        generate_seqstarts_k(mode, batch, seqlen_k, seqlens_q, seqlen_q);

    using TypeConfig = FmhaFwdTypeConfig<DataType>;

    using QDataType           = typename TypeConfig::QDataType;
    using KDataType           = typename TypeConfig::KDataType;
    using VDataType           = typename TypeConfig::VDataType;
    using BiasDataType        = typename TypeConfig::BiasDataType;
    using LSEDataType         = typename TypeConfig::LSEDataType;
    using SaccDataType        = typename TypeConfig::SaccDataType;
    using SMPLComputeDataType = typename TypeConfig::SMPLComputeDataType;
    using PDataType           = typename TypeConfig::PDataType;
    using OaccDataType        = typename TypeConfig::OaccDataType;
    using ODataType           = typename TypeConfig::ODataType;

    // accumulation numbers for performance evaluation
    std::size_t flop = 0, num_byte = 0;
    auto max_seqlen_q =
        std::numeric_limits<int32_t>::min(); // we will use max seqlen to decide grid size
    {
        for(ck::index_t wb = 0; wb < batch; ++wb)
        {
            const int32_t real_seqlen_q = seqstart_q_host[wb + 1] - seqstart_q_host[wb];
            const int32_t real_seqlen_k = seqstart_k_host[wb + 1] - seqstart_k_host[wb];

            if(max_seqlen_q < real_seqlen_q)
            {
                max_seqlen_q = real_seqlen_q;
            }

            using namespace ck::literals;

            flop += nhead * (2_uz * real_seqlen_q * real_seqlen_k * hdim_q +
                             2_uz * real_seqlen_q * hdim_v * real_seqlen_k);

            num_byte += nhead * (sizeof(QDataType) * real_seqlen_q * hdim_q +
                                 sizeof(KDataType) * real_seqlen_k * hdim_q +
                                 sizeof(VDataType) * hdim_v * real_seqlen_k +
                                 sizeof(ODataType) * real_seqlen_q * hdim_v);
        }
    }

    auto get_lengths = [&](int permute,
                           ck::index_t b /*batch*/,
                           ck::index_t h /*nhead*/,
                           ck::index_t s /*seqlen*/,
                           ck::index_t d /*hdim*/) {
        if(permute)
            return std::array<ck::index_t, 4>{b, h, s, d};
        else
            return std::array<ck::index_t, 4>{b, s, h, d};
    };

    constexpr bool is_v_rowmajor = ck::is_same_v<VLayout, ck::tensor_layout::gemm::RowMajor>;

    // host memory for storing all the tensor elements
    const ck::index_t shape_batch = (mode == mode_enum::batch ? batch : 1);
    const ck::index_t shape_seqlen_q =
        (mode == mode_enum::batch ? seqlen_q : seqstart_q_host.back());
    const ck::index_t shape_seqlen_k =
        (mode == mode_enum::batch ? seqlen_k : seqstart_k_host.back());

    Tensor<QDataType> q_host(get_lengths(i_perm, shape_batch, nhead, shape_seqlen_q, hdim_q));
    Tensor<KDataType> k_host(get_lengths(i_perm, shape_batch, nhead_k, shape_seqlen_k, hdim_q));
    Tensor<VDataType> v_host(
        is_v_rowmajor ? get_lengths(i_perm, shape_batch, nhead_k, shape_seqlen_k, hdim_v)
                      : get_lengths(i_perm, shape_batch, nhead_k, hdim_v, shape_seqlen_k));
    // use bias shape = [1, 1, shape_seqlen_q, shape_seqlen_k]. if use_bias=false, the bias_host
    // will not be used for verification at all (but will be copied to device anyway).
    Tensor<KDataType> bias_host(
        use_bias ? get_lengths(i_perm, 1, 1, shape_seqlen_q, shape_seqlen_k)
                 : std::array<ck::index_t, 4>{1, 1, 1, 1} /* dummy shape for simplifying code */);
    // self define lse data layout as [shape_batch, nhead, shape_seqlen_q, 1]
    Tensor<LSEDataType> lse_host(
        store_lse ? std::array<ck::index_t, 4>{shape_batch, nhead, shape_seqlen_q, 1}
                  : std::array<ck::index_t, 4>{1, 1, 1, 1} /* dummy shape for simplifying code */);

    Tensor<ODataType> o_host(get_lengths(o_perm, shape_batch, nhead, shape_seqlen_q, hdim_v));

    if(init_method == 0)
    {
        ck::utils::FillUniformDistributionIntegerValue<QDataType>{-2.f, 2.f}(q_host);
        ck::utils::FillUniformDistributionIntegerValue<KDataType>{-2.f, 2.f}(k_host);
        ck::utils::FillUniformDistributionIntegerValue<VDataType>{-2.f, 2.f}(v_host);
        ck::utils::FillUniformDistributionIntegerValue<BiasDataType>{-2.f, 2.f}(bias_host);
    }
    else if(init_method == 1)
    {
        ck::utils::FillUniformDistribution<QDataType>{0.f, 1.f}(q_host);
        ck::utils::FillUniformDistribution<KDataType>{0.f, 1.f}(k_host);
        ck::utils::FillUniformDistribution<VDataType>{-.5f, .5f}(v_host);
        ck::utils::FillUniformDistribution<BiasDataType>{0.f, 1.f}(bias_host);
    }
    else if(init_method == 2)
    {
        ck::utils::FillTrigValue<QDataType>{}(q_host);
        ck::utils::FillTrigValue<KDataType>{}(k_host);
        ck::utils::FillTrigValue<VDataType>{}(v_host);
        ck::utils::FillTrigValue<BiasDataType>{}(bias_host);
    }

    DeviceMem q_buf(q_host.GetElementSpaceSizeInBytes());
    DeviceMem k_buf(k_host.GetElementSpaceSizeInBytes());
    DeviceMem v_buf(v_host.GetElementSpaceSizeInBytes());
    DeviceMem bias_buf(bias_host.GetElementSpaceSizeInBytes());
    DeviceMem lse_buf(lse_host.GetElementSpaceSizeInBytes());
    DeviceMem o_buf(o_host.GetElementSpaceSizeInBytes());
    DeviceMem seqstart_q(seqstart_q_host.size() * sizeof(int32_t));
    DeviceMem seqstart_k(seqstart_k_host.size() * sizeof(int32_t));

    q_buf.ToDevice(q_host.data());
    k_buf.ToDevice(k_host.data());
    v_buf.ToDevice(v_host.data());
    bias_buf.ToDevice(bias_host.data());
    seqstart_q.ToDevice(seqstart_q_host.data());
    seqstart_k.ToDevice(seqstart_k_host.data());

    // clang-format off
    auto layout_str = [&](int permute){
        if (permute) return std::string("bhsd");
        else return std::string("bshd");
    };
    auto io_layout = [&](int iperm_, int operm_) {
        if (iperm_ == operm_) return layout_str(iperm_);
        else return layout_str(iperm_) + std::string("-") + layout_str(operm_);
    };
    // clang-format on
    const std::string prec = arg_parser.get_str("prec");

    std::cout << "[" << prec << "|" << mode << "|" << io_layout(i_perm, o_perm) << "] b:" << batch
              << ", h:" << nhead << "/" << nhead_k << ", s:" << seqlen_q << "/" << seqlen_k
              << ", d:" << hdim_q << "/" << hdim_v << ", scale:" << scale << ", bias:" << use_bias
              << ", lse:" << store_lse << ", mask:" << mask
              << ", v:" << std::string(VLayout::name)[0] << std::flush;

#define INVOKE_FMHA_KERNEL(hdim_)                                              \
    fmha_fwd_kernel_invoker<hdim_, DataType>{mode, use_bias, store_lse, mask}( \
        stream_config,                                                         \
        q_buf.GetDeviceBuffer(),                                               \
        k_buf.GetDeviceBuffer(),                                               \
        v_buf.GetDeviceBuffer(),                                               \
        bias_buf.GetDeviceBuffer(),                                            \
        lse_buf.GetDeviceBuffer(),                                             \
        o_buf.GetDeviceBuffer(),                                               \
        seqstart_q.GetDeviceBuffer(),                                          \
        seqstart_k.GetDeviceBuffer(),                                          \
        nullptr,                                                               \
        batch,                                                                 \
        nhead,                                                                 \
        nhead_k,                                                               \
        shape_seqlen_q,                                                        \
        shape_seqlen_k,                                                        \
        hdim_q,                                                                \
        hdim_v,                                                                \
        max_seqlen_q,                                                          \
        scale,                                                                 \
        i_perm,                                                                \
        o_perm,                                                                \
        mask.y,                                                                \
        mask.x)

    float ave_time = 0;
    if(hdim_q == hdim_v && hdim_q == 64)
    {
        ave_time = INVOKE_FMHA_KERNEL(64);
    }
    else if(hdim_q == hdim_v && hdim_q == 128)
    {
        ave_time = INVOKE_FMHA_KERNEL(128);
    }
    else
    {
        std::cerr << "not support hdim, will not run" << std::endl;
        return false;
    }

    float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

    float gb_per_sec = num_byte / 1.E6 / ave_time;

    std::cout << std::fixed << ", " << std::setprecision(3) << ave_time << " ms, "
              << std::setprecision(2) << tflops << " TFlops, " << std::setprecision(2) << gb_per_sec
              << " GB/s" << std::flush;

    if(!do_validation)
    {
        std::cout << std::endl;
        return true;
    }

    o_buf.FromDevice(o_host.data());
    lse_buf.FromDevice(lse_host.data());

    bool pass = true;

    for(ck::index_t wb = 0; wb < batch; ++wb)
    {
        const ck::index_t real_seqlen_q = seqstart_q_host[wb + 1] - seqstart_q_host[wb];
        const ck::index_t real_seqlen_k = seqstart_k_host[wb + 1] - seqstart_k_host[wb];

        // adjust matrix index according to the mode
        const ck::index_t b            = (mode == mode_enum::batch ? wb : 0);
        const ck::index_t query_offset = (mode == mode_enum::batch ? 0 : seqstart_q_host[wb]);
        const ck::index_t key_offset   = (mode == mode_enum::batch ? 0 : seqstart_k_host[wb]);

        const auto v_host_ref_lengths = std::array<ck::index_t, 3>{nhead, hdim_v, real_seqlen_k};
        const auto v_host_ref_strides =
            is_v_rowmajor ? std::array<ck::index_t, 3>{hdim_v * real_seqlen_k, 1, hdim_v}
                          : std::array<ck::index_t, 3>{hdim_v * real_seqlen_k, real_seqlen_k, 1};

        Tensor<QDataType> q_host_ref({nhead, real_seqlen_q, hdim_q});
        Tensor<KDataType> k_host_ref({nhead, real_seqlen_k, hdim_q});
        Tensor<VDataType> v_host_ref(v_host_ref_lengths, v_host_ref_strides);
        Tensor<ODataType> o_host_ref({nhead, real_seqlen_q, hdim_v});

        Tensor<SMPLComputeDataType> s_host_ref({nhead, real_seqlen_q, real_seqlen_k});
        Tensor<PDataType> p_host_ref({nhead, real_seqlen_q, real_seqlen_k});
        Tensor<SMPLComputeDataType> lse_host_ref({nhead, real_seqlen_q, 1});

        ck::index_t nr = nhead / nhead_k;

        // clang-format off
        // permute
        if(i_perm) q_host_ref.ForEach([&](auto& self, auto i) { self(i) = q_host(b, i[0], i[1] + query_offset, i[2]); });
        else       q_host_ref.ForEach([&](auto& self, auto i) { self(i) = q_host(b, i[1] + query_offset, i[0], i[2]); });

        if(i_perm) k_host_ref.ForEach([&](auto& self, auto i) { self(i) = k_host(b, i[0] / nr, i[1] + key_offset, i[2]); });
        else       k_host_ref.ForEach([&](auto& self, auto i) { self(i) = k_host(b, i[1] + key_offset, i[0] / nr, i[2]); });

        if constexpr (is_v_rowmajor) {
            //                                                             v_host_ref: [nhead, hdim, seq], v_host: [b, h_k, s, d]
            if(i_perm) v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(b, i[0] / nr, i[2] + key_offset, i[1]); });
            //                                                             v_host_ref: [nhead, hdim, seq], v_host: [b, s, h_k, d]
            else       v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(b, i[2] + key_offset, i[0] / nr, i[1]); });
        }
        else {
            if(i_perm) v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(b, i[0] / nr, i[1], i[2] + key_offset); });
            else       v_host_ref.ForEach([&](auto& self, auto i) { self(i) = v_host(b, i[1], i[0] / nr, i[2] + key_offset); });
        }

        // reference
        reference_batched_gemm<QDataType, KDataType, SaccDataType, SMPLComputeDataType>(
            q_host_ref, k_host_ref, s_host_ref,
            ck::identity{}, ck::identity{},
            [&](SaccDataType x) { return scale * x; });

        if(use_bias)
        {
            Tensor<BiasDataType> bias_host_ref({1, real_seqlen_q, real_seqlen_k});
            if(i_perm)
                bias_host_ref.ForEach([&](auto& self, auto i) { self(i) = bias_host(0, 0, i[1] + query_offset, i[2] + key_offset); });
            else
                bias_host_ref.ForEach([&](auto& self, auto i) { self(i) = bias_host(0, i[1] + query_offset, 0, i[2] + key_offset); });

            // broadcast from [1, real_seqlen_q, real_seqlen_k] to [nhead, real_seqlen_q, real_seqlen_k]
            reference_batched_elementwise<SMPLComputeDataType, BiasDataType, SMPLComputeDataType, SMPLComputeDataType>(
                s_host_ref, bias_host_ref, s_host_ref);
        }

        if(mask.type == mask_enum::no_mask) {
            reference_batched_masking<SaccDataType>(s_host_ref, FmhaMasks::NoMask{real_seqlen_q, real_seqlen_k});
        } else if(mask.type == mask_enum::window_generic) {
            reference_batched_masking<SaccDataType>(s_host_ref,
                FmhaMasks::GenericMask{mask.y, mask.x, real_seqlen_q, real_seqlen_k});
        } else {
            reference_batched_masking<SaccDataType>(s_host_ref,
                FmhaMasks::CausalMask{mask.y, mask.x, real_seqlen_q, real_seqlen_k});
        }
        if(store_lse){
            reference_batched_softmax<SMPLComputeDataType, SMPLComputeDataType, PDataType>(s_host_ref, p_host_ref, &lse_host_ref);
        }
        else{
            reference_batched_softmax<SMPLComputeDataType, SMPLComputeDataType, PDataType>(s_host_ref, p_host_ref);
        }
        
        reference_batched_gemm<PDataType, VDataType, OaccDataType, ODataType>(p_host_ref, v_host_ref, o_host_ref);

        Tensor<ODataType> o_host_result({nhead, real_seqlen_q, hdim_v});
        // permute
        if(o_perm) o_host_result.ForEach([&](auto& self, auto idx) { self(idx) = o_host(b, idx[0], idx[1] + query_offset, idx[2]); });
        else       o_host_result.ForEach([&](auto& self, auto idx) { self(idx) = o_host(b, idx[1] + query_offset, idx[0], idx[2]); });
        // clang-format on

        auto [rtol, atol] = get_elimit<DataType>(init_method);
        bool cur_pass     = ck::utils::check_err(
            o_host_result, o_host_ref, std::string("O Error: Incorrect results!"), rtol, atol);
        pass &= cur_pass;
        if(!cur_pass)
        {
            std::cerr << "O mismatch found at batch: " << wb << std::endl
                      << "\tseqlen_q: " << real_seqlen_q << std::endl
                      << "\tseqlen_k: " << real_seqlen_k << std::endl
                      << "\tseqstart_q: " << seqstart_q_host << std::endl
                      << "\tseqstart_k: " << seqstart_k_host << std::endl;

            break;
        }

        if(store_lse)
        {
            Tensor<SMPLComputeDataType> lse_host_result({nhead, real_seqlen_q, 1});
            lse_host_result.ForEach([&](auto& self, auto idx) {
                self(idx) = lse_host(b, idx[0], idx[1] + query_offset, idx[2]);
            });

            bool lse_pass = ck::utils::check_err(
                lse_host_result, lse_host_ref, "LSE Error: Incorrect results!", rtol, atol);

            pass &= lse_pass;
            if(!cur_pass)
            {
                std::cerr << "LSE mismatch found at batch: " << wb << std::endl
                          << "\tseqlen_q: " << real_seqlen_q << std::endl
                          << "\tseqlen_k: " << real_seqlen_k << std::endl
                          << "\tseqstart_q: " << seqstart_q_host << std::endl
                          << "\tseqstart_k: " << seqstart_k_host << std::endl;

                break;
            }
        }
    }

    std::cout << "check valid:" << (pass ? "y" : "n") << std::flush << std::endl;

    return pass;
}

int main(int argc, char* argv[])
{
    auto [result, arg_parser] = create_args(argc, argv);
    if(!result)
        return -1;

    const std::string data_type = arg_parser.get_str("prec");
    if(data_type == "fp16")
    {
        return run<ck::half_t>(arg_parser) ? 0 : -2;
    }
    else if(data_type == "bf16")
    {
        return run<ck::bhalf_t>(arg_parser) ? 0 : -2;
    }

    return -3;
}
