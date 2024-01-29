# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import argparse
import itertools
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import copy

DTYPE_MAP = {
    "fp16": "ck::half_t",
    "bf16": "ck::bhalf_t",
    "fp8" : "ck::f8_t"
}

MASK_MAP = {
    "no" : "FmhaMasks::NoMask",
    "causal" : "FmhaMasks::CausalMask",
    "generic" : "FmhaMasks::GenericMask"
}

MODE_MAP = {
    "batch" : "false",
    "group" : "true"
}

LAYOUT_MAP = {
    "row" : "true",
    "col" : "false"
}

PIPELINE_MAP = {
    "qr" : "ck::tile_program::block::BlockFmhaPipelineQRKSVS",
    "qr_fp8" : "ck::tile_program::block::BlockFmhaPipelineQRKSVSFp8",
    "qr_async" : "ck::tile_program::block::BlockFmhaPipelineQRKSVSAsync",
}

BOOL_MAP = {
    "t" : "true",
    "f" : "false"
}

MASKS = ["no", "causal", "generic"]
DIRECTIONS = ["fwd"]
GEN_DIR = ""    # in Cmake, have to generate files in same folder

FMHA_FWD_KERNEL_HEADER = """// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.\n
// auto generated by generate.py
#include "fmha_fwd.hpp"
"""

FMHA_FWD_KERNEL_BODY="""
using fmha_dtype_{F_idx} = {F_dtype};

using fmha_block_tile_{F_idx} = ck::Sequence<{F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0blen}>;
using fmha_block_warps_{F_idx} = ck::Sequence<{F_rm}, {F_rn}, {F_rk}>;
using fmha_warp_tile_{F_idx} = ck::Sequence<{F_wm}, {F_wn}, {F_wk}>;

using fmha_shape_{F_idx} = ck::tile_program::TileFmhaShape<fmha_block_tile_{F_idx},
                                      fmha_block_warps_{F_idx},
                                      fmha_warp_tile_{F_idx},
                                      fmha_block_warps_{F_idx},
                                      fmha_warp_tile_{F_idx},
                                      {F_vlayout}>;

using fmha_trait_{F_idx} = ck::tile_program::TileFmhaTraits<{F_spad},
                                                    {F_skpad},
                                                    {F_dpad},
                                                    {F_dvpad},
                                                    {F_bias},
                                                    {F_lse},
                                                    {F_occupancy}>;
using fmha_mask_{F_idx} = {F_mask};

using fmha_pipeline_problem_{F_idx} = ck::tile_program::block::BlockFmhaPipelineProblem<
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::QDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::KDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::VDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::SaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::SMPLComputeDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::BiasDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::LSEDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::PDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::OaccDataType,
    typename FmhaFwdTypeConfig<fmha_dtype_{F_idx}>::ODataType,
    fmha_shape_{F_idx},
    {F_mode},
    fmha_mask_{F_idx},
    fmha_trait_{F_idx}>;

using fmha_pipeline_{F_idx} = {F_pipeline}<
    fmha_pipeline_problem_{F_idx}>;

using fmha_epilogue_{F_idx} =
    FmhaFwdEpilogue<FmhaFwdEpilogueProblem<typename FmhaFwdTypeConfig<{F_dtype}>::OaccDataType,
                                           typename FmhaFwdTypeConfig<{F_dtype}>::ODataType>>;

using fmha_kernel_{F_idx} = 
    FmhaFwdKernel<FmhaFwdTilePartitioner<fmha_shape_{F_idx}>,
                  fmha_pipeline_{F_idx},
                  fmha_epilogue_{F_idx}>;

using trait_{F_idx} = fmha_fwd_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_vlayout}, fmha_mask_{F_idx}, {F_bias}, {F_lse}>;

template<>
float fmha_fwd_<trait_{F_idx}>(const StreamConfig& s, fmha_fwd_args a)
{{
    using k_ = fmha_kernel_{F_idx};
    auto [kargs, grids] = fmha_fwd_create_kargs_and_grids<k_>(a);
    constexpr dim3 blocks             = k_::BlockSize();
    constexpr ck::index_t kBlockPerCu = k_::kBlockPerCu;
    return launch_kernel<blocks.x, kBlockPerCu>(s, k_{{}}, grids, blocks, 0, kargs);
}}
"""

FMHA_FWD_API_FILENAME="fmha_fwd_api.cpp"
FMHA_FWD_API="""
float fmha_fwd(fmha_fwd_traits t, fmha_fwd_args a, const StreamConfig& s){{
    float r = -1;
{F_dispatch}
    return r;
}}
"""

FMHA_FWD_API_PER_DTYPE="""    {F_if}(t.data_type.compare(\"{F_dtype}\") == 0){{
        switch (t.hdim){{
{F_hdim_case}
            default:
            break;
        }}
    }}
"""
FMHA_FWD_API_PER_HDIM_CASE="""            case {F_hdim}: {{
{F_inner_dispatch}
            }}
            break;
"""
MASK_CHECK_MAP = {
    "no" : "t.mask_type == mask_enum::no_mask",
    "causal" : "t.mask_type == mask_enum::causal_top_left || t.mask_type == mask_enum::causal_bottom_right",
    "generic" : "t.mask_type == mask_enum::window_generic",
}

FMHA_FWD_API_INNER_DISPATCH="""                {F_if}((t.is_group_mode == {F_mode}) && (t.is_v_rowmajor == {F_vlayout}) && ({F_mask_check}) && (t.has_bias == {F_bias}) && (t.has_lse == {F_lse})) {{
                    using trait_ = fmha_fwd_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_vlayout}, {F_mask}, {F_bias}, {F_lse}>;
                    return fmha_fwd_<trait_>(s, a);
                }}
"""

@dataclass
class FmhaFwdApiTrait:
    # sync with fmha_fwd_traits<>, to generate fallback calls
    hdim      : str
    dtype     : str  # data type
    mode      : str  # value from MODE_MAP
    vlayout   : str
    mask      : str
    bias      : str  # true/false
    lse       : str  #

    @property
    def name(self) -> str:
        return f'{self.hdim}-{self.dtype}-{self.mode}-{self.vlayout}-{self.mask}-{self.bias}-{self.lse}'

class FmhaFwdApiPool:
    def __init__(self):
        self.pool = dict()

    def register_traits(self, trait : FmhaFwdApiTrait) -> None:
        # TODO: do we need to check duplication?
        if trait.dtype not in self.pool.keys():
            self.pool[trait.dtype] = dict()
        if trait.hdim not in self.pool[trait.dtype].keys():
            self.pool[trait.dtype][trait.hdim] = list()

        self.pool[trait.dtype][trait.hdim].append(copy.copy(trait))

    @property
    def api(self) -> str:
        per_dtypes=str()
        for i, dtype in enumerate(self.pool.keys()):
            per_hdim_case=str()
            for hdim in self.pool[dtype].keys():
                traits=self.pool[dtype][hdim]
                inners=str()
                for j, trait in enumerate(traits):
                    if0 = 'if' if j == 0 else 'else if'
                    inners = inners + FMHA_FWD_API_INNER_DISPATCH.format(F_if=if0, F_mode=MODE_MAP[trait.mode], F_vlayout=LAYOUT_MAP[trait.vlayout], F_mask=MASK_MAP[trait.mask],
                                   F_mask_check=MASK_CHECK_MAP[trait.mask], F_bias=BOOL_MAP[trait.bias], F_lse=BOOL_MAP[trait.lse], F_hdim=hdim, F_dtype=DTYPE_MAP[dtype])
            
                per_hdim_case = per_hdim_case + FMHA_FWD_API_PER_HDIM_CASE.format(F_hdim=hdim, F_inner_dispatch=inners)
            if1 = 'if' if i == 0 else 'else if'
            per_dtypes = per_dtypes + FMHA_FWD_API_PER_DTYPE.format(F_if=if1, F_dtype=dtype, F_hdim_case=per_hdim_case)

        return FMHA_FWD_KERNEL_HEADER + FMHA_FWD_API.format(F_dispatch = per_dtypes)

@dataclass
class FmhaFwdTileSize:
    F_bm0       : int  # tile size along q seqlen (block size)
    F_bn0       : int  # tile size along qk seqlen
    F_bk0       : int  # tile size along qk gemm unroll
    F_bn1       : int  # tile size along v head_dim
    F_bk1       : int  # tile size along kv gemm unroll
    F_bk0blen   : int  # total length of K0, used for pipeline that need load Q at once (or repeately load Q as a whole tile)
    F_rm        : int  # number of warps along q seqlen (block warps)
    F_rn        : int  # number of warps along k seqlen(not used)
    F_rk        : int  # number of warps along gemm-k(not used)
    F_wm        : int  # warp size along m (warp size)
    F_wn        : int  # warp size along n
    F_wk        : int  # warp size along k
    F_occupancy : int  # occupancy
    @property
    def name(self) -> str:
        return f"b{self.F_bm0}x{self.F_bn0}x{self.F_bk0}x{self.F_bn0}x{self.F_bk1}x{self.F_bk0blen}" +\
        f"_r{self.F_rm}x{self.F_rn}x{self.F_rk}_w{self.F_wm}x{self.F_wn}x{self.F_wk}_o{self.F_occupancy}"

@dataclass
class FmhaFwdKernel:
    direction   : str
    F_idx       : int  # this is not a tunable, but a counter to differentiate symbol    
    F_hdim      : int  # hdim
    F_dtype     : str  # data type
    F_tile      : FmhaFwdTileSize
    F_vlayout   : str  # row/col
    F_spad      : str  # true/false
    F_skpad     : str  #
    F_dpad      : str  #
    F_dvpad     : str  #
    F_bias      : str  # true/false
    F_lse       : str  #
    F_mask      : str  # value from MASK_MAP
    F_mode      : str  # value from MODE_MAP
    F_pipeline  : str  # value from PIPIELINE_MAP

    @property
    def template(self) -> str:
        return FMHA_FWD_KERNEL_HEADER + \
            FMHA_FWD_KERNEL_BODY.format(
                F_idx       = self.F_idx,
                F_hdim      = self.F_hdim,
                F_dtype     = DTYPE_MAP[self.F_dtype],
                F_bm0       = self.F_tile.F_bm0,
                F_bn0       = self.F_tile.F_bn0,
                F_bk0       = self.F_tile.F_bk0,
                F_bn1       = self.F_tile.F_bn1,
                F_bk1       = self.F_tile.F_bk1,
                F_bk0blen   = self.F_tile.F_bk0blen,
                F_rm        = self.F_tile.F_rm,
                F_rn        = self.F_tile.F_rn,
                F_rk        = self.F_tile.F_rk,
                F_wm        = self.F_tile.F_wm,
                F_wn        = self.F_tile.F_wn,
                F_wk        = self.F_tile.F_wk,
                F_vlayout   = LAYOUT_MAP[self.F_vlayout],
                F_spad      = BOOL_MAP[self.F_spad],
                F_skpad     = BOOL_MAP[self.F_skpad],
                F_dpad      = BOOL_MAP[self.F_dpad],
                F_dvpad     = BOOL_MAP[self.F_dvpad],
                F_bias      = BOOL_MAP[self.F_bias],
                F_lse       = BOOL_MAP[self.F_lse],
                F_occupancy = self.F_tile.F_occupancy ,
                F_mask      = MASK_MAP[self.F_mask],
                F_mode      = MODE_MAP[self.F_mode],
                F_pipeline  = PIPELINE_MAP[self.F_pipeline])

    @property
    def name(self) -> str:
        # TODO: we don't encode idx here
        return f"fmha_{self.direction}_d{self.F_hdim}_{self.F_dtype}_{self.F_mode}_" + self.F_tile.name + f"_v{self.F_vlayout[0]}" +\
            f"_p{BOOL_MAP[self.F_spad][0]}{BOOL_MAP[self.F_skpad][0]}{BOOL_MAP[self.F_dpad][0]}{BOOL_MAP[self.F_dvpad][0]}" +\
            f"_{BOOL_MAP[self.F_bias][0]}_m{self.F_mask[0]}_l{BOOL_MAP[self.F_lse][0]}_{self.F_pipeline}"

    @property
    def filename(self) -> str:
        return self.name + ".cpp"

    def api_trait(self) -> FmhaFwdApiTrait:
        return FmhaFwdApiTrait(hdim=str(self.F_hdim),
                dtype=self.F_dtype,
                mode=self.F_mode,
                vlayout=self.F_vlayout,
                mask=self.F_mask,
                bias=self.F_bias,
                lse=self.F_lse)

# TODO: design a more practical way to do it
# this is current supported tile size.
def get_fmha_fwd_tile_dict_from_dtype(direction : str, dtype : str) -> Optional[dict]:
    if direction == 'fwd':
        if dtype == 'fp16' or dtype == 'bf16':
            return {
                '32'  : FmhaFwdTileSize(128, 64, 16, 32, 32, 32,     2, 1, 1, 32, 32, 16, 2),
                '64'  : FmhaFwdTileSize(128, 64, 32, 64, 32, 64,     4, 1, 1, 32, 32, 16, 3),
                '128' : FmhaFwdTileSize(128, 128, 32, 128, 32, 128,  4, 1, 1, 32, 32, 16, 2),
                '256' : FmhaFwdTileSize(128, 128, 32, 256, 32, 256,  4, 1, 1, 32, 32, 16, 1),
            }
        elif dtype == 'fp8' or dtype == 'bf8':
            return {
                '128' : FmhaFwdTileSize(128, 128, 32, 128, 32, 128,  4, 1, 1, 32, 32, 32, 2)
            }
        else:
            return None
    else:
        return None

def get_blobs() -> Tuple[FmhaFwdApiPool, List[FmhaFwdKernel]]:
    # TODO: we don't support tuning yet, so pick up one value for vlayout/pipeline/pad
    #       support this in future
    def get_vlayout(dtype, hdim):
        if dtype in ['fp16', 'bf16']:
            return 'row'
        elif dtype in ['fp8', 'bf8']:
            return 'col'
        else:
            assert Fasle
    def get_pipeline(dtype, hdim):
        if dtype in ['fp16', 'bf16']:
            if hdim == 256:
                return 'qr'
            else:
                return 'qr'
        elif dtype in ['fp8', 'bf8']:
            return 'qr_fp8'
        else:
            assert Fasle
    def get_pad(dtype, hdim):
        return 'f'

    gen = list()
    api_pool = FmhaFwdApiPool()

    for direction, dtype in itertools.product(DIRECTIONS, DTYPE_MAP.keys()):
        d = get_fmha_fwd_tile_dict_from_dtype(direction, dtype)
        if d == None:
            continue
        for hdim_str, mode, mask, bias, lse in itertools.product(d.keys(), MODE_MAP.keys(), MASK_MAP.keys(), ["t", "f"], ["t", "f"]):
            tile = d[hdim_str]
            hdim = int(hdim_str)
            if dtype in ['fp8', 'bf8'] and lse == "t":
                continue
            k = FmhaFwdKernel(direction=direction, F_idx=0, F_hdim=hdim, F_dtype=dtype, F_tile=tile, F_vlayout=get_vlayout(dtype, hdim),
                                F_spad='t', F_skpad='t', F_dpad='f',
                                F_dvpad='f', F_bias=bias, F_lse=lse, F_mask=mask, F_mode=mode,
                                F_pipeline=get_pipeline(dtype, hdim))
            api_pool.register_traits(k.api_trait())
            gen.append(k)

    return (api_pool, gen)

def write_single_kernel(kernel: FmhaFwdKernel, autogen_dir: Path) -> None:
    (autogen_dir / kernel.filename).write_text(kernel.template)

def write_api(api_pool : FmhaFwdApiPool, autogen_dir: Path) -> None:
    (autogen_dir / FMHA_FWD_API_FILENAME).write_text(api_pool.api)

def write_blobs(output_dir: Optional[str]) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir) / GEN_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    api_pool, kernels = get_blobs()
    for kernel in kernels:
        write_single_kernel(kernel, output_dir)
    write_api(api_pool, output_dir)

# list all the files that will be generated
def list_blobs(output_file: Optional[str]) -> None:
    assert output_file is not None
    file_path = Path(output_file)
    with file_path.open('a') as f:
        _, kernels = get_blobs()
        for kernel in kernels:
            f.write(str(file_path.parent / GEN_DIR / kernel.filename) + "\n")
        f.write(str(file_path.parent / GEN_DIR / FMHA_FWD_API_FILENAME) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen api for CK fmha kernel",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="write all the blobs into a directory"
    )
    parser.add_argument(
        "-l",
        "--list_blobs",
        required=False,
        help="list all the kernels to a file"
    )
    args = parser.parse_args()
    if args.list_blobs is not None:
        list_blobs(args.list_blobs)
    else:
        write_blobs(args.output_dir)
