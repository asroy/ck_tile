add_example_executable(example_fmha_fwd fmha_fwd.cpp)

# NOTE: this is dangerous since will change the whole kernel to flush denormals
#       WIP with compiler team for an exp2 intrinsic..., then remove this
if(NOT DEFINED FMHA_FWD_FAST_EXP2)
    set(FMHA_FWD_FAST_EXP2 true)
endif()

if(FMHA_FWD_FAST_EXP2)
set_source_files_properties(fmha_fwd.cpp PROPERTIES COMPILE_OPTIONS "-DCK_FMHA_FWD_FAST_EXP2=1;-fgpu-flush-denormals-to-zero")
else()
set_source_files_properties(fmha_fwd.cpp PROPERTIES COMPILE_OPTIONS "-DCK_FMHA_FWD_FAST_EXP2=0")
endif()
