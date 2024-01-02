add_example_executable(example_fmha_fwd fmha_fwd.cpp)

file(GLOB FMHA_FWD_KERNEL_IMPL_FILES ${CMAKE_CURRENT_LIST_DIR}/invoke_fmha_kernel_*.cpp)
target_sources(example_fmha_fwd PRIVATE
  mask.cpp
  utils.cpp
  ${FMHA_FWD_KERNEL_IMPL_FILES}
)

# NOTE: this is dangerous since will change the whole kernel to flush denormals
#       WIP with compiler team for an exp2 intrinsic..., then remove this
if(NOT DEFINED FMHA_FWD_FAST_EXP2)
    set(FMHA_FWD_FAST_EXP2 true)
endif()

if(FMHA_FWD_FAST_EXP2)
set_source_files_properties(fmha_fwd.cpp ${FMHA_FWD_KERNEL_IMPL_FILES} PROPERTIES COMPILE_OPTIONS "-DCK_FMHA_FWD_FAST_EXP2=1;-fgpu-flush-denormals-to-zero")
else()
set_source_files_properties(fmha_fwd.cpp ${FMHA_FWD_KERNEL_IMPL_FILES} PROPERTIES COMPILE_OPTIONS "-DCK_FMHA_FWD_FAST_EXP2=0")
endif()
