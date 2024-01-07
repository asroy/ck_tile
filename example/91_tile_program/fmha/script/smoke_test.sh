#!/bin/sh
# TODO: run this script from CK root
BUILD=build
EXE=$BUILD/bin/example_fmha_fwd

for mode in 0 1 ; do  # batch/group
for prec in "fp16" "bf16" ; do
for perm in 0 1 ; do  # bshd/bhsd
for hdim in 128 64 ; do
for bias in 0 1 ; do  # no-bias/with-bias

$EXE -prec=$prec -mode=$mode -b=1 -h=1        -d=$hdim -s=1024          -bias=$bias -iperm=$perm -operm=$perm                -v=1
$EXE -prec=$prec -mode=$mode -b=1 -h=4 -h_k=2 -d=$hdim -s=256           -bias=$bias -iperm=$perm -operm=$perm                -v=1
$EXE -prec=$prec -mode=$mode -b=2 -h=2 -h_k=1 -d=$hdim -s=512  -s_k=256 -bias=$bias -iperm=$perm -operm=$perm                -v=1
$EXE -prec=$prec -mode=$mode -b=1 -h=2        -d=$hdim -s=256  -s_k=512 -bias=$bias -iperm=$perm -operm=$perm                -v=1
$EXE -prec=$prec -mode=$mode -b=1 -h=1        -d=$hdim -s=1024 -s_k=256 -bias=$bias -iperm=$perm -operm=$perm -mask=1        -v=1
$EXE -prec=$prec -mode=$mode -b=1 -h=1        -d=$hdim -s=1024 -s_k=256 -bias=$bias -iperm=$perm -operm=$perm -mask=2        -v=1
$EXE -prec=$prec -mode=$mode -b=1 -h=1        -d=$hdim -s=256  -s_k=512 -bias=$bias -iperm=$perm -operm=$perm -mask=g:128,32 -v=1


done
done
done
done
done
