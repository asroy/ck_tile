#!/bin/sh
# TODO: run this script from CK root
BUILD=build
EXE=$BUILD/bin/example_fmha_fwd
VALID=0

for prec in "fp16" "bf16" ; do
for mode in 0 1 ; do  # batch/group
for perm in 0 1 ; do  # bshd/bhsd

$EXE -prec=$prec -mode=$mode -b=32 -h=16 -d=128 -s=512   -iperm=$perm -operm=$perm -v=$VALID ; sleep 3
$EXE -prec=$prec -mode=$mode -b=16 -h=16 -d=128 -s=1024  -iperm=$perm -operm=$perm -v=$VALID ; sleep 3
$EXE -prec=$prec -mode=$mode -b=8  -h=16 -d=128 -s=2048  -iperm=$perm -operm=$perm -v=$VALID ; sleep 3
$EXE -prec=$prec -mode=$mode -b=4  -h=16 -d=128 -s=4096  -iperm=$perm -operm=$perm -v=$VALID ; sleep 3
$EXE -prec=$prec -mode=$mode -b=2  -h=16 -d=128 -s=8192  -iperm=$perm -operm=$perm -v=$VALID ; sleep 3
$EXE -prec=$prec -mode=$mode -b=1  -h=16 -d=128 -s=16384 -iperm=$perm -operm=$perm -v=$VALID ; sleep 3

$EXE -prec=$prec -mode=$mode -b=32 -h=32 -d=64 -s=512   -iperm=$perm -operm=$perm -v=$VALID ; sleep 3
$EXE -prec=$prec -mode=$mode -b=16 -h=32 -d=64 -s=1024  -iperm=$perm -operm=$perm -v=$VALID ; sleep 3
$EXE -prec=$prec -mode=$mode -b=8  -h=32 -d=64 -s=2048  -iperm=$perm -operm=$perm -v=$VALID ; sleep 3
$EXE -prec=$prec -mode=$mode -b=4  -h=32 -d=64 -s=4096  -iperm=$perm -operm=$perm -v=$VALID ; sleep 3
$EXE -prec=$prec -mode=$mode -b=2  -h=32 -d=64 -s=8192  -iperm=$perm -operm=$perm -v=$VALID ; sleep 3
$EXE -prec=$prec -mode=$mode -b=1  -h=32 -d=64 -s=16384 -iperm=$perm -operm=$perm -v=$VALID ; sleep 3

done
done
done
