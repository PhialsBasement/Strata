#!/usr/bin/env bash
# Build k40_expertd with CUDA 11.8 in Docker (image cuda118-k40:
# ubuntu 22.04, gcc 11, cuda-nvcc-11-8, cuda-cudart-dev-11-8).
set -euo pipefail
src=$(cd "$(dirname "$0")/../../.." && pwd)
out=${1:-$src/build-expertd}
mkdir -p "$out"
docker run --rm -u "$(id -u):$(id -g)" -v "$src:/src:ro" -v "$out:/out" cuda118-k40 bash -c '
set -e
N=/usr/local/cuda-11.8/bin/nvcc
F="-std=c++17 -O3 -arch=sm_35 -Wno-deprecated-gpu-targets -Xptxas -dlcm=ca"
I="-I /src/src/dsv41 -I /src/include -I /src/third_party/ggml"
$N $F $I -c /src/src/dsv41/ds4_iq.cu -o /out/ds4_iq.o
$N $F $I -c /src/src/dsv41/ds4_tq.cu -o /out/ds4_tq.o
$N $F $I -c /src/src/dsv41/expertd/k40_expertd.cu -o /out/k40_expertd.o
$N -arch=sm_35 -Wno-deprecated-gpu-targets -cudart static /out/k40_expertd.o /out/ds4_iq.o /out/ds4_tq.o -o /out/k40_expertd
'
echo "built $out/k40_expertd"
