# k40_expertd: routed experts on a second GPU

`k40_expertd` holds some of DeepSeek-V4.1's routed experts on a separate CUDA GPU (written for a Tesla K40, compute
capability 3.5) and computes them for `strata-dsv41`'s CPU lane. Strata's GPU path is unchanged: the CPU lane sends the
picks the daemon holds, computes its own experts meanwhile, and adds the daemon's sums into the answer it already
returns. If the daemon is down or late, the lane computes those experts itself.

## Build and run

The daemon needs CUDA 11.x (the last toolkit with Kepler) and runs against NVIDIA's 470 driver. `build.sh` compiles it
in a Docker image with CUDA 11.8 and gcc 11 (native sm_35 code, the CUDA runtime linked statically):

```
src/dsv41/expertd/build.sh            # -> build-expertd/k40_expertd
build-expertd/k40_expertd --model <the pack's GGUF directory> [--port 30060]
DS4_EXPERTD=127.0.0.1:30060 strata launch ...
```

## Protocol (expertd_proto.hpp)

One TCP connection, little-endian, `TCP_NODELAY`.

| From | Message |
|---|---|
| client | `Hello` (magic, version, `dim`, `ff`, the SwiGLU clamp) |
| daemon | `HelloReply` (its free device memory) |
| client | `u32 n`, then `n` x `Id` (layer, expert): candidates, hottest first |
| daemon | `u32 m`, then `m` x `Id`: the ones it now holds (layers whose gate, up and down are IQ1_S, while memory lasts) |
| client | per CPU-lane request with held experts: `ComputeHdr` (seq, layer, `T`, pairs), `T` x `dim` int8 activations, `T` x `dim / 32` F32 scales (the rounding of `q8_quant`), then the pairs (expert, token, route weight) |
| daemon | `ReplyHdr` (seq, token mask), then `dim` F32 for every token in the mask: the sum of weight x down over its pairs |

The daemon computes gate and up with `iq::mv`, the clamped SwiGLU, rounds `h` to int8 a 32 (as the device path does),
then down and the weighted sum, all on one stream.
