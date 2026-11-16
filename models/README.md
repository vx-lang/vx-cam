# The segmentation model

`selfie_segmentation_landscape.f32` holds the weights of MediaPipe Selfie Segmentation, the
landscape model, by Tingbo Hou, Siargey Pisarchyk and Karthik Raveendran at Google (2021). Google
publishes it under the Apache License, Version 2.0, a copy of which is in [LICENSE](LICENSE).

- Model: <https://storage.googleapis.com/mediapipe-assets/selfie_segmentation_landscape.tflite>
  (250145 bytes, SHA-256
  `a77d03f4659b9f6b6c1f5106947bf40e99d7655094b6527f214ea7d451106edd`)
- Model card, which gives the license:
  <https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Selfie%20Segmentation.pdf>
- The MediaPipe page that lists both:
  <https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/models.md#selfie-segmentation>

The network takes a 144 x 256 RGB image with values in 0 to 1, and gives a 144 x 256 mask of
the person. vx-cam runs it in Vx (`vxcam_model` in `pipeline/vxcam.vx`).

## What changed

The .tflite file stores its weights as float16. `tools/convert_model.py` turns them into
float32 and writes them one layer after another, in the order of the network: each conv,
depthwise conv and transposed conv gives its kernel (in TFLite's layout) and then its bias.
That is 55 layers and 106401 floats. Nothing else changed.

    python3 tools/convert_model.py selfie_segmentation_landscape.tflite \
        models/selfie_segmentation_landscape.f32

`tools/check_model.py` compares vx-cam's mask with TFLite's on the frames that
`vx-cam --headless` writes out.

# The noise removal model

`rnnoise.f32` holds the weights of RNNoise v0.1.1 by Jean-Marc Valin (Mozilla and the Xiph.Org
Foundation, 2017), and its table of tanh. RNNoise is under a BSD license, a copy of which is in
[LICENSE-rnnoise](LICENSE-rnnoise). `pipeline/denoise.vx` ports its C code to Vx, under the same
license.

- Code and weights: <https://github.com/xiph/rnnoise>, the tag `v0.1.1` (commit 6cbfd53e).
  The weights are `src/rnn_data.c` (SHA-256
  `7b90ed18f065a34597d80b4c00e140a7c5eb6380ef8c9c6bf265384c944ae708`), and the table is
  `src/tansig_table.h`.
- The paper: Jean-Marc Valin, "A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band
  Speech Enhancement", MMSP 2018. <https://arxiv.org/abs/1709.08243>

The network takes 42 features of a frame of 10 ms of audio at 48 kHz. A dense layer and three
GRUs (24, 48 and 96 units) give a gain for each of 22 bands, and the probability of a voice.

## What changed

`rnn_data.c` stores the weights as int8 values, and the network scales each sum by 1/256.
`tools/convert_rnnoise.py` writes the same values as float32, one layer after another: for
each layer its input weights, then its recurrent weights (a GRU only), then its bias, in the
layout of `rnn_data.c`. After them come the 201 values of the table of tanh. That is 87704
floats. Nothing else changed.

    python3 tools/convert_rnnoise.py rnnoise/src models/rnnoise.f32

`make check-denoise` compares the Vx port with the C code on the same audio
(`tools/check_denoise.py`).
