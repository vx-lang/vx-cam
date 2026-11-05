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
