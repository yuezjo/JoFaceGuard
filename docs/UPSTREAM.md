# Upstream source, model provenance, and licenses

Verified against primary repository files on October 5, 2026.

## App fork

JoFaceGuard is a derivative fork of [KINN-CH/FaceUnlock](https://github.com/KINN-CH/FaceUnlock), based on commit `6547a3a3e14e2499e7ac3d99093eb0f5022c653f`.

The original app is MIT licensed, copyright © 2026 Cheolho Kim. [The upstream license](https://github.com/KINN-CH/FaceUnlock/blob/6547a3a3e14e2499e7ac3d99093eb0f5022c653f/LICENSE) permits using, modifying, forking, and redistributing its code, including commercially, provided the copyright and permission notice remain in copies or substantial portions. The original `LICENSE` is preserved in this fork. This is an independent derivative and does not imply endorsement by its author.

Useful upstream implementation retained or adapted includes AVFoundation camera handling, Vision facial landmarks, five-point similarity alignment, pixel orientation checks, and vector math. JoFaceGuard replaces the unlock workflow with a conservative lock-only classifier and continuous confirmation state machine. Password storage/injection, FileVault-related guidance, blink-based unlock, and automatic update logic are outside this app's scope.

## Why this source

| Project | Verified license and implementation | Selection |
| --- | --- | --- |
| [KINN-CH/FaceUnlock](https://github.com/KINN-CH/FaceUnlock) | MIT; native Swift, AVFoundation, Vision, Core ML; command-line build; alignment/orientation self-tests; model weights kept separate. | Chosen because the small, separable pipeline is suited to a lock-only derivative. |
| [HasBrain/FaceUnlock](https://github.com/HasBrain/FaceUnlock) | MIT app; Swift pipeline and multiple enrollment poses. README separately identifies its InsightFace model as non-commercial research only. | More password/unlock machinery to remove; same restricted default model. |
| [jonnyoo/glance](https://github.com/jonnyoo/glance) | MIT app; native Swift and fuller recognition UI; ArcFace model and a Vision feature-print fallback. | Larger product surface and model licensing must be treated separately. Generic feature-print fallback is unsuitable for deciding that someone is a stranger. |
| [dweep-desai/FaceGate-Mac](https://github.com/dweep-desai/FaceGate-Mac) | MIT app; app-locker rather than whole-screen guard; bundled InsightFace-derived MobileFaceNet. Its development fallback samples image pixels to make pseudo-embeddings. | Application-locking architecture and pseudo-embedding fallback are inappropriate for this guard. |

The source candidates are young projects; selecting one is not a claim of independently measured recognition reliability or an audit of their full security.

## Why SFace replaces the upstream ArcFace weights

[InsightFace's official license policy](https://github.com/deepinsight/insightface#license) separates MIT code from the downloaded training data and model weights, which are limited to **non-commercial research purposes**. A free app, local download, or personal daily use does not by itself establish compliance with that restriction. JoFaceGuard therefore does **not** ship, download, or use InsightFace weights.

[AdaFace's repository](https://github.com/mk-minchul/AdaFace) has an MIT license, but the author's [current response to the pretrained-weights commercial-use question](https://github.com/mk-minchul/AdaFace/issues/174#issuecomment-4721165960) directs users to MSU Technologies. It was not selected as a clearer alternative.

Instead this build uses [OpenCV Zoo's SFace](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface), an actual face-recognition model. Its [directory README](https://github.com/opencv/opencv_zoo/blob/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface/README.md) explicitly licenses **all files in that directory** under Apache 2.0; this includes the ONNX weights. A copy of that license is preserved as `Resources/SFACE_LICENSE.txt`.

Attribution: SFace contributed by **Yaoyao Zhong**; ONNX conversion by **Chengrui Wang**; OpenCV Zoo distribution. The associated Zoo wrapper identifies copyright © 2021 Shenzhen Institute of Artificial Intelligence and Robotics for Society. Model conversion for this app changes the container and computational representation from ONNX to Core ML, while retaining the trained weights. Conversion is an explicit modification, described here and in the model metadata.

Apache 2.0 permits use, modification, and redistribution subject to its terms. Redistributors must include its license and relevant attribution, identify modifications, and preserve any applicable notices. No endorsement or trademark license is implied. This app ships both its MIT notice and the SFace Apache license in its resources.

## Pinned model

- OpenCV Zoo revision: `47534e27c9851bb1128ccc0102f1145e27f23f98`
- File: `face_recognition_sface_2021dec.onnx`
- Source: [official pinned ONNX download](https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface/face_recognition_sface_2021dec.onnx)
- SHA-256: `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`
- Conversion: `tools/fetch_sface.py`, task-local build dependencies in `requirements-model.txt`.
- Runtime: `Resources/Models/SFace.mlpackage`, created locally and ignored by Git.
- Input: **raw RGB** float32 tensor, NCHW `1 × 3 × 112 × 112`, values `0…255`. Normalization is already inside the ONNX/Core ML graph. The old ArcFace `(pixel − 127.5) / 127.5` preprocessing must not be applied a second time.
- Output: **128 dimensions**, normalized to unit length by the app before cosine comparison.
- Alignment: the same standard five face landmarks used by the upstream aligner, with the source's image-coordinate orientation preserved.

The input contract follows [OpenCV's official FaceRecognizerSF implementation](https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/face_recognize.cpp): a 112 × 112 aligned image, RGB order, zero mean, unit scale. The source [SFace wrapper](https://github.com/opencv/opencv_zoo/blob/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface/sface.py) uses a 0.363 cosine threshold as a reference for pair verification. JoFaceGuard's separate Jo / uncertain / unknown bands still need validation on Jo's actual camera and enrollment; a benchmark threshold cannot guarantee a false-lock rate.

## Conversion verification

The converter checks the pinned download checksum, removes redundant initializer declarations from the old ONNX graph, then converts ONNX → PyTorch → Core ML. Four deterministic synthetic raw-RGB inputs are checked against ONNX Runtime, with cosine agreement ≥ 0.999 required for both conversion steps. It also reloads the saved Core ML package and runs a prediction. Failed verification leaves no new accepted runtime model.

The generated `build/model-cache/sface-conversion-report.json` and `Resources/Models/sface-model-manifest.json` record the observed results. These tests establish numerical conversion parity and model loading only; they do not measure Jo recognition accuracy, unfamiliar-person detection, or end-to-end lock latency. No camera images are involved in conversion tests.

Observed local verification on October 5, 2026: all four ONNX/Core ML cosines were approximately 1.000000; minimum ONNX/PyTorch cosine was 0.99999988. Saved Core ML reload and prediction passed. The model package is approximately 37 MB. Tiny floating-point rounding may report cosine slightly above 1.0 in the raw receipt.
