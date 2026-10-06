#!/usr/bin/env python3
"""Fetch pinned Apache-2.0 SFace and verify a local Core ML conversion.

The ONNX model and license come from OpenCV Zoo. No face images are downloaded
or uploaded. Core ML output is accepted only after comparison to the original
ONNX Runtime output on several deterministic synthetic RGB tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
REVISION = "47534e27c9851bb1128ccc0102f1145e27f23f98"
FILENAME = "face_recognition_sface_2021dec.onnx"
SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
URL = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{REVISION}/models/face_recognition_sface/{FILENAME}"
OUTPUT = ROOT / "Resources/Models/SFace.mlpackage"
CACHE = ROOT / "build/model-cache"
SHAPE = (1, 3, 112, 112)
DIMENSION = 128
MIN_COSINE = 0.999


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        partial = destination.with_suffix(".part")
        subprocess.run(["/usr/bin/curl", "--fail", "--location", "--retry", "3", "--output", str(partial), URL], check=True)
        partial.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"Model checksum mismatch: expected {SHA256}, got {digest}. Remove {destination} and retry.")
    print(f"Verified source model SHA-256: {digest}", flush=True)


def convert_child(source: Path, output: Path, report: Path) -> None:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import coremltools as ct
    from onnx2torch import convert

    torch.set_num_threads(2)
    graph = onnx.load(str(source))
    # Old MXNet exports list constant parameters as graph inputs. They remain
    # initializers; removing only duplicate input declarations avoids treating
    # hundreds of weights as user inputs during the PyTorch trace.
    initializer_names = {item.name for item in graph.graph.initializer}
    inputs = [item for item in graph.graph.input if item.name not in initializer_names]
    del graph.graph.input[:]
    graph.graph.input.extend(inputs)
    if len(inputs) != 1:
        raise RuntimeError(f"Expected one image input, got {[i.name for i in inputs]}")
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(graph.SerializeToString(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    network = convert(graph).eval()
    rng = np.random.default_rng(42)
    cases = [rng.uniform(0, 255, SHAPE).astype(np.float32) for _ in range(3)]
    gradient = np.linspace(10, 230, 112, dtype=np.float32)
    cases.append(np.broadcast_to(gradient.reshape(1, 1, 1, 112), SHAPE).copy())
    example = torch.from_numpy(cases[0])
    with torch.no_grad():
        traced = torch.jit.trace(network, example, strict=False)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="input", shape=SHAPE, dtype=np.float32)],
        outputs=[ct.TensorType(name="embedding", dtype=np.float32)],
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS14,
    )
    mlmodel.short_description = "SFace MobileFaceNet from OpenCV Zoo, converted for JoFaceGuard. Raw RGB NCHW 1x3x112x112, values 0–255. 128-d output."
    mlmodel.author = "SFace: Yaoyao Zhong; ONNX: Chengrui Wang / OpenCV Zoo; Core ML conversion: JoFaceGuard"
    mlmodel.license = "Apache-2.0. See Resources/SFACE_LICENSE.txt and docs/UPSTREAM.md."
    mlmodel.user_defined_metadata["source_revision"] = REVISION
    mlmodel.user_defined_metadata["source_sha256"] = SHA256
    mlmodel.user_defined_metadata["preprocessing"] = "RGB 0–255 NCHW; normalization already inside model. Do not apply ArcFace normalization."
    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output))
    receipts = []
    for index, case in enumerate(cases):
        reference = np.asarray(session.run(None, {input_name: case})[0]).reshape(-1)
        with torch.no_grad():
            intermediate = network(torch.from_numpy(case)).detach().numpy().reshape(-1)
        actual = np.asarray(mlmodel.predict({"input": case})["embedding"]).reshape(-1)
        if reference.size != DIMENSION or actual.size != DIMENSION:
            raise RuntimeError(f"Wrong output shape: ONNX {reference.shape}, CoreML {actual.shape}")
        if not all(np.all(np.isfinite(x)) and np.linalg.norm(x) > 1e-8 for x in [reference, intermediate, actual]):
            raise RuntimeError("Model output was nonfinite or zero")
        cosine = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        ort_torch = cosine(reference, intermediate)
        ort_coreml = cosine(reference, actual)
        if min(ort_torch, ort_coreml) < MIN_COSINE:
            raise RuntimeError(f"Conversion parity failed: ONNX/PyTorch={ort_torch}, ONNX/CoreML={ort_coreml}")
        receipts.append({"case": index, "onnx_vs_torch_cosine": ort_torch, "onnx_vs_coreml_cosine": ort_coreml})
        print(f"Case {index}: ONNX→Torch {ort_torch:.9f}; ONNX→CoreML {ort_coreml:.9f}", flush=True)
    # A saved runtime reload must work as well as the converter's model object.
    reloaded = ct.models.MLModel(str(output), compute_units=ct.ComputeUnit.CPU_ONLY)
    again = np.asarray(reloaded.predict({"input": cases[0]})["embedding"]).reshape(-1)
    if again.size != DIMENSION or not np.isfinite(again).all():
        raise RuntimeError("Saved model reload smoke test failed")
    report.write_text(json.dumps({
        "source_url": URL, "source_revision": REVISION, "source_sha256": SHA256,
        "model": "SFace", "license": "Apache-2.0", "input": "RGB raw 0–255 NCHW float32", "input_shape": SHAPE,
        "output_dimension": DIMENSION, "precision": "FLOAT32", "minimum_cosine": MIN_COSINE,
        "synthetic_parity": receipts, "saved_model_reload": "passed",
        "note": "Numerical conversion and load tests; not measured human identification accuracy or latency.",
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing official ONNX file; checksum is still verified")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--report", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    source = (args.source or CACHE / FILENAME).resolve()
    output = args.output.resolve()
    CACHE.mkdir(parents=True, exist_ok=True)
    report = args.report or CACHE / "sface-conversion-report.json"
    if args.child:
        convert_child(source, output, report)
        return
    download(source)
    # Keep conversion isolated from later Core ML loads; framework conversion
    # failures must never leave a model accepted by the app.
    temporary = output.with_name("SFace.pending.mlpackage")
    result = subprocess.run([sys.executable, __file__, "--child", "--source", str(source), "--output", str(temporary), "--report", str(report)])
    if result.returncode != 0:
        shutil.rmtree(temporary, ignore_errors=True)
        raise SystemExit("SFace conversion failed. No new runtime model was installed.")
    if output.exists():
        shutil.rmtree(output)
    temporary.replace(output)
    manifest = output.parent / "sface-model-manifest.json"
    shutil.copyfile(report, manifest)
    print(f"Validated SFace model: {output}\nConversion receipt: {report}", flush=True)


if __name__ == "__main__":
    main()
