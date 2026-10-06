# JoFaceGuard — validation record

Date: October 5, 2026 (America/New_York). Host: Apple Silicon / macOS 27.2 (26B5091g), Apple Swift 6.4, Command Line Tools. No full Xcode installation was used.

## 0.1.3 uncertain-face regression fix

Jo reported another person remained uncertain with automatic locking enabled, and clarified that the person was standing with their full face in view. Reading the running app's accessibility text confirmed `守卫 · 不确定`, `守卫中` and `请让整张脸进入画面。` The old message covered multiple quality failures, so this establishes a pre-classification quality rejection, not the exact gate hit during Jo's test. Standing versus sitting is not an input to the classifier.

Two false rejections were reproduced with public-domain NASA portraits. The Collins edge fixture has a detector box extending 6.43 pixels outside the image, but complete landmarks and zero missing aligned pixels. The Armstrong portrait has 2.87% opaque black pixels, despite good illumination and detail. The old margin and black-pixel gates reject these usable faces. The fix checks landmark geometry and actual aligned alpha coverage, clips only emitted tracking bounds, and preserves pose, size, lighting, blur and capture-quality gates. A genuinely cut-off portrait and a dimmed portrait remain rejected.

Actual SFace inference gave cross-person cosine approximately 0.080–0.082 across runs (Unknown at the unchanged 0.20 boundary) and cropped same-person cosine 0.98845 (Jo at the unchanged 0.50 boundary). The accepted edge stranger emitted one lock decision after two controlled seconds without invoking any lock API. Repeated reference vectors satisfy enrollment count only; these two public photos are not a population-accuracy benchmark or a replacement for webcam testing.

The UI now distinguishes quality failure, unavailable/invalid enrollment, failed feature extraction, ambiguous similarity, and invalid continuity evidence. No face images or embeddings are written to diagnostic logs. The classifier/timer suite now passes 115 checks, including diagnostic result consistency; quality/alignment checks also pass. Native release compilation and bundled-model smoke passed. Real stranger recognition and actual screen locking after this repair still require Jo's follow-up test.

```sh
make test
JO_FACE_GUARD_MODEL="$PWD/Resources/Models/SFace.mlpackage" make crop-test
make release
```

## 0.1.1 enrollment regression fix

Jo reported that enrollment never advanced despite adjusting head position. Reproduction with the NASA public-domain astronaut fixture confirmed that the landmarks-only request implicitly used rectangle detector revision 2: `pitch` was nil, so the old pipeline rejected the portrait with `Head pose unavailable`. The UI incorrectly translated missing data into a request to keep turning the head.

The pipeline now explicitly runs `VNDetectFaceRectanglesRequestRevision3` and passes its observations into the landmarks request. The reference portrait now has finite yaw/pitch/roll. The revision-3 detector assigns confidence about 0.876 to this usable portrait, so the detection gate changed from 0.90 to 0.80 while preserving the independent pose, face-size, capture-quality, raw-light, blur, crop and identity gates. It remains a provisional threshold, not a measured accuracy claim.

Added `Tests/EnrollmentRegressionTests.swift` and the public-domain reference image with its source, checksum and attribution. The production pipeline passes this portrait at 1024×1024 and 1280×720 camera dimensions and rejects a dimmed version as uncertain. Actual Core ML inference on the accepted aligned face returns 128 finite unit-normalized features. This covers real face detection missing from the initial synthetic tests; it does not substitute for Jo's camera acceptance test.

The updated UI distinguishes unavailable pose data from an actually unsuitable head angle, explicitly says when a frame is ready to capture, and shows version 0.1.1. Duplicate launches activate the existing instance to avoid camera contention. No Jo camera image was captured, saved or committed during this repair.

```sh
make test
JO_FACE_GUARD_MODEL="$PWD/Resources/Models/SFace.mlpackage" ./build/enrollment-regression-tests Tests/Fixtures/astronaut.png
```

## Initial release checks

| Check | Observed result |
| --- | --- |
| GitHub fork | `yuezjo/JoFaceGuard`, fork of `KINN-CH/FaceUnlock`; preserved history and MIT license. |
| Native release compilation | Passed, `arm64-apple-macos14.0`, optimized Swift 5 language mode. |
| Local app signing | Ad-hoc signed; strict/deep codesign verification passed in clean staging and `/Applications`. Not Developer ID signed or notarized. |
| Classifier / timer tests | **86 deterministic checks passed**. Covers Jo/unknown/ambiguous/invalid embeddings, bad enrollment, exact threshold behavior, ≥2 seconds, ≥8 frames, one trigger per sequence, interruptions, stale frames, nonmonotonic timestamps, gaps, changed face identity and bounds, invalid policy configurations. |
| Quality / alignment tests | **25 checks passed**. Covers dim/black/bright input, contrast, blur, incomplete crops, invalid inputs, no-face Vision path, geometry transforms, and colored landmark markers. Maximum observed marker error 1.22 px on a 112 px crop. |
| SFace conversion | Pinned ONNX SHA-256 verified; 4 deterministic synthetic inputs checked ONNX Runtime → PyTorch → Core ML with minimum agreement approximately 0.99999988. Saved model reload passed. |
| Swift model input integration | 3 deterministic synthetic RGBA images matched independent ONNX reference embeddings, cosine approximately 1.0. Intentionally wrong RGB/BGR or row ordering yielded lower cosine (0.696–0.879), so the checks distinguish those common preprocessing bugs. Alpha ignored and invalid image size rejected. |
| Bundled app smoke | Loaded the bundled Core ML model; returned 128 finite normalized values; repeat prediction cosine approximately 1.0. No camera opened, Keychain accessed, or lock requested by this smoke command. |
| Lock API availability | `SACLockScreenImmediate` resolved successfully on this host. It was **not invoked** during validation. |
| Native UI smoke | App created the native menu/window, loaded the model, rendered its own paused window, and confirmed monitoring=false and armed=false. The UI startup path reads the app's own Keychain item; no enrollment was present or created. |
| Runtime source inspection | No URLSession/URLRequest, password injection, keyboard event synthesis, update download or runtime upload path. Model build scripts download public model/dependency assets only. |
| Review fixes | Serialized profile saves/deletes, rejected stale storage callbacks, allowed recovery from unreadable/incompatible profiles, invalidated old lock-status callbacks, and based arming status on validated policy output. Independent re-review and full typecheck passed. |

The model-conversion fixtures are synthetic patterns. Subsequent enrollment/crop tests use attributed public NASA portraits, never Jo's images or biometric enrollment records. No Jo face image or embedding is committed to Git or included in the build.

## Initial release gaps and current acceptance status

Jo has since used enrollment and enabled automatic locking, but reported the unfamiliar-person test failed by remaining uncertain. That feedback does not establish a successful profile round trip, reliable recognition, or an actual lock. The following have not been systematically validated by the implementation tests:

- Real camera permission flow, actual webcam capture, camera sharing/interruption, and external-camera orientation.
- Jo enrollment and saved-profile round trip using Jo's genuine face, including glasses and varied lighting.
- Real unfamiliar-person recognition, end-to-end latency from sitting down, or continuous-session false-lock rate.
- An actual macOS lock request and observed transition, unlock, sleep/wake, or fast user switching.
- Long-duration CPU/battery use, spoof resistance, or behavior on another macOS version.

The two-second rule is validated with controlled timestamps. It does not guarantee a lock exactly two seconds after a person enters the room; sufficient visible face quality, confident classification and fresh frames must first be available.

## Manual acceptance with Jo

Use **Observe** first; it cannot lock.

1. Enroll all five instructed poses with one person visible and ordinary desk lighting. Cancel mid-enrollment once and verify the old complete profile remains.
2. Observe Jo for several minutes: neutral face, slight turns, glasses, ordinary working distance. Record any Unknown results; do not enable automatic locking if Jo becomes Unknown.
3. Cover the camera, dim the room, move out of frame, turn far sideways, and invite a second consenting person into the frame. Confirm no countdown survives uncertainty; complete darkness must read uncertain.
4. Let that other person appear alone in normal light, standing or sitting with a sufficiently large, clear, near-frontal face. In Observe, confirm a continuous stranger countdown reaches two seconds; pass-by shorter than two seconds must not accumulate across interruptions.
5. Return Jo to the camera; after Jo is recognized, enable automatic locking. Repeat the controlled stranger test with work saved. Verify the real lock screen appears and the guard stays paused after unlock.
6. Pause/quit during a countdown; disconnect the camera or open another camera app. Confirm no stale result locks the screen. Test sleep/wake and user switching.

If any of these fail, leave automatic locking off and keep the result as a calibration/compatibility issue. Do not weaken uncertainty handling just to obtain a faster lock.

## Reproduction

```sh
make test
make model-test
make release
make smoke
```

Installed-app UI smoke (takes a window-only screenshot, never starts the camera):

```sh
/Applications/JoFaceGuard.app/Contents/MacOS/JoFaceGuard --ui-smoke-test /tmp/JoFaceGuard-paused.png
```

Release ZIP includes the MIT notice, full Apache-2.0 model license and model/source attribution. Build artifacts and local conversion environments are excluded from the Git commit.
