# JoFaceGuard 0.1.0 — validation record

Date: October 5, 2026 (America/New_York). Host: Apple Silicon / macOS 27.2 (26B5091g), Apple Swift 6.4, Command Line Tools. No full Xcode installation was used.

## Completed checks

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

The model fixtures are synthetic patterns, **not people or biometric enrollment records**. No Jo face image or embedding is committed to Git or included in the build.

## Not yet measured or exercised

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
4. Let that other person sit alone in normal light. In Observe, confirm a continuous stranger countdown reaches two seconds; pass-by shorter than two seconds must not accumulate across interruptions.
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
