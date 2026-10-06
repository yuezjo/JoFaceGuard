import Foundation
import CoreVideo

enum QualityTests {
    static func run() -> Int {
        var failures = 0
        func check(_ condition: Bool, _ name: String) {
            print("\(condition ? "PASS" : "FAIL") quality: \(name)")
            if !condition { failures += 1 }
        }
        let size = 112
        func fixture(_ value: (Int, Int) -> UInt8) -> [UInt8] {
            var pixels = [UInt8](repeating: 255, count: size * size * 4)
            for y in 0..<size {
                for x in 0..<size {
                    let i = (y * size + x) * 4
                    let v = value(x, y)
                    pixels[i] = v; pixels[i + 1] = v; pixels[i + 2] = v
                }
            }
            return pixels
        }
        func assess(_ pixels: [UInt8]) -> ImageQuality.Result {
            ImageQuality.evaluate(rgba: pixels, width: size, height: size)
        }
        let detailed = fixture { x, y in (x / 4 + y / 4) % 2 == 0 ? 90 : 170 }
        let good = assess(detailed)
        check(good.accepted && good.contrast >= 14 && good.laplacianVariance >= 35,
              "well-lit detailed input passes")

        let dim = assess(fixture { x, y in (x / 4 + y / 4) % 2 == 0 ? 10 : 35 })
        check(!dim.accepted && dim.reason.contains("Low light"),
              "raw low light is uncertain before normalization")
        let bright = assess(fixture { x, y in (x / 4 + y / 4) % 2 == 0 ? 220 : 250 })
        check(!bright.accepted && bright.reason.contains("too bright"), "overexposure is rejected")
        let flat = assess(fixture { _, _ in 128 })
        check(!flat.accepted && flat.reason.contains("contrast"), "flat gray lacks detail")
        let blurry = assess(fixture { x, _ in UInt8(70 + x) })
        check(!blurry.accepted && blurry.contrast >= 14 && blurry.reason.contains("blurred"),
              "high-contrast smooth gradient fails blur gate")
        let halfDark = assess(fixture { x, _ in x < 60 ? 20 : 200 })
        check(!halfDark.accepted && halfDark.mean > 45 && halfDark.reason.contains("Low light"),
              "large shadow region is rejected despite acceptable mean")
        let clipped = assess(fixture { x, y in x < 40 ? 255 : ((x / 4 + y / 4) % 2 == 0 ? 70 : 130) })
        check(!clipped.accepted && clipped.mean < 210 && clipped.reason.contains("too bright"),
              "large clipped region is rejected despite acceptable mean")

        var transparent = detailed
        for p in 0..<200 { transparent[p * 4 + 3] = 0 }
        check(!assess(transparent).accepted && assess(transparent).reason.contains("incomplete"),
              "partly transparent crop is rejected")
        var darkDetail = detailed
        for p in 0..<400 {
            darkDetail[p * 4] = 0; darkDetail[p * 4 + 1] = 0; darkDetail[p * 4 + 2] = 0
        }
        check(assess(darkDetail).accepted,
              "real dark details are not mistaken for missing crop coverage")
        let black = assess(fixture { _, _ in 0 })
        check(!black.accepted && black.reason.contains("Low light"),
              "fully black opaque crop remains rejected for low light")
        check(!assess([]).accepted, "incorrect pixel count fails closed")
        check(!ImageQuality.evaluate(rgba: [], width: Int.max, height: Int.max).accepted,
              "invalid dimensions fail without overflow")

        // Exercise the real analyze entrypoint: a dark camera is uncertain even
        // without a detected face, while a usable blank frame is genuinely noFace.
        let pipeline = FacePipeline()
        for (value, expectedCount, expectedReason) in [(UInt8(0), -1, "Low light"),
                                                      (UInt8(35), -1, "Low light"),
                                                      (UInt8(255), -1, "too bright"),
                                                      (UInt8(128), 0, "No face")] {
            if let buffer = blankFrame(value) {
                let result = pipeline.analyze(buffer)
                check(result.faceCount == expectedCount && !result.qualityOK && result.pixels == nil
                      && result.reason.contains(expectedReason),
                      "raw blank frame \(value) reports \(expectedReason)")
            } else { check(false, "synthetic camera buffer creation") }
        }
        return failures
    }

    private static func blankFrame(_ value: UInt8) -> CVPixelBuffer? {
        var buffer: CVPixelBuffer?
        guard CVPixelBufferCreate(nil, 640, 480, kCVPixelFormatType_32BGRA, nil, &buffer)
                == kCVReturnSuccess, let buffer,
              CVPixelBufferLockBaseAddress(buffer, []) == kCVReturnSuccess else { return nil }
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        for y in 0..<480 {
            for x in 0..<640 {
                let offset = y * rowBytes + x * 4
                bytes[offset] = value; bytes[offset + 1] = value; bytes[offset + 2] = value
                bytes[offset + 3] = 255
            }
        }
        return buffer
    }
}
