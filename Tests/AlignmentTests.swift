// Adapted from FaceUnlock tools/AlignTest/SelfTest.swift (MIT).
// Copyright (c) 2026 Cheolho Kim. See LICENSE and THIRD_PARTY_NOTICES.md.
import CoreGraphics
import CoreVideo
import Foundation

/// Geometry and real bitmap-rendering regressions, without any face photographs.
enum AlignmentTests {
    private static let colors: [(UInt8, UInt8, UInt8)] = [
        (255, 40, 40), (40, 255, 40), (40, 40, 255), (255, 255, 40), (255, 40, 255)
    ]

    static func run() -> Int {
        var failures = 0
        func check(_ condition: Bool, _ name: String) {
            print("\(condition ? "PASS" : "FAIL") alignment: \(name)")
            if !condition { failures += 1 }
        }
        for degrees in [-19.0, 0.0, 15.0] {
            let angle: CGFloat = degrees * .pi / 180
            let scale: CGFloat = 2.3
            let placement = CGAffineTransform(a: scale * cos(angle), b: scale * sin(angle),
                                              c: -scale * sin(angle), d: scale * cos(angle),
                                              tx: 190, ty: 120)
            let source = ArcFaceCanonical.points.map { $0.applying(placement) }
            if let solved = SimilarityTransform.solve(from: source, to: ArcFaceCanonical.points) {
                let worst = zip(source, ArcFaceCanonical.points).map { point, target in
                    let mapped = point.applying(solved)
                    return hypot(mapped.x - target.x, mapped.y - target.y)
                }.max() ?? .infinity
                check(worst < 0.01, "similarity transform at \(degrees) degrees")
            } else {
                check(false, "similarity transform at \(degrees) degrees")
            }
        }
        check(SimilarityTransform.solve(from: [.zero, .zero],
                                         to: [CGPoint(x: 1, y: 2), CGPoint(x: 3, y: 4)]) == nil,
              "degenerate landmarks are rejected")
        check(SimilarityTransform.solve(from: [.zero], to: [.zero, .zero]) == nil,
              "mismatched landmark counts are rejected")
        failures += renderOrientation()
        return failures
    }

    private static func renderOrientation() -> Int {
        let angle: CGFloat = -8 * .pi / 180
        let scale: CGFloat = 1.9
        let placement = CGAffineTransform(a: scale * cos(angle), b: scale * sin(angle),
                                          c: -scale * sin(angle), d: scale * cos(angle),
                                          tx: 210, ty: 130)
        let source = ArcFaceCanonical.points.map { $0.applying(placement) }
        guard let buffer = frame(width: 640, height: 480, markers: source) else {
            print("FAIL alignment: synthetic frame creation")
            return 1
        }
        let landmarks = FaceLandmarks5(leftEye: source[0], rightEye: source[1], nose: source[2],
                                       leftMouth: source[3], rightMouth: source[4])
        guard let pixels = FaceAligner().alignedPixels(from: buffer, landmarks: landmarks,
                                                       applyCLAHE: false, normalize: false) else {
            print("FAIL alignment: synthetic frame rendering")
            return 1
        }
        // Original SFace / ArcFace locations in model bitmap coordinates (top-left).
        // Eyes must occupy row 52 and mouth row 92, not their vertically flipped rows.
        let expected = [CGPoint(x: 38.2946, y: 51.6963), CGPoint(x: 73.5318, y: 51.5014),
                        CGPoint(x: 56.0252, y: 71.7366), CGPoint(x: 41.5493, y: 92.3655),
                        CGPoint(x: 70.7299, y: 92.2041)]
        var failures = 0
        for index in colors.indices {
            guard let found = centroid(colors[index], pixels: pixels, size: FaceAligner.size) else {
                print("FAIL alignment: RGBA marker \(index) missing")
                failures += 1
                continue
            }
            let distance = hypot(found.x - expected[index].x, found.y - expected[index].y)
            let passed = distance < 2.0
            print(String(format: "%@ alignment: RGBA marker %d at (%.2f, %.2f), error %.2f px",
                         passed ? "PASS" : "FAIL", index, found.x, found.y, distance))
            if !passed { failures += 1 }
        }
        return failures
    }

    private static func frame(width: Int, height: Int, markers: [CGPoint]) -> CVPixelBuffer? {
        var buffer: CVPixelBuffer?
        guard CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, nil, &buffer)
                == kCVReturnSuccess, let buffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let ptr = base.bindMemory(to: UInt8.self, capacity: rowBytes * height)
        for row in 0..<height {
            for col in 0..<width {
                let offset = row * rowBytes + col * 4
                ptr[offset] = 128; ptr[offset + 1] = 128
                ptr[offset + 2] = 128; ptr[offset + 3] = 255
            }
        }
        for (index, point) in markers.enumerated() {
            let (red, green, blue) = colors[index]
            let centerX = Int(point.x.rounded())
            let centerRow = height - 1 - Int(point.y.rounded())
            for dy in -4...4 {
                for dx in -4...4 {
                    let row = centerRow + dy, col = centerX + dx
                    guard row >= 0, row < height, col >= 0, col < width else { continue }
                    let offset = row * rowBytes + col * 4
                    ptr[offset] = blue; ptr[offset + 1] = green
                    ptr[offset + 2] = red; ptr[offset + 3] = 255
                }
            }
        }
        return buffer
    }

    private static func centroid(_ color: (UInt8, UInt8, UInt8), pixels: [UInt8], size: Int) -> CGPoint? {
        var sumX = 0.0, sumY = 0.0, count = 0.0
        let (targetR, targetG, targetB) = (Double(color.0), Double(color.1), Double(color.2))
        for row in 0..<size {
            for col in 0..<size {
                let offset = (row * size + col) * 4
                let r = Double(pixels[offset]), g = Double(pixels[offset + 1]), b = Double(pixels[offset + 2])
                let distance = ((r - targetR) * (r - targetR) + (g - targetG) * (g - targetG)
                    + (b - targetB) * (b - targetB)).squareRoot()
                guard distance < 90 else { continue }
                sumX += Double(col); sumY += Double(row); count += 1
            }
        }
        return count > 0 ? CGPoint(x: sumX / count, y: sumY / count) : nil
    }
}
