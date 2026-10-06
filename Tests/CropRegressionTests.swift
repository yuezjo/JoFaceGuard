import CoreGraphics
import CoreImage
import CoreVideo
import Foundation
import ImageIO
import Vision

/// Public-photo integration regressions: real dark image content is not missing
/// pixels, and an oversized detector box is not an incomplete aligned face.
/// No camera, personal profile, network, or screen-lock API is used.
@main
struct CropRegressionTests {
    static func main() throws {
        let fixtureDirectory = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : "Tests/Fixtures"
        let collins = try portrait(fixtureDirectory + "/astronaut.png")
        let armstrong = try portrait(fixtureDirectory + "/armstrong.jpg")
        let pipeline = FacePipeline()
        let collinsResult = pipeline.analyze(try frame(collins))
        let armstrongResult = pipeline.analyze(try frame(armstrong))
        try require(collinsResult.qualityOK, "Collins portrait accepted: \(collinsResult.reason)")
        try require(armstrongResult.qualityOK, "Armstrong portrait accepted: \(armstrongResult.reason)")
        guard let collinsPixels = collinsResult.pixels, let armstrongPixels = armstrongResult.pixels else {
            throw GuardError.message("Accepted portraits must supply identity pixels")
        }
        let blackFraction = Double(stride(from: 0, to: armstrongPixels.count, by: 4).filter {
            armstrongPixels[$0] <= 2 && armstrongPixels[$0 + 1] <= 2 && armstrongPixels[$0 + 2] <= 2
        }.count) / Double(112 * 112)
        try require(blackFraction > 0.02, "Armstrong still exercises the previous 2% black-pixel false rejection")
        try require(missingCount(armstrongPixels) == 0, "Dark content in Armstrong portrait has full alpha coverage")
        print(String(format: "PASS opaque dark portrait: %.2f%% black pixels; complete aligned face accepted", blackFraction * 100))

        // The box extends below this real image boundary while all five landmarks
        // and the complete aligned crop remain visible. Keep original observation
        // coordinates. Future Vision revisions can change the detected geometry.
        let edgeFrame = try frame(collins, cropBottom: 674)
        let detection = VNDetectFaceRectanglesRequest()
        detection.revision = VNDetectFaceRectanglesRequestRevision3
        try VNImageRequestHandler(cvPixelBuffer: edgeFrame, orientation: .up).perform([detection])
        guard let originalBounds = detection.results?.first?.boundingBox else {
            throw GuardError.message("Edge fixture must contain a detectable face")
        }
        try require(originalBounds.minY < 0, "Edge fixture still exercises an overflowing Vision box")
        let edgeResult = pipeline.analyze(edgeFrame)
        try require(edgeResult.qualityOK, "Visible edge face accepted: \(edgeResult.reason)")
        guard let edgePixels = edgeResult.pixels, let visibleBounds = edgeResult.bounds else {
            throw GuardError.message("Accepted edge face must supply pixels and tracking bounds")
        }
        try require(missingCount(edgePixels) == 0, "Accepted edge face has no missing aligned pixels")
        try require(visibleBounds.minX >= 0 && visibleBounds.minY >= 0
                    && visibleBounds.maxX <= 1 && visibleBounds.maxY <= 1,
                    "Tracking evidence bounds are clipped into the image")
        try require(visibleBounds.minY == 0, "Overflowing bottom of tracking box is clipped to zero")
        print(String(format: "PASS visible edge face: original detector minY %.2f px; aligned pixels complete",
                     originalBounds.minY * Double(CVPixelBufferGetHeight(edgeFrame))))

        // Same real photo farther off-frame: landmarks remain detectable but the
        // aligned crop now contains genuinely missing transparent pixels.
        let cutOff = pipeline.analyze(try frame(collins, cropBottom: 700))
        try require(!cutOff.qualityOK && cutOff.pixels == nil, "Actual incomplete crop cannot reach identity inference")
        try require(cutOff.reason.contains("crop is incomplete"), "Real cut-off fixture exercises alpha coverage: \(cutOff.reason)")
        let dimmed = pipeline.analyze(try frame(armstrong, dimmed: true))
        try require(!dimmed.qualityOK && dimmed.pixels == nil && dimmed.reason.contains("Low light"),
                    "Dim portrait stays uncertain and supplies no identity pixels")
        print("PASS negative controls: real incomplete crop and dim portrait rejected")

        // Actual model features, not invented vectors. Repetition only satisfies
        // the minimum sample count; this is not a population-accuracy benchmark.
        if let modelPath = ProcessInfo.processInfo.environment["JO_FACE_GUARD_MODEL"] {
            let model = try EmbeddingModel(url: URL(fileURLWithPath: modelPath))
            let collinsEmbedding = try model.embed(collinsPixels)
            let armstrongEmbedding = try model.embed(armstrongPixels)
            let edgeEmbedding = try model.embed(edgePixels)
            let classifier = FaceClassifier()
            try require(classifier.joThreshold == 0.50 && classifier.unknownThreshold == 0.20,
                        "Identity thresholds have not been relaxed for crop fixes")
            let collinsEnrollment = Array(repeating: collinsEmbedding, count: 15)
            try require(classifier.classify(embedding: edgeEmbedding, enrollment: collinsEnrollment) == .jo,
                        "Accepted edge crop retains the same identity")
            try require(classifier.classify(embedding: armstrongEmbedding, enrollment: collinsEnrollment) == .unknown,
                        "Different accepted real face reaches Unknown at unchanged thresholds")
            let armstrongEnrollment = Array(repeating: armstrongEmbedding, count: 15)
            let unknown = classifier.classify(embedding: edgeEmbedding, enrollment: armstrongEnrollment)
            try require(unknown == .unknown, "Edge crop also distinguishes the other reference identity")
            var policy = GuardPolicy()
            var lockDecisions = 0
            for index in 0...11 {
                let time = Double(index) * 0.2
                let decision = policy.consume(FrameEvidence(classification: unknown, timestamp: time,
                    faceBounds: visibleBounds, embedding: edgeEmbedding), now: time + 0.01)
                try require(decision.classification == .unknown, "Clipped tracking bounds preserve Unknown")
                if decision.shouldLock { lockDecisions += 1 }
            }
            try require(lockDecisions == 1, "Unchanged two-second policy emits one decision for accepted edge stranger")
            print(String(format: "PASS actual SFace features: cross-person cosine %.5f; edge self cosine %.5f; one 2-second decision (no lock API)",
                VectorMath.cosineSimilarity(collinsEmbedding, armstrongEmbedding),
                VectorMath.cosineSimilarity(collinsEmbedding, edgeEmbedding)))
        } else {
            print("SKIP optional model comparison: set JO_FACE_GUARD_MODEL to enable")
        }
        print("PASS crop regressions: real portraits, frame-edge geometry, missing pixels, and low light")
    }

    private static func portrait(_ path: String) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw GuardError.message("Cannot read public fixture: \(path)")
        }
        return image
    }

    /// Normalize to 1024 pixels wide, preserving aspect ratio. cropBottom uses
    /// Core Image's bottom-left coordinates and removes source pixels only.
    private static func frame(_ portrait: CGImage, cropBottom: Int = 0,
                              dimmed: Bool = false) throws -> CVPixelBuffer {
        let width = 1024
        let scale = Double(width) / Double(portrait.width)
        let height = Int(Double(portrait.height) * scale) - cropBottom
        guard height > 0 else { throw GuardError.message("Invalid test crop") }
        var candidate: CVPixelBuffer?
        let attributes = [kCVPixelBufferCGImageCompatibilityKey: true,
                          kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                  kCVPixelFormatType_32BGRA, attributes, &candidate) == kCVReturnSuccess,
              let buffer = candidate else { throw GuardError.message("Cannot create public-fixture buffer") }
        let image = CIImage(cgImage: portrait)
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            .transformed(by: CGAffineTransform(translationX: 0, y: -CGFloat(cropBottom)))
        CIContext().render(image, to: buffer, bounds: CGRect(x: 0, y: 0, width: width, height: height),
                           colorSpace: CGColorSpaceCreateDeviceRGB())
        if dimmed {
            guard CVPixelBufferLockBaseAddress(buffer, []) == kCVReturnSuccess else {
                throw GuardError.message("Cannot access test buffer")
            }
            defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
            guard let base = CVPixelBufferGetBaseAddress(buffer) else { throw GuardError.message("Missing test pixels") }
            let bytes = base.assumingMemoryBound(to: UInt8.self)
            let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
            for y in 0..<height {
                for x in 0..<width {
                    for channel in 0..<3 { bytes[y * rowBytes + x * 4 + channel] /= 10 }
                }
            }
        }
        return buffer
    }

    private static func missingCount(_ pixels: [UInt8]) -> Int {
        stride(from: 3, to: pixels.count, by: 4).filter { pixels[$0] < 250 }.count
    }

    private static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw GuardError.message(message) }
    }
}
