import CoreGraphics
import CoreImage
import CoreVideo
import Foundation
import ImageIO
import Vision

/// Exercises an actual detectable public-domain face through the production path.
/// Synthetic marker tests cannot catch Vision returning nil pose fields for faces.
/// No camera access, enrollment storage, networking, or screen locking occurs here.
@main
struct EnrollmentRegressionTests {
    static func main() throws {
        let path = CommandLine.arguments.count > 1
            ? CommandLine.arguments[1] : "Tests/Fixtures/astronaut.png"
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let portrait = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw GuardError.message("Cannot read public-domain astronaut fixture")
        }
        try require(portrait.width == 512 && portrait.height == 512, "Expected 512 x 512 source fixture")
        let pipeline = FacePipeline()
        let square = try frame(portrait, cameraDimensions: false)

        // Log the previous API behavior for diagnosis; future OS revisions may fix
        // it, so nil pitch is not itself a requirement for this regression test.
        let oldRequest = VNDetectFaceLandmarksRequest()
        oldRequest.revision = VNDetectFaceLandmarksRequestRevision3
        try VNImageRequestHandler(cvPixelBuffer: square, orientation: .up).perform([oldRequest])
        print("INFO landmarks-only pitch: \(String(describing: oldRequest.results?.first?.pitch)); explicit rectangle revision 3 remains required")

        var usablePixels: [UInt8]?
        for (name, buffer) in [("1024 x 1024 portrait", square),
                               ("1280 x 720 camera-sized portrait", try frame(portrait, cameraDimensions: true))] {
            let result = pipeline.analyze(buffer)
            try require(result.faceCount == 1, "\(name): exactly one face; got \(result.faceCount)")
            try require(result.qualityOK, "\(name): production pipeline accepts clear face; got \(result.reason)")
            try require(result.pixels?.count == 112 * 112 * 4, "\(name): aligned RGBA pixels reach enrollment")
            try require(result.yaw?.isFinite == true && result.pitch?.isFinite == true,
                        "\(name): explicit detector supplies finite yaw and pitch")
            try require(result.bounds != nil, "\(name): detection bounds survive analysis")
            usablePixels = result.pixels
            print("PASS enrollment portrait: \(name), yaw \(result.yaw!), pitch \(result.pitch!)")
        }

        let dark = try frame(portrait, cameraDimensions: true, dimmed: true)
        let rejected = pipeline.analyze(dark)
        try require(!rejected.qualityOK && rejected.pixels == nil,
                    "Dimmed portrait remains unusable and supplies no identity pixels")
        try require(rejected.faceCount != 0 && rejected.reason.contains("Low light"),
                    "Dimmed portrait is uncertain for low light, never confident no-face")
        print("PASS enrollment negative control: dimmed portrait → \(rejected.reason)")

        if let modelPath = ProcessInfo.processInfo.environment["JO_FACE_GUARD_MODEL"] {
            let model = try EmbeddingModel(url: URL(fileURLWithPath: modelPath))
            guard let pixels = usablePixels else { throw GuardError.message("No accepted portrait pixels") }
            let vector = try model.embed(pixels)
            try require(vector.count == EmbeddingModel.dimension && vector.allSatisfy(\.isFinite),
                        "Accepted portrait produces 128 finite Core ML features")
            let norm = sqrt(vector.reduce(Float(0)) { $0 + $1 * $1 })
            try require(abs(norm - 1) < 0.0001, "Portrait embedding has unit length")
            print("PASS enrollment model: real portrait → 128 finite features; norm \(norm)")
        } else {
            print("SKIP optional portrait model inference: set JO_FACE_GUARD_MODEL to enable")
        }
        print("PASS enrollment regression: real-face detection, pose, quality, alignment, and low-light rejection")
    }

    private static func frame(_ portrait: CGImage, cameraDimensions: Bool,
                              dimmed: Bool = false) throws -> CVPixelBuffer {
        let width = cameraDimensions ? 1280 : 1024
        let height = cameraDimensions ? 720 : 1024
        let bounds = CGRect(x: 0, y: 0, width: width, height: height)
        var candidate: CVPixelBuffer?
        let attributes = [kCVPixelBufferCGImageCompatibilityKey: true,
                          kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, width, height,
                                  kCVPixelFormatType_32BGRA, attributes, &candidate) == kCVReturnSuccess,
              let buffer = candidate else { throw GuardError.message("Cannot create fixture pixel buffer") }
        var image = CIImage(cgImage: portrait).transformed(by: CGAffineTransform(scaleX: 2, y: 2))
        if cameraDimensions {
            // Keep the face in the original top part of the photograph, center the
            // 1024-pixel-wide crop on a neutral canvas with native camera dimensions.
            image = image.transformed(by: CGAffineTransform(translationX: 128, y: -304))
                .composited(over: CIImage(color: CIColor(red: 0.5, green: 0.5, blue: 0.5)).cropped(to: bounds))
        }
        CIContext().render(image, to: buffer, bounds: bounds, colorSpace: CGColorSpaceCreateDeviceRGB())
        if dimmed {
            guard CVPixelBufferLockBaseAddress(buffer, []) == kCVReturnSuccess else {
                throw GuardError.message("Cannot access fixture pixel buffer")
            }
            defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
            guard let base = CVPixelBufferGetBaseAddress(buffer) else {
                throw GuardError.message("Missing fixture pixel data")
            }
            let bytes = base.assumingMemoryBound(to: UInt8.self)
            let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
            for y in 0..<height {
                for x in 0..<width {
                    let offset = y * rowBytes + x * 4
                    for channel in 0..<3 { bytes[offset + channel] /= 10 }
                }
            }
        }
        return buffer
    }

    private static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw GuardError.message(message) }
    }
}
