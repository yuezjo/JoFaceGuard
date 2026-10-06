import CoreGraphics
import CoreVideo
import Foundation
import Vision

/// A failed observation never supplies pixels to the identity classifier.
struct FaceAnalysis {
    /// -1 means unusable input or detection failed; 0 means a usable frame contained no faces.
    let faceCount: Int
    let qualityOK: Bool
    let reason: String
    let pixels: [UInt8]?
    /// Vision coordinates: normalized, bottom-left origin.
    let bounds: CGRect?
    /// Radians, as reported by Vision.
    let yaw: Double?
    let pitch: Double?
}

/// Use this same instance/path for enrollment and recognition, on one serial queue.
/// No captured image is saved or sent over the network.
final class FacePipeline {
    private let aligner = FaceAligner()

    func analyze(_ buffer: CVPixelBuffer) -> FaceAnalysis {
        // Darkness can prevent detection altogether. Check the raw camera frame
        // before Vision so a covered camera never becomes a confident "No face".
        if let reason = Self.frameLightingRejection(buffer) {
            return rejected(count: -1, reason: reason)
        }
        let handler = VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up)
        let landmarksRequest = VNDetectFaceLandmarksRequest()
        landmarksRequest.revision = VNDetectFaceLandmarksRequestRevision3

        do {
            try handler.perform([landmarksRequest])
        } catch {
            return rejected(count: -1, reason: "Face detection unavailable")
        }

        guard let faces = landmarksRequest.results else {
            return rejected(count: -1, reason: "Face detection returned no result")
        }
        guard !faces.isEmpty else { return rejected(count: 0, reason: "No face") }
        // Selecting only the largest face can hide Jo beside a visitor. Version 1
        // intentionally makes every multiple-face frame uncertain.
        guard faces.count == 1, let face = faces.first else {
            return rejected(count: faces.count, reason: "Multiple faces — uncertain")
        }

        func reject(_ reason: String) -> FaceAnalysis {
            rejected(count: 1, reason: reason, face: face)
        }
        guard face.confidence.isFinite, face.confidence >= 0.90 else {
            return reject("Face detection confidence is low")
        }

        let width = CGFloat(CVPixelBufferGetWidth(buffer))
        let height = CGFloat(CVPixelBufferGetHeight(buffer))
        let box = face.boundingBox
        guard width > 0, height > 0,
              [box.minX, box.minY, box.maxX, box.maxY].allSatisfy(\.isFinite),
              min(box.width * width, box.height * height) >= 120 else {
            return reject("Face is too small — move closer")
        }
        guard box.minX * width >= 5, box.minY * height >= 5,
              box.maxX * width <= width - 5, box.maxY * height <= height - 5 else {
            return reject("Face is too close to the frame edge")
        }
        guard let yaw = face.yaw?.doubleValue,
              let pitch = face.pitch?.doubleValue,
              let roll = face.roll?.doubleValue,
              [yaw, pitch, roll].allSatisfy(\.isFinite) else {
            return reject("Head pose unavailable")
        }
        guard abs(yaw) <= 25 * .pi / 180,
              abs(pitch) <= 20 * .pi / 180,
              abs(roll) <= 20 * .pi / 180 else {
            return reject("Look toward the camera")
        }

        // Landmarks alone do not populate faceCaptureQuality. Run the dedicated
        // request and require a result; a nil quality is never a passing score.
        let qualityRequest = VNDetectFaceCaptureQualityRequest()
        qualityRequest.inputFaceObservations = [face]
        do {
            try handler.perform([qualityRequest])
        } catch {
            return reject("Face quality assessment unavailable")
        }
        guard qualityRequest.results?.count == 1,
              let quality = qualityRequest.results?.first?.faceCaptureQuality,
              quality.isFinite else {
            return reject("Face quality unavailable")
        }
        guard quality >= 0.30 else { return reject("Face capture quality is low") }

        guard let points = FaceLandmarks5(observation: face,
                                          imageSize: CGSize(width: width, height: height)),
              Self.valid(points, width: width, height: height) else {
            return reject("Facial landmarks are incomplete")
        }
        // SFace receives raw aligned RGB. Exposure correction can make a nearly
        // black image look usable, so it must not precede the raw-light gate.
        guard let pixels = aligner.alignedPixels(from: buffer, landmarks: points,
                                                 applyCLAHE: false, normalize: false) else {
            return reject("Face alignment failed")
        }
        let imageQuality = ImageQuality.evaluate(rgba: pixels, width: FaceAligner.size,
                                             height: FaceAligner.size)
        guard imageQuality.accepted else { return reject(imageQuality.reason) }
        return FaceAnalysis(faceCount: 1, qualityOK: true, reason: "Face quality accepted",
                            pixels: pixels, bounds: face.boundingBox, yaw: yaw, pitch: pitch)
    }

    private func rejected(count: Int, reason: String, face: VNFaceObservation? = nil) -> FaceAnalysis {
        FaceAnalysis(faceCount: count, qualityOK: false, reason: reason, pixels: nil,
                     bounds: face?.boundingBox, yaw: face?.yaw?.doubleValue,
                     pitch: face?.pitch?.doubleValue)
    }

    /// Roughly 64 x 48 samples avoid copying full camera frames. CameraCapture
    /// explicitly requests BGRA; unsupported or unreadable input fails uncertain.
    private static func frameLightingRejection(_ buffer: CVPixelBuffer) -> String? {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        guard CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA,
              width > 0, height > 0, width <= 16_384, height <= 16_384,
              CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess else {
            return "Camera frame unavailable — uncertain"
        }
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        guard rowBytes >= width * 4, let base = CVPixelBufferGetBaseAddress(buffer) else {
            return "Camera frame unavailable — uncertain"
        }
        let bytes = base.assumingMemoryBound(to: UInt8.self)
        var total = 0.0, sum = 0.0, dark = 0.0, clipped = 0.0
        for y in stride(from: 0, to: height, by: max(1, height / 48)) {
            for x in stride(from: 0, to: width, by: max(1, width / 64)) {
                let offset = y * rowBytes + x * 4
                let value = 0.114 * Double(bytes[offset]) + 0.587 * Double(bytes[offset + 1])
                    + 0.299 * Double(bytes[offset + 2])
                total += 1; sum += value
                if value < 30 { dark += 1 }
                if value > 245 { clipped += 1 }
            }
        }
        if sum / total < 45 || dark / total > 0.80 { return "Low light — uncertain" }
        if sum / total > 235 || clipped / total > 0.90 {
            return "Lighting is too bright — uncertain"
        }
        return nil
    }

    private static func valid(_ points: FaceLandmarks5, width: CGFloat, height: CGFloat) -> Bool {
        guard points.asArray.allSatisfy({ point in
            point.x.isFinite && point.y.isFinite && point.x >= 5 && point.y >= 5
                && point.x <= width - 5 && point.y <= height - 5
        }) else { return false }
        let eyeDistance = hypot(points.rightEye.x - points.leftEye.x,
                                points.rightEye.y - points.leftEye.y)
        let mouthDistance = hypot(points.rightMouth.x - points.leftMouth.x,
                                  points.rightMouth.y - points.leftMouth.y)
        let eyeY = (points.leftEye.y + points.rightEye.y) / 2
        let mouthY = (points.leftMouth.y + points.rightMouth.y) / 2
        return eyeDistance >= 25 && mouthDistance >= 15
            && points.leftEye.x < points.rightEye.x
            && points.leftMouth.x < points.rightMouth.x
            && eyeY > points.nose.y && points.nose.y > mouthY
    }
}

/// Conservative, provisional image-quality gates, measured before normalization.
/// These filter unsuitable input; they do not establish a recognition accuracy.
enum ImageQuality {
    struct Result {
        let accepted: Bool
        let reason: String
        let mean: Double
        let contrast: Double
        let laplacianVariance: Double
    }

    static func evaluate(rgba: [UInt8], width: Int, height: Int) -> Result {
        guard width >= 3, height >= 3, width <= 4096, height <= 4096,
              rgba.count == width * height * 4 else {
            return Result(accepted: false, reason: "Invalid aligned image", mean: 0,
                          contrast: 0, laplacianVariance: 0)
        }
        let count = width * height
        var luma = [Double](repeating: 0, count: count)
        var sum = 0.0, sumSquares = 0.0
        var dark = 0, clipped = 0, transparent = 0, black = 0
        for p in 0..<count {
            let i = p * 4
            let value = 0.299 * Double(rgba[i]) + 0.587 * Double(rgba[i + 1])
                + 0.114 * Double(rgba[i + 2])
            luma[p] = value
            sum += value
            sumSquares += value * value
            if value < 30 { dark += 1 }
            if value > 245 { clipped += 1 }
            if rgba[i + 3] < 250 { transparent += 1 }
            if rgba[i] <= 2 && rgba[i + 1] <= 2 && rgba[i + 2] <= 2 { black += 1 }
        }
        let total = Double(count)
        let mean = sum / total
        let contrast = max(0, sumSquares / total - mean * mean).squareRoot()

        var lapSum = 0.0, lapSumSquares = 0.0
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let p = y * width + x
                let lap = luma[p - 1] + luma[p + 1] + luma[p - width]
                    + luma[p + width] - 4 * luma[p]
                lapSum += lap
                lapSumSquares += lap * lap
            }
        }
        let lapCount = Double((width - 2) * (height - 2))
        let lapMean = lapSum / lapCount
        let lapVariance = max(0, lapSumSquares / lapCount - lapMean * lapMean)

        func result(_ accepted: Bool, _ reason: String) -> Result {
            Result(accepted: accepted, reason: reason, mean: mean, contrast: contrast,
                   laplacianVariance: lapVariance)
        }
        if Double(transparent) / total > 0.005 || Double(black) / total > 0.02 {
            return result(false, "Face crop is incomplete or black")
        }
        if mean < 45 || Double(dark) / total > 0.50 {
            return result(false, "Low light — uncertain")
        }
        if mean > 210 || Double(clipped) / total > 0.25 {
            return result(false, "Lighting is too bright — uncertain")
        }
        if contrast < 14 { return result(false, "Face contrast is too low") }
        if lapVariance < 35 { return result(false, "Face is blurred — uncertain") }
        return result(true, "Image quality accepted")
    }
}
