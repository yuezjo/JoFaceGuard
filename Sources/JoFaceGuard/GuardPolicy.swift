import Foundation
import CoreGraphics

enum FaceClassification: String, Equatable, Sendable {
    case jo
    case unknown
    case uncertain
    case noFace
}

/// `timestamp` is the capture time on the same monotonic clock as `now`.
/// The camera pipeline must apply its image-quality gate before emitting `.unknown`.
struct FrameEvidence: Sendable {
    let classification: FaceClassification
    let timestamp: Double
    let faceBounds: CGRect?
    let embedding: [Float]?

    init(classification: FaceClassification, timestamp: Double,
         faceBounds: CGRect? = nil, embedding: [Float]? = nil) {
        self.classification = classification
        self.timestamp = timestamp
        self.faceBounds = faceBounds
        self.embedding = embedding
    }
}

struct GuardDecision: Equatable, Sendable {
    let classification: FaceClassification
    let unknownDuration: Double
    let shouldLock: Bool
}

/// A failure or score between the two thresholds is always inconclusive.
/// Scores are deliberately not presented as calibrated probabilities.
struct FaceClassifier: Sendable {
    let expectedEmbeddingDimension: Int
    let minimumEnrollmentSamples: Int
    let joThreshold: Float
    let unknownThreshold: Float

    init(expectedEmbeddingDimension: Int = 128, minimumEnrollmentSamples: Int = 15,
         joThreshold: Float = 0.50, unknownThreshold: Float = 0.20) {
        self.expectedEmbeddingDimension = expectedEmbeddingDimension
        self.minimumEnrollmentSamples = minimumEnrollmentSamples
        self.joThreshold = joThreshold
        self.unknownThreshold = unknownThreshold
    }

    func classify(embedding: [Float]?, enrollment: [[Float]]) -> FaceClassification {
        guard expectedEmbeddingDimension > 0,
              minimumEnrollmentSamples > 0,
              joThreshold.isFinite, unknownThreshold.isFinite,
              (-1...1).contains(unknownThreshold), (-1...1).contains(joThreshold),
              unknownThreshold < joThreshold,
              enrollment.count >= minimumEnrollmentSamples,
              let embedding,
              PolicyVector.isValid(embedding, dimension: expectedEmbeddingDimension),
              enrollment.allSatisfy({ PolicyVector.isValid($0, dimension: expectedEmbeddingDimension) })
        else { return .uncertain }

        let bestScore = enrollment.map { PolicyVector.cosine(embedding, $0) }.max()!
        if bestScore >= joThreshold { return .jo }
        if bestScore <= unknownThreshold { return .unknown }
        return .uncertain
    }
}

/// A lock requires fresh, ordered samples of the same clearly non-Jo face.
/// Any interruption clears the timer, and a continuous sequence locks only once.
struct GuardPolicy: Sendable {
    let duration: Double
    let maxGap: Double
    let minimumSamples: Int
    let maximumFrameAge: Double
    let minimumIoU: Double
    let minimumTrackingCosine: Float
    let expectedEmbeddingDimension: Int

    private var lastFrameTimestamp: Double?
    private var sequenceStart: Double?
    private var previousUnknownTimestamp: Double?
    private var previousBounds: CGRect?
    private var previousEmbedding: [Float]?
    private var firstEmbedding: [Float]?
    private var samples = 0
    private var didLock = false

    init(duration: Double = 2, maxGap: Double = 0.5, minimumSamples: Int = 8,
         maximumFrameAge: Double = 0.5, minimumIoU: Double = 0.25,
         minimumTrackingCosine: Float = 0.65, expectedEmbeddingDimension: Int = 128) {
        self.duration = duration
        self.maxGap = maxGap
        self.minimumSamples = minimumSamples
        self.maximumFrameAge = maximumFrameAge
        self.minimumIoU = minimumIoU
        self.minimumTrackingCosine = minimumTrackingCosine
        self.expectedEmbeddingDimension = expectedEmbeddingDimension
    }

    mutating func reset() {
        lastFrameTimestamp = nil
        resetSequence()
    }

    mutating func consume(_ evidence: FrameEvidence, now: Double) -> GuardDecision {
        guard configurationIsValid, now.isFinite, evidence.timestamp.isFinite,
              evidence.timestamp >= 0, now >= evidence.timestamp,
              now - evidence.timestamp <= maximumFrameAge,
              lastFrameTimestamp.map({ evidence.timestamp > $0 }) ?? true
        else {
            reset()
            return decision(.uncertain)
        }
        lastFrameTimestamp = evidence.timestamp

        guard evidence.classification == .unknown else {
            resetSequence()
            return decision(evidence.classification)
        }
        guard let bounds = evidence.faceBounds, Self.validBounds(bounds),
              let embedding = evidence.embedding,
              PolicyVector.isValid(embedding, dimension: expectedEmbeddingDimension)
        else {
            resetSequence()
            return decision(.uncertain)
        }

        if let previousTime = previousUnknownTimestamp,
           let previousBounds, let previousEmbedding, let firstEmbedding {
            let continuous = evidence.timestamp - previousTime <= maxGap
                && Self.intersectionOverUnion(bounds, previousBounds) >= minimumIoU
                && PolicyVector.cosine(embedding, previousEmbedding) >= minimumTrackingCosine
                && PolicyVector.cosine(embedding, firstEmbedding) >= minimumTrackingCosine
            if !continuous { resetSequence() }
        }

        if sequenceStart == nil {
            sequenceStart = evidence.timestamp
            firstEmbedding = embedding
        }
        previousUnknownTimestamp = evidence.timestamp
        previousBounds = bounds
        previousEmbedding = embedding
        samples += 1

        let elapsed = evidence.timestamp - sequenceStart!
        let shouldLock = !didLock && elapsed >= duration && samples >= minimumSamples
        if shouldLock { didLock = true }
        return GuardDecision(classification: .unknown, unknownDuration: elapsed, shouldLock: shouldLock)
    }

    private var configurationIsValid: Bool {
        duration.isFinite && duration > 0 && maxGap.isFinite && maxGap > 0
            && maximumFrameAge.isFinite && maximumFrameAge >= 0 && minimumSamples >= 2
            && minimumIoU.isFinite && minimumIoU > 0 && minimumIoU <= 1
            && minimumTrackingCosine.isFinite && minimumTrackingCosine > 0
            && minimumTrackingCosine <= 1 && expectedEmbeddingDimension > 0
    }

    private mutating func resetSequence() {
        sequenceStart = nil
        previousUnknownTimestamp = nil
        previousBounds = nil
        previousEmbedding = nil
        firstEmbedding = nil
        samples = 0
        didLock = false
    }

    private func decision(_ classification: FaceClassification) -> GuardDecision {
        GuardDecision(classification: classification, unknownDuration: 0, shouldLock: false)
    }

    private static func validBounds(_ rect: CGRect) -> Bool {
        rect.origin.x.isFinite && rect.origin.y.isFinite
            && rect.width.isFinite && rect.height.isFinite
            && rect.width > 0 && rect.height > 0
            && rect.minX >= 0 && rect.minY >= 0 && rect.maxX <= 1 && rect.maxY <= 1
    }

    private static func intersectionOverUnion(_ a: CGRect, _ b: CGRect) -> Double {
        let overlap = a.intersection(b)
        if overlap.isNull || overlap.isEmpty { return 0 }
        let area = overlap.width * overlap.height
        let union = a.width * a.height + b.width * b.height - area
        return Double(area / union)
    }
}

private enum PolicyVector {
    static func isValid(_ vector: [Float], dimension: Int) -> Bool {
        guard vector.count == dimension, vector.allSatisfy(\.isFinite) else { return false }
        let normSquared = vector.reduce(0.0) { $0 + Double($1) * Double($1) }
        return abs(normSquared.squareRoot() - 1) <= 0.01
    }

    // Validation happens before every call. Dividing by norms avoids tolerance
    // around unit normalization influencing threshold comparisons.
    static func cosine(_ a: [Float], _ b: [Float]) -> Float {
        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for index in a.indices {
            let x = Double(a[index])
            let y = Double(b[index])
            dot += x * y
            normA += x * x
            normB += y * y
        }
        return Float(max(-1, min(1, dot / (normA * normB).squareRoot())))
    }
}
