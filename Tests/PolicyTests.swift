import Foundation
import CoreGraphics

@main
enum PolicyTests {
    nonisolated static func vector(_ score: Float = 1) -> [Float] {
        var result = [Float](repeating: 0, count: 128)
        result[0] = score
        result[1] = sqrt(max(0, 1 - score * score))
        return result
    }

    static func main() {
        var checks = 0
        var failures: [String] = []
        func expect(_ condition: @autoclosure () -> Bool, _ name: String) {
            checks += 1
            if !condition() { failures.append(name) }
        }
        let bounds = CGRect(x: 0.2, y: 0.2, width: 0.4, height: 0.4)
        let otherBounds = CGRect(x: 0.75, y: 0.2, width: 0.2, height: 0.4)
        func frame(_ time: Double, _ classification: FaceClassification = .unknown,
                   box: CGRect? = bounds, face: [Float]? = vector()) -> FrameEvidence {
            FrameEvidence(classification: classification, timestamp: time,
                          faceBounds: box, embedding: face)
        }
        func feed(_ policy: inout GuardPolicy, from start: Double = 100,
                  through end: Double = 2, step: Double = 0.2) -> [GuardDecision] {
            (0...Int((end / step).rounded())).map { index in
                let time = start + Double(index) * step
                return policy.consume(frame(time), now: time)
            }
        }

        let classifier = FaceClassifier()
        let enrollment = Array(repeating: vector(), count: 15)
        expect(classifier.classify(embedding: vector(), enrollment: enrollment) == .jo,
               "matching Jo template")
        expect(classifier.classify(embedding: vector(0.50), enrollment: enrollment) == .jo,
               "Jo threshold is inclusive")
        expect(classifier.classify(embedding: vector(0.49), enrollment: enrollment) == .uncertain,
               "below Jo threshold is uncertain")
        expect(classifier.classify(embedding: vector(0.20), enrollment: enrollment) == .unknown,
               "unknown threshold is inclusive")
        expect(classifier.classify(embedding: vector(0.21), enrollment: enrollment) == .uncertain,
               "above unknown threshold is uncertain")
        expect(classifier.classify(embedding: vector(-1), enrollment: enrollment) == .unknown,
               "opposite vector is unknown")
        expect(classifier.classify(embedding: vector(0), enrollment: enrollment + [vector(0)]) == .jo,
               "best of all enrollment templates wins")
        expect(classifier.classify(embedding: vector(), enrollment: Array(enrollment.prefix(14))) == .uncertain,
               "too few enrollment samples is uncertain")
        expect(classifier.classify(embedding: vector(), enrollment: []) == .uncertain,
               "empty enrollment is uncertain")
        expect(classifier.classify(embedding: nil, enrollment: enrollment) == .uncertain,
               "missing embedding is uncertain")
        var nanVector = vector()
        nanVector[0] = .nan
        var infinityVector = vector()
        infinityVector[0] = .infinity
        let invalidVectors = [nanVector, infinityVector, [Float](repeating: 0, count: 128),
                              [1, 0], vector().map { $0 * 0.5 }]
        for (index, invalid) in invalidVectors.enumerated() {
            expect(classifier.classify(embedding: invalid, enrollment: enrollment) == .uncertain,
                   "invalid probe \(index) is uncertain")
            expect(classifier.classify(embedding: vector(), enrollment: enrollment + [invalid]) == .uncertain,
                   "invalid enrollment \(index) is uncertain")
        }
        expect(FaceClassifier(joThreshold: 0.1, unknownThreshold: 0.2)
            .classify(embedding: vector(), enrollment: enrollment) == .uncertain,
               "inverted classifier thresholds disable classification")
        expect(FaceClassifier(joThreshold: .nan)
            .classify(embedding: vector(), enrollment: enrollment) == .uncertain,
               "NaN classifier threshold disables classification")
        expect(FaceClassifier(expectedEmbeddingDimension: 2, minimumEnrollmentSamples: 1)
            .classify(embedding: [1, 0], enrollment: [[1, 0]]) == .jo,
               "explicit alternate dimensions are supported")

        expect(classifier.evaluate(embedding: vector(), enrollment: []) ==
            FaceMatchResult(classification: .uncertain, similarity: nil, issue: .missingProfile),
               "diagnostics identify a missing profile without presenting a similarity")
        expect(classifier.evaluate(embedding: vector(), enrollment: Array(enrollment.prefix(14))) ==
            FaceMatchResult(classification: .uncertain, similarity: nil, issue: .invalidProfile),
               "diagnostics identify an incomplete profile")
        expect(classifier.evaluate(embedding: nil, enrollment: enrollment) ==
            FaceMatchResult(classification: .uncertain, similarity: nil, issue: .invalidEmbedding),
               "failed inference remains distinct from a missing profile")
        for (index, invalid) in invalidVectors.enumerated() {
            expect(classifier.evaluate(embedding: invalid, enrollment: enrollment) ==
                FaceMatchResult(classification: .uncertain, similarity: nil, issue: .invalidEmbedding),
                   "diagnostics identify invalid probe \(index) without a score")
            expect(classifier.evaluate(embedding: vector(), enrollment: enrollment + [invalid]) ==
                FaceMatchResult(classification: .uncertain, similarity: nil, issue: .invalidProfile),
                   "diagnostics identify invalid profile sample \(index) without a score")
        }
        for invalidClassifier in [FaceClassifier(joThreshold: 0.1, unknownThreshold: 0.2),
                                  FaceClassifier(joThreshold: .nan),
                                  FaceClassifier(expectedEmbeddingDimension: 0),
                                  FaceClassifier(minimumEnrollmentSamples: 0)] {
            expect(invalidClassifier.evaluate(embedding: vector(), enrollment: enrollment) ==
                FaceMatchResult(classification: .uncertain, similarity: nil, issue: .invalidConfiguration),
                   "diagnostics identify invalid configuration without classification")
        }
        let ambiguous = classifier.evaluate(embedding: vector(0.30), enrollment: enrollment)
        expect(ambiguous.classification == .uncertain && ambiguous.issue == .ambiguous,
               "a valid intermediate score is identified as ambiguity")
        expect(ambiguous.similarity.map { abs($0 - 0.30) < 0.0001 } == true,
               "ambiguity diagnostics retain the actual cosine similarity")
        let matching = classifier.evaluate(embedding: vector(), enrollment: enrollment)
        expect(matching == FaceMatchResult(classification: .jo, similarity: 1, issue: .none),
               "Jo diagnostics include the actual similarity and no error")
        let unknown = classifier.evaluate(embedding: vector(0), enrollment: enrollment)
        expect(unknown == FaceMatchResult(classification: .unknown, similarity: 0, issue: .none),
               "unknown diagnostics include the actual similarity and no error")
        for score: Float in [-1, 0, 0.20, 0.21, 0.30, 0.49, 0.50, 1] {
            expect(classifier.classify(embedding: vector(score), enrollment: enrollment) ==
                classifier.evaluate(embedding: vector(score), enrollment: enrollment).classification,
                   "classification wrapper matches evaluation at similarity \(score)")
        }

        var policy = GuardPolicy()
        let decisions = feed(&policy)
        expect(decisions.dropLast().allSatisfy { !$0.shouldLock }, "no lock before two seconds")
        expect(decisions.last?.shouldLock == true, "locks at two seconds with sufficient samples")
        expect(decisions.last?.unknownDuration == 2, "reports elapsed capture time")
        expect(!policy.consume(frame(102.2), now: 102.2).shouldLock, "continuous run triggers only once")
        policy.reset()
        expect(feed(&policy).last?.shouldLock == true, "explicit reset permits a new run")

        policy = GuardPolicy()
        for time in stride(from: 100.0, through: 102.0, by: 0.5) {
            expect(!policy.consume(frame(time), now: time).shouldLock,
                   "elapsed time without enough samples does not lock at \(time)")
        }
        expect(!policy.consume(frame(102.1), now: 102.1).shouldLock, "six samples do not lock")
        expect(!policy.consume(frame(102.2), now: 102.2).shouldLock, "seven samples do not lock")
        expect(policy.consume(frame(102.3), now: 102.3).shouldLock, "eight fresh samples can lock")

        for classification in [FaceClassification.jo, .noFace, .uncertain] {
            policy = GuardPolicy()
            _ = feed(&policy, through: 1.8)
            let decision = policy.consume(frame(102, classification), now: 102)
            expect(decision.classification == classification && !decision.shouldLock && decision.unknownDuration == 0,
                   "\(classification.rawValue) resets the sequence")
            let next = policy.consume(frame(102.2), now: 102.2)
            expect(next.unknownDuration == 0 && !next.shouldLock,
                   "unknown after \(classification.rawValue) starts a new sequence")
        }

        policy = GuardPolicy()
        _ = feed(&policy, through: 1.8)
        let gap = policy.consume(frame(102.31), now: 102.31)
        expect(gap.unknownDuration == 0 && !gap.shouldLock, "gap greater than 0.5 seconds resets")
        policy = GuardPolicy()
        _ = feed(&policy, through: 1.8)
        expect(policy.consume(frame(102.3), now: 102.3).shouldLock,
               "gap exactly 0.5 seconds preserves continuity")

        for (name, time, now) in [
            ("stale", 102.0, 102.51), ("future", 102.0, 101.99),
            ("backward", 101.7, 101.7), ("duplicate", 101.8, 101.8),
            ("NaN capture", Double.nan, 102), ("infinite capture", Double.infinity, 102),
            ("NaN current time", 102, Double.nan), ("infinite current time", 102, Double.infinity),
            ("negative capture", -1, 102)
        ] {
            policy = GuardPolicy()
            _ = feed(&policy, through: 1.8)
            let result = policy.consume(frame(time), now: now)
            expect(result.classification == .uncertain && !result.shouldLock && result.unknownDuration == 0,
                   "\(name) timing cannot trigger lock")
            expect(policy.consume(frame(103), now: 103).unknownDuration == 0,
                   "\(name) timing clears sequence")
        }
        policy = GuardPolicy()
        _ = feed(&policy, through: 1.8)
        expect(policy.consume(frame(102), now: 102.5).shouldLock,
               "frame age exactly 0.5 seconds is accepted")

        let invalidBounds: [CGRect?] = [nil, .zero, .null,
            CGRect(x: -0.1, y: 0.2, width: 0.4, height: 0.4),
            CGRect(x: 0.9, y: 0.2, width: 0.4, height: 0.4),
            CGRect(x: CGFloat.nan, y: 0.2, width: 0.4, height: 0.4)]
        for (index, box) in invalidBounds.enumerated() {
            policy = GuardPolicy()
            _ = feed(&policy, through: 1.8)
            let result = policy.consume(frame(102, box: box), now: 102)
            expect(result.classification == .uncertain && !result.shouldLock,
                   "invalid tracking bounds \(index) cannot lock")
        }
        for (index, embedding) in ([nil] + invalidVectors.map(Optional.some)).enumerated() {
            policy = GuardPolicy()
            _ = feed(&policy, through: 1.8)
            let result = policy.consume(frame(102, face: embedding), now: 102)
            expect(result.classification == .uncertain && !result.shouldLock,
                   "invalid tracking embedding \(index) cannot lock")
        }

        policy = GuardPolicy()
        _ = feed(&policy, through: 1.8)
        expect(policy.consume(frame(102, box: otherBounds), now: 102).unknownDuration == 0,
               "discontinuous location starts a new sequence")
        policy = GuardPolicy()
        _ = feed(&policy, through: 1.8)
        expect(policy.consume(frame(102, face: vector(0.64)), now: 102).unknownDuration == 0,
               "different identity starts a new sequence")
        policy = GuardPolicy()
        _ = feed(&policy, through: 1.8)
        expect(policy.consume(frame(102, face: vector(0.65)), now: 102).shouldLock,
               "tracking similarity threshold is inclusive")
        policy = GuardPolicy()
        _ = feed(&policy, through: 1.8)
        expect(policy.consume(frame(102, box: bounds.offsetBy(dx: 0.01, dy: 0.01)), now: 102).shouldLock,
               "small normal face motion preserves continuity")
        policy = GuardPolicy()
        _ = policy.consume(frame(100), now: 100)
        _ = policy.consume(frame(100.2, face: vector(0.7)), now: 100.2)
        expect(policy.consume(frame(100.4, face: vector(0.2)), now: 100.4).unknownDuration == 0,
               "gradual identity drift is checked against sequence anchor")

        for invalidPolicy in [GuardPolicy(duration: .nan), GuardPolicy(duration: 0),
                              GuardPolicy(maxGap: 0), GuardPolicy(minimumSamples: 1),
                              GuardPolicy(minimumIoU: 0), GuardPolicy(minimumTrackingCosine: .nan)] {
            var disabled = invalidPolicy
            expect(disabled.consume(frame(100), now: 100).classification == .uncertain,
                   "invalid policy configuration disables locking")
        }

        if failures.isEmpty {
            print("PASS: \(checks) deterministic classifier and continuity checks")
        } else {
            for failure in failures { print("FAIL: \(failure)") }
            print("\(failures.count) failures across \(checks) checks")
            exit(1)
        }
    }
}
