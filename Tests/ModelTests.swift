import Foundation

/// Integration test: real Swift RGBA→RGB NCHW packing and Core ML inference
/// must agree with independent ONNX Runtime results, using no face images.
@main
struct ModelTests {
    struct Fixture: Decodable {
        let source_sha256: String
        let dimension: Int
        let minimum_cosine: Float
        let maximum_negative_cosine: Float
        let cases: [Reference]
    }
    struct Reference: Decodable {
        let seed: Int
        let embedding: [Float]
    }

    static func main() throws {
        guard let path = ProcessInfo.processInfo.environment["JO_FACE_GUARD_MODEL"] else {
            throw GuardError.message("Set JO_FACE_GUARD_MODEL to Resources/Models/SFace.mlpackage")
        }
        let fixturePath = CommandLine.arguments.count > 1
            ? CommandLine.arguments[1] : "Tests/Fixtures/sface-reference.json"
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: URL(fileURLWithPath: fixturePath)))
        guard fixture.dimension == EmbeddingModel.dimension,
              fixture.source_sha256 == "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
              fixture.cases.count == 3 else { throw GuardError.message("Unexpected SFace fixture contract") }
        let model = try EmbeddingModel(url: URL(fileURLWithPath: path))
        for reference in fixture.cases {
            let pixels = syntheticRGBA(seed: reference.seed)
            let actual = try model.embed(pixels)
            let agreement = try cosine(actual, reference.embedding)
            try require(agreement >= fixture.minimum_cosine,
                        "seed \(reference.seed): Swift/ONNX cosine \(agreement), expected ≥ \(fixture.minimum_cosine)")
            var bgr = pixels
            var vertical = pixels
            var horizontal = pixels
            var opaque = pixels
            for y in 0..<112 {
                for x in 0..<112 {
                    let offset = (y * 112 + x) * 4
                    bgr[offset] = pixels[offset + 2]
                    bgr[offset + 2] = pixels[offset]
                    opaque[offset + 3] = 255
                    for channel in 0..<4 {
                        vertical[offset + channel] = pixels[((111 - y) * 112 + x) * 4 + channel]
                        horizontal[offset + channel] = pixels[(y * 112 + 111 - x) * 4 + channel]
                    }
                }
            }
            let bgrScore = try cosine(model.embed(bgr), reference.embedding)
            let verticalScore = try cosine(model.embed(vertical), reference.embedding)
            let horizontalScore = try cosine(model.embed(horizontal), reference.embedding)
            for (name, score) in [("RGB/BGR swap", bgrScore), ("vertical flip", verticalScore), ("horizontal flip", horizontalScore)] {
                try require(score < fixture.maximum_negative_cosine,
                            "seed \(reference.seed): \(name) negative control too similar (\(score)); fixture is not sensitive enough")
            }
            let alphaScore = try cosine(model.embed(opaque), actual)
            try require(alphaScore >= 0.999999, "Alpha unexpectedly changed RGB model input")
            print(String(format: "PASS SFace seed %d: Swift/ONNX %.9f; BGR %.6f; vertical %.6f; horizontal %.6f",
                         reference.seed, agreement, bgrScore, verticalScore, horizontalScore))
        }
        do {
            _ = try model.embed([0, 0, 0, 255])
            throw GuardError.message("Malformed RGBA length was accepted")
        } catch GuardError.message(let message) where message == "人脸裁剪尺寸错误" {
            print("PASS malformed image size rejected")
        }
        print("PASS model integration: 3 synthetic images; channel order, row order, alpha exclusion, unit embeddings, invalid-size guard")
    }

    static func syntheticRGBA(seed: Int) -> [UInt8] {
        var pixels = [UInt8](repeating: 0, count: 112 * 112 * 4)
        for y in 0..<112 {
            for x in 0..<112 {
                let offset = (y * 112 + x) * 4
                pixels[offset] = UInt8((3 * x + 5 * y + (x / 7) * (y / 11) * 17 + seed * 19) % 256)
                pixels[offset + 1] = UInt8((11 * x + 2 * y + ((x * y + seed * 13) % 97) * 3 + seed * 43) % 256)
                pixels[offset + 2] = UInt8((2 * x + 13 * y + ((x / 9 + y / 5) % 2) * 79 + seed * 71) % 256)
                pixels[offset + 3] = UInt8((7 * x + 9 * y + seed * 11) % 256)
            }
        }
        return pixels
    }

    static func cosine(_ lhs: [Float], _ rhs: [Float]) throws -> Float {
        try require(lhs.count == 128 && rhs.count == 128, "Embedding length must be 128")
        try require(lhs.allSatisfy(\.isFinite) && rhs.allSatisfy(\.isFinite), "Embedding must be finite")
        let leftNorm = sqrt(lhs.reduce(Float(0)) { $0 + $1 * $1 })
        let rightNorm = sqrt(rhs.reduce(Float(0)) { $0 + $1 * $1 })
        try require(abs(leftNorm - 1) < 0.0001 && abs(rightNorm - 1) < 0.0001, "Embeddings must be L2 normalized")
        return zip(lhs, rhs).reduce(Float(0)) { $0 + $1.0 * $1.1 } / (leftNorm * rightNorm)
    }

    static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw GuardError.message(message) }
    }
}
