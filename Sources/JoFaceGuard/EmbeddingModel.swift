import CoreML
import Foundation

/// SFace uses the same alignment geometry as ArcFace, but RAW RGB input and 128 outputs.
/// Keep this contract separate from the upstream InsightFace normalization.
final class EmbeddingModel {
    static let dimension = 128
    static let contract = "opencv-sface-2021dec-rgb-raw-112-v1"
    private let model: MLModel
    private let input: MLMultiArray

    init(url: URL? = nil) throws {
        let source = url ?? ProcessInfo.processInfo.environment["JO_FACE_GUARD_MODEL"].map { URL(fileURLWithPath: $0) }
            ?? Bundle.main.url(forResource: "SFace", withExtension: "mlpackage", subdirectory: "Models")
        guard let source else { throw GuardError.message("缺少人脸模型，请先运行 make model 并重新构建。") }
        // Core ML owns the temporary compiled location; never load an unrelated stale cache.
        let compiled = source.pathExtension == "mlmodelc" ? source : try MLModel.compileModel(at: source)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        model = try MLModel(contentsOf: compiled, configuration: config)
        guard let constraint = model.modelDescription.inputDescriptionsByName["input"]?.multiArrayConstraint,
              constraint.shape.map(\.intValue) == [1, 3, 112, 112],
              model.modelDescription.outputDescriptionsByName["embedding"] != nil else {
            throw GuardError.message("模型输入或输出不兼容。守卫保持暂停。")
        }
        input = try MLMultiArray(shape: [1, 3, 112, 112], dataType: .float32)
    }

    func embed(_ pixels: [UInt8]) throws -> [Float] {
        guard pixels.count == 112 * 112 * 4 else { throw GuardError.message("人脸裁剪尺寸错误") }
        let p = input.dataPointer.bindMemory(to: Float.self, capacity: 3 * 112 * 112)
        for index in 0..<(112 * 112) {
            for channel in 0..<3 { p[channel * 112 * 112 + index] = Float(pixels[index * 4 + channel]) }
        }
        let features = try MLDictionaryFeatureProvider(dictionary: ["input": input])
        guard let output = try model.prediction(from: features).featureValue(for: "embedding")?.multiArrayValue,
              output.count == Self.dimension else { throw GuardError.message("模型没有返回有效人脸特征") }
        // NSNumber indexing respects actual data type and strides (no unsafe Float assumption).
        let values = (0..<output.count).map { output[$0].floatValue }
        let norm = sqrt(values.reduce(Float(0)) { $0 + $1 * $1 })
        guard values.allSatisfy(\.isFinite), norm.isFinite, norm > 0.0001 else {
            throw GuardError.message("人脸特征无效")
        }
        return values.map { $0 / norm }
    }
}

enum GuardError: LocalizedError {
    case message(String)
    var errorDescription: String? { if case .message(let message) = self { return message }; return nil }
}
