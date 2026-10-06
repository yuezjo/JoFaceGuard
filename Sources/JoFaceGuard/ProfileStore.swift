import Foundation
import Security

struct FaceProfile: Codable {
    var version = 1
    var modelContract = EmbeddingModel.contract
    var samples: [[Float]]
    var createdAt = Date()

    var valid: Bool {
        version == 1 && modelContract == EmbeddingModel.contract && samples.count == 15 && samples.allSatisfy {
            $0.count == EmbeddingModel.dimension && $0.allSatisfy(\.isFinite)
                && abs(sqrt($0.reduce(0) { $0 + $1 * $1 }) - 1) < 0.01
        }
    }
}

/// Only embeddings are stored, in a non-synchronizing local Keychain item. No image files.
enum ProfileStore {
    private static var query: [String: Any] { [kSecClass as String: kSecClassGenericPassword,
        kSecAttrService as String: "io.github.yuezjo.JoFaceGuard",
        kSecAttrAccount as String: "jo-profile-v1", kSecAttrSynchronizable as String: false] }

    static func load() throws -> FaceProfile? {
        var q = query
        q[kSecReturnData as String] = true
        q[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        let status = SecItemCopyMatching(q as CFDictionary, &result)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess, let data = result as? Data else { throw error(status) }
        let profile = try JSONDecoder().decode(FaceProfile.self, from: data)
        guard profile.valid else { throw GuardError.message("已保存的人脸资料无效或模型已更换，请重新录入。") }
        return profile
    }

    static func save(_ profile: FaceProfile) throws {
        guard profile.valid else { throw GuardError.message("录入尚未完成") }
        let data = try JSONEncoder().encode(profile)
        let status = SecItemUpdate(query as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if status == errSecItemNotFound {
            var q = query
            q[kSecValueData as String] = data
            q[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
            let added = SecItemAdd(q as CFDictionary, nil)
            guard added == errSecSuccess else { throw error(added) }
        } else if status != errSecSuccess { throw error(status) }
    }

    static func delete() throws {
        let status = SecItemDelete(query as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else { throw error(status) }
    }

    private static func error(_ status: OSStatus) -> GuardError {
        .message("无法访问本机钥匙串（\(status)），守卫保持暂停。")
    }
}
