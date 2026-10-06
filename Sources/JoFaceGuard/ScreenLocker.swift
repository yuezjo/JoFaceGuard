import AppKit
import Darwin

/// Runtime binding: this is a PRIVATE macOS API, not suitable for the Mac App Store.
/// A void call is a request, never proof of a successfully locked session.
final class ScreenLocker {
    private let handle: UnsafeMutableRawPointer?
    private let function: (@convention(c) () -> Void)?
    var available: Bool { function != nil }

    init() {
        handle = dlopen("/System/Library/PrivateFrameworks/login.framework/login", RTLD_LAZY)
        if let handle, let symbol = dlsym(handle, "SACLockScreenImmediate") {
            function = unsafeBitCast(symbol, to: (@convention(c) () -> Void).self)
        } else { function = nil }
    }
    deinit { if let handle { dlclose(handle) } }

    func requestLock() throws {
        guard let function else { throw GuardError.message("当前 macOS 不支持锁屏接口，无法启用自动锁屏。") }
        function()
    }

    static var sessionLocked: Bool {
        guard let values = CGSessionCopyCurrentDictionary() as? [String: Any] else { return true }
        return (values["CGSSessionScreenIsLocked"] as? NSNumber)?.boolValue ?? false
    }
}
