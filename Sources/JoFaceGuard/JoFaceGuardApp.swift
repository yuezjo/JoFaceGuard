import AppKit
import Combine
import SwiftUI

@main
enum JoFaceGuardApp {
    @MainActor static func main() {
        if CommandLine.arguments.contains("--smoke-test") {
            do {
                let model = try EmbeddingModel()
                var pixels = [UInt8](repeating: 0, count: 112 * 112 * 4)
                for i in pixels.indices { pixels[i] = UInt8((i * 37 + 29) % 256) }
                let first = try model.embed(pixels), second = try model.embed(pixels)
                let similarity = VectorMath.cosineSimilarity(first, second)
                guard first.count == 128, similarity > 0.999 else { throw GuardError.message("Model repeatability failed") }
                print("PASS: bundled Core ML SFace loads; 128 finite normalized values; repeat cosine \(similarity)")
                print("Lock API available: \(ScreenLocker().available); no lock requested; no camera opened; no profile accessed")
                return
            } catch { fputs("FAIL: \(error.localizedDescription)\n", stderr); exit(1) }
        }
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        if !CommandLine.arguments.contains("--ui-smoke-test"),
           let existing = NSRunningApplication.runningApplications(withBundleIdentifier: "io.github.yuezjo.JoFaceGuard")
            .first(where: { $0.processIdentifier != ProcessInfo.processInfo.processIdentifier }) {
            existing.activate(options: [.activateAllWindows])
            return
        }
        let delegate = AppDelegate()
        app.delegate = delegate
        withExtendedLifetime(delegate) { app.run() }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate {
    private var controller: GuardController!
    private var item: NSStatusItem!
    private var window: NSWindow!
    private var statusMenuItem: NSMenuItem!
    private var pauseMenuItem: NSMenuItem!
    private var cancellables: Set<AnyCancellable> = []

    func applicationDidFinishLaunching(_ notification: Notification) {
        controller = GuardController()
        item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let menuIcon = NSImage(named: "MenuIcon") {
            menuIcon.size = NSSize(width: 22, height: 22)
            menuIcon.isTemplate = false // Preserve the supplied pink cheeks and white face.
            item.button?.image = menuIcon
        } else {
            item.button?.image = NSImage(systemSymbolName: "person.crop.circle.badge.checkmark", accessibilityDescription: "JoFaceGuard")
        }
        item.button?.title = " Jo"
        let menu = NSMenu()
        statusMenuItem = NSMenuItem(title: "已暂停", action: nil, keyEquivalent: "")
        menu.addItem(statusMenuItem)
        menu.addItem(.separator())
        menu.addItem(withTitle: "打开 JoFaceGuard…", action: #selector(showWindow), keyEquivalent: "o").target = self
        pauseMenuItem = menu.addItem(withTitle: "暂停守卫与摄像头", action: #selector(pause), keyEquivalent: "p")
        pauseMenuItem.target = self
        menu.addItem(.separator())
        menu.addItem(withTitle: "退出", action: #selector(quit), keyEquivalent: "q").target = self
        item.menu = menu
        controller.$status.sink { [weak self] value in
            self?.statusMenuItem.title = value
            self?.item.button?.toolTip = "JoFaceGuard · \(value)"
            self?.item.button?.title = value.contains("陌生") ? " ?" : " Jo"
        }.store(in: &cancellables)
        controller.$monitoring.sink { [weak self] value in self?.pauseMenuItem.isEnabled = value }.store(in: &cancellables)
        menu.autoenablesItems = false
        statusMenuItem.isEnabled = false
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 650, height: 700),
                          styleMask: [.titled, .closable, .miniaturizable], backing: .buffered, defer: false)
        window.title = "JoFaceGuard"
        window.contentView = NSHostingView(rootView: GuardView(controller: controller))
        window.isReleasedWhenClosed = false
        window.delegate = self
        window.center()
        showWindow()
        if let option = CommandLine.arguments.firstIndex(of: "--ui-smoke-test"),
           CommandLine.arguments.indices.contains(option + 1) {
            let path = CommandLine.arguments[option + 1]
            DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [self] in
                guard let view = window.contentView,
                      let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { exit(1) }
                view.cacheDisplay(in: view.bounds, to: bitmap)
                do {
                    guard let data = bitmap.representation(using: .png, properties: [:]) else { exit(1) }
                    try data.write(to: URL(fileURLWithPath: path))
                    guard controller.ready, !controller.monitoring, !controller.armed else { exit(1) }
                    print("PASS: native window renders; model ready; paused; camera stopped; auto-lock off")
                    exit(0)
                } catch { fputs("UI smoke failed: \(error)\n", stderr); exit(1) }
            }
        }
    }

    @objc private func showWindow() {
        controller.setPreviewVisible(true)
        window.makeKeyAndOrderFront(nil)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }
    @objc private func pause() { controller.pause() }
    @objc private func quit() { controller.pause(); NSApplication.shared.terminate(nil) }
    func windowWillClose(_ notification: Notification) {
        controller.setPreviewVisible(false)
        if controller.enrolling { controller.pause() }
    }
    func applicationWillTerminate(_ notification: Notification) { controller.pause() }
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        showWindow()
        return true
    }
}

struct GuardView: View {
    @ObservedObject var controller: GuardController

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .center) {
                if let brand = NSImage(named: "BrandIcon") {
                    Image(nsImage: brand).resizable().scaledToFit().frame(width: 44, height: 44)
                        .accessibilityLabel("JoFaceGuard 小骷髅图标")
                } else {
                    Image(systemName: "person.crop.circle.badge.checkmark")
                        .font(.system(size: 34)).foregroundStyle(.purple)
                }
                VStack(alignment: .leading, spacing: 3) {
                    Text("JoFaceGuard · \(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "dev")").font(.title2.bold())
                    Text("只有明确的陌生人，才会启动两秒确认。")
                        .font(.callout).foregroundStyle(.secondary)
                }
                Spacer()
                Text(controller.armed ? "守卫中" : "未启用锁屏")
                    .font(.caption.weight(.medium)).padding(8)
                    .background(controller.armed ? Color.green.opacity(0.12) : Color.secondary.opacity(0.1))
                    .clipShape(Capsule())
            }
            ZStack {
                RoundedRectangle(cornerRadius: 14).fill(Color.black.opacity(0.92))
                if let preview = controller.preview {
                    Image(decorative: preview, scale: 1).resizable().scaledToFit()
                        .scaleEffect(x: -1, y: 1) // Mirror only the preview, never the ML input.
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                } else {
                    VStack(spacing: 10) {
                        Image(systemName: "camera").font(.system(size: 32))
                        Text(controller.monitoring ? "等待摄像头画面…" : "摄像头已停止")
                    }.foregroundStyle(.white.opacity(0.7))
                }
            }.frame(height: 265)
            VStack(alignment: .leading, spacing: 7) {
                Text(controller.status).font(.headline)
                Text(controller.detail).foregroundStyle(.secondary).font(.callout).frame(minHeight: 36, alignment: .topLeading)
                if controller.classification == .unknown {
                    ProgressView(value: min(controller.elapsed / 2, 1)).tint(.orange)
                }
                if let error = controller.error {
                    Text(error).font(.callout).foregroundStyle(.red).textSelection(.enabled)
                }
            }
            Divider()
            if controller.enrolling {
                Text("姿态 \(controller.enrollmentStep + 1) / 5：\(controller.enrollmentPrompt)").font(.headline)
                Text("已采集 \(controller.enrollmentCount) / 15 张特征样本。只保留特征，原始画面不会保存。")
                    .font(.caption).foregroundStyle(.secondary)
                HStack {
                    Button(controller.capturing ? "等待清晰画面…" : "采集当前姿态（3 张）") { controller.capturePose() }
                        .buttonStyle(.borderedProminent).disabled(controller.capturing)
                    Button("取消录入") { controller.pause() }
                }
            } else {
                HStack {
                    Button(controller.profile == nil ? "录入 Jo" : "重新录入 Jo") { controller.beginEnrollment() }
                        .disabled(!controller.ready || controller.profileBusy)
                    Button("观察识别") { controller.observe() }.disabled(!controller.ready || controller.monitoring)
                    Button("开启自动锁屏") { controller.arm() }.buttonStyle(.borderedProminent).disabled(!controller.canArm)
                    Button("暂停") { controller.pause() }.disabled(!controller.monitoring)
                }
                Text(controller.profile == nil ? "在正常光线下录入 5 个轻微不同的姿态。常戴眼镜时请戴着录入。" : "观察模式连续认出 Jo 后才能开启。先检查本人、无人和暗光，再测试陌生人。")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
            HStack {
                Text("本机处理 · 无网络请求 · 不保存照片").font(.caption).foregroundStyle(.secondary)
                Spacer()
                if controller.profile != nil {
                    Button("删除人脸资料") {
                        let alert = NSAlert()
                        alert.messageText = "删除 Jo 的人脸资料？"
                        alert.informativeText = "会暂停守卫并删除本机钥匙串中的特征。之后需要重新录入。"
                        alert.addButton(withTitle: "取消")
                        alert.addButton(withTitle: "删除")
                        if alert.runModal() == .alertSecondButtonReturn { controller.deleteProfile() }
                    }.font(.caption).disabled(controller.profileBusy)
                }
            }
        }
        .padding(24).frame(width: 650, height: 700)
        .background(Color(nsColor: .windowBackgroundColor))
        .tint(.purple)
    }
}
