import AppKit
import AVFoundation
import CoreImage
import QuartzCore
import SwiftUI

private struct AnalyzedFrame {
    let analysis: FaceAnalysis
    let embedding: [Float]?
    let preview: CGImage?
    let timestamp: Double
    let generation: Int
}

@MainActor
final class GuardController: ObservableObject {
    @Published private(set) var status = "已暂停"
    @Published private(set) var detail = "所有人脸处理均在本机完成。先录入 Jo，再观察识别结果。"
    @Published private(set) var preview: CGImage?
    @Published private(set) var profile: FaceProfile?
    @Published private(set) var monitoring = false
    @Published private(set) var armed = false
    @Published private(set) var enrolling = false
    @Published private(set) var enrollmentCount = 0
    @Published private(set) var capturing = false
    @Published private(set) var classification: FaceClassification = .uncertain
    @Published private(set) var elapsed: Double = 0
    @Published private(set) var ready = false
    @Published private(set) var error: String?
    @Published private(set) var profileBusy = true

    private let camera = CameraCapture()
    private let locker = ScreenLocker()
    private var policy = GuardPolicy()
    private let classifier = FaceClassifier()
    private var generation = 0
    private var lastFrameAt = 0.0
    private var startedAt = 0.0
    private var joFrames = 0
    private var samples: [[Float]] = []
    private var remainingCaptures = 0
    private var lastCaptureAt = 0.0
    private var watchdog: Timer?
    private var tokens: [NSObjectProtocol] = []
    private var systemSuspended = false
    private var previewVisible = false
    private var storageGeneration = 0

    var canArm: Bool { ready && !profileBusy && profile != nil && monitoring && !enrolling && !armed && joFrames >= 5 && locker.available }
    var enrollmentStep: Int { min(enrollmentCount / 3, 4) }
    var enrollmentPrompt: String {
        ["正面看摄像头", "脸稍微转向一侧（约 10°）", "脸稍微转向另一侧（约 10°）", "稍微抬头（约 10°）", "稍微低头（约 10°）"][enrollmentStep]
    }

    init() {
        camera.onFailure = { [weak self] message, token in
            Task { @MainActor in
                guard let self, token == self.generation else { return }
                self.pause(); self.error = message; self.detail = message
            }
        }
        let center = DistributedNotificationCenter.default()
        tokens.append(center.addObserver(forName: .init("com.apple.screenIsLocked"), object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.suspendForSystem("电脑已锁定") }
        })
        tokens.append(center.addObserver(forName: .init("com.apple.screenIsUnlocked"), object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in
                self?.systemSuspended = false
                self?.status = "已暂停"
                self?.detail = "已解锁；点击观察识别后可重新开启守卫。"
            }
        })
        for name in [NSWorkspace.willSleepNotification, NSWorkspace.screensDidSleepNotification, NSWorkspace.sessionDidResignActiveNotification] {
            tokens.append(NSWorkspace.shared.notificationCenter.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                Task { @MainActor in self?.suspendForSystem("系统休眠，守卫已暂停") }
            })
        }
        for name in [NSWorkspace.didWakeNotification, NSWorkspace.screensDidWakeNotification, NSWorkspace.sessionDidBecomeActiveNotification] {
            tokens.append(NSWorkspace.shared.notificationCenter.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                Task { @MainActor in self?.systemSuspended = false }
            })
        }
        watchdog = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.checkFreshness() }
        }
        // Model loading and Keychain access may take time; keep the menu responsive.
        camera.queue.async { [weak self] in
            do {
                let model = try EmbeddingModel()
                let pipeline = FacePipeline()
                let renderer = CIContext()
                var profile: FaceProfile?
                var profileError: String?
                do { profile = try ProfileStore.load() }
                catch { profileError = error.localizedDescription }
                Task { @MainActor in
                    guard let self else { return }
                    self.profile = profile
                    self.profileBusy = false
                    self.error = profileError
                    self.camera.onFrame = { [weak self] buffer, time, token in
                        let analysis = pipeline.analyze(buffer)
                        let embedding = analysis.pixels.flatMap { try? model.embed($0) }
                        // Preview remains memory-only, and is never logged or persisted.
                        let image = CIImage(cvPixelBuffer: buffer)
                        let small = image.transformed(by: CGAffineTransform(scaleX: 0.5, y: 0.5))
                        let cg = renderer.createCGImage(small, from: small.extent)
                        let frame = AnalyzedFrame(analysis: analysis, embedding: embedding, preview: cg, timestamp: time, generation: token)
                        Task { @MainActor in self?.consume(frame) }
                    }
                    self.ready = true
                    self.detail = profile == nil ? "准备好了。请在明亮环境下录入 Jo。" : "已载入 Jo 的本机人脸资料，点击观察识别。"
                    if !self.locker.available { self.error = "此 macOS 的锁屏接口不可用；只能观察，无法启用自动锁屏。" }
                }
            } catch {
                Task { @MainActor in self?.profileBusy = false; self?.error = error.localizedDescription; self?.detail = error.localizedDescription }
            }
        }
    }

    func setPreviewVisible(_ visible: Bool) {
        previewVisible = visible
        if !visible { preview = nil }
    }

    func observe() { start(enrollment: false) }
    func beginEnrollment() { start(enrollment: true) }

    private func start(enrollment: Bool) {
        guard ready, !profileBusy, !ScreenLocker.sessionLocked, !systemSuspended else { return }
        pause()
        error = nil
        enrolling = enrollment
        samples = []
        enrollmentCount = 0
        generation += 1
        let token = generation
        status = "请求摄像头权限"
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: cameraAuthorized(token: token)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                Task { @MainActor in
                    guard let self, token == self.generation else { return }
                    if granted { self.cameraAuthorized(token: token) }
                    else { self.pause(); self.error = "需要摄像头权限，请在系统设置 → 隐私与安全性 → 摄像头中允许 JoFaceGuard。" }
                }
            }
        default:
            pause()
            error = "需要摄像头权限，请在系统设置 → 隐私与安全性 → 摄像头中允许 JoFaceGuard。"
        }
    }

    private func cameraAuthorized(token: Int) {
        guard token == generation, !ScreenLocker.sessionLocked, !systemSuspended else { return }
        monitoring = true
        startedAt = CACurrentMediaTime()
        lastFrameAt = startedAt
        status = enrolling ? "正在录入 Jo" : "观察模式 · 不会锁屏"
        detail = "等待清晰画面…"
        camera.start(generation: token)
    }

    func pause() {
        generation += 1
        camera.stop()
        monitoring = false; armed = false; enrolling = false; capturing = false
        remainingCaptures = 0; samples = []; joFrames = 0
        classification = .uncertain; elapsed = 0; policy.reset(); preview = nil
        status = "已暂停"
        detail = "摄像头已停止。点击观察识别，或重新录入。"
    }

    func arm() {
        guard canArm, CACurrentMediaTime() - lastFrameAt < 0.5, !ScreenLocker.sessionLocked else { return }
        armed = true
        policy.reset(); elapsed = 0
        status = "守卫已开启 · Jo"
        detail = "持续约 2 秒确认陌生人后锁屏；不确定时不会锁屏。"
    }

    func capturePose() {
        guard enrolling, monitoring, !capturing else { return }
        remainingCaptures = 3
        capturing = true
    }

    func deleteProfile() {
        pause()
        storageGeneration += 1
        let operation = storageGeneration
        profileBusy = true
        camera.queue.async { [weak self] in
            do {
                try ProfileStore.delete()
                Task { @MainActor in
                    guard let self, self.storageGeneration == operation else { return }
                    self.profile = nil; self.profileBusy = false; self.detail = "已删除 Jo 的本机人脸资料。"
                }
            } catch {
                Task { @MainActor in
                    guard let self, self.storageGeneration == operation else { return }
                    self.profileBusy = false; self.error = error.localizedDescription
                }
            }
        }
    }

    private func consume(_ frame: AnalyzedFrame) {
        guard monitoring, frame.generation == generation, !systemSuspended else { return }
        let now = CACurrentMediaTime()
        guard now - frame.timestamp <= 0.5, now >= frame.timestamp else {
            policy.reset(); elapsed = 0; joFrames = 0; classification = .uncertain
            detail = "处理延迟过高 · 不确定"; return
        }
        lastFrameAt = frame.timestamp
        if previewVisible { preview = frame.preview }
        let analysis = frame.analysis
        if enrolling {
            if analysis.qualityOK, frame.embedding != nil {
                detail = capturing ? "画面合格，正在采集（本步还需 \(remainingCaptures) 张）…"
                    : "画面合格。\(enrollmentPrompt)，点击「采集当前姿态」。"
            } else if analysis.qualityOK {
                detail = "人脸特征提取失败，请暂停后重新开始；无需继续调整头部。"
            } else { detail = Self.qualityGuidance(analysis.reason) }
            guard capturing, analysis.qualityOK, let vector = frame.embedding,
                  now - lastCaptureAt >= 0.45 else { return }
            if let first = samples.first, VectorMath.cosineSimilarity(first, vector) < 0.50 {
                detail = "与首张样本差异较大，请恢复正面、调整光线再采集。"
                return
            }
            lastCaptureAt = now
            samples.append(vector); enrollmentCount = samples.count; remainingCaptures -= 1
            if remainingCaptures == 0 { capturing = false }
            if samples.count == 15 {
                let result = FaceProfile(samples: samples)
                // Save only a completed set; cancelling never overwrites a previous profile.
                pause()
                storageGeneration += 1
                let operation = storageGeneration
                profileBusy = true
                detail = "正在保存 Jo 的本机特征…"
                camera.queue.async { [weak self] in
                    do {
                        try ProfileStore.save(result)
                        Task { @MainActor in
                            guard let self, self.storageGeneration == operation else { return }
                            self.profile = result; self.profileBusy = false
                            self.detail = "Jo 录入完成。先点击观察识别，检查正常光线、眼镜与姿势变化。"
                        }
                    } catch {
                        Task { @MainActor in
                            guard let self, self.storageGeneration == operation else { return }
                            self.profileBusy = false; self.error = error.localizedDescription
                        }
                    }
                }
            }
            return
        }
        let kind: FaceClassification
        if analysis.faceCount == 0 { kind = .noFace }
        else if !analysis.qualityOK { kind = .uncertain }
        else { kind = classifier.classify(embedding: frame.embedding, enrollment: profile?.samples ?? []) }
        let decision = policy.consume(FrameEvidence(classification: kind, timestamp: frame.timestamp,
            faceBounds: analysis.bounds, embedding: frame.embedding), now: now)
        classification = decision.classification
        joFrames = decision.classification == .jo ? joFrames + 1 : 0
        elapsed = decision.unknownDuration
        let label: String
        switch decision.classification {
        case .jo: label = "Jo"; detail = "已识别为 Jo"
        case .noFace: label = "无人"; detail = "没有检测到人脸，不会触发锁屏。"
        case .uncertain: label = "不确定"; detail = analysis.qualityOK ? "相似度不明确或资料不可用，不会触发锁屏。" : Self.qualityGuidance(analysis.reason)
        case .unknown: label = "陌生人"; detail = String(format: "连续确认 %.1f / 2.0 秒%@", elapsed, armed ? "" : "（观察模式，不会锁屏）")
        }
        status = (armed ? "守卫 · " : "观察 · ") + label
        if armed && decision.shouldLock { lockForUnknown() }
    }

    private func lockForUnknown() {
        guard armed, !ScreenLocker.sessionLocked else { pause(); return }
        pause()
        let lockGeneration = generation
        status = "正在请求锁屏"
        detail = "已连续确认陌生人，等待系统锁屏确认。"
        do {
            try locker.requestLock()
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
                guard let self, self.generation == lockGeneration else { return }
                if ScreenLocker.sessionLocked { self.status = "电脑已锁定" }
                else { self.status = "锁屏未确认 · 已暂停"; self.error = "系统没有确认锁屏，请使用 Control–Command–Q 手动锁屏。" }
            }
        } catch { self.error = error.localizedDescription; status = "锁屏失败 · 已暂停" }
    }

    private func suspendForSystem(_ message: String) {
        pause(); systemSuspended = true; status = message
    }

    private func checkFreshness() {
        guard monitoring else { return }
        let now = CACurrentMediaTime()
        if ScreenLocker.sessionLocked { suspendForSystem("电脑已锁定"); return }
        if now - lastFrameAt > 0.5 {
            policy.reset(); elapsed = 0; joFrames = 0; classification = .uncertain
            if now - startedAt > 1 { status = "不确定 · 等待摄像头"; detail = "画面中断，陌生人计时已清零。" }
        }
        if now - lastFrameAt > 8 { pause(); error = "摄像头长时间没有有效画面，请重新开始。" }
    }

    private static func qualityGuidance(_ reason: String) -> String {
        let text = reason.lowercased()
        if text.contains("low light") || text.contains("black") { return "光线不足或画面被遮挡 · 不确定" }
        if text.contains("bright") { return "光线太强，请避开直射光 · 不确定" }
        if text.contains("multiple") { return "画面中有多个人 · 不确定" }
        if text == "no face" { return "没有检测到人脸，请面向摄像头。" }
        if text.contains("small") { return "脸部太小，请稍微靠近摄像头。" }
        if text.contains("edge") || text.contains("incomplete") { return "请让整张脸进入画面。" }
        if text.contains("blur") || text.contains("contrast") { return "画面不够清晰，请调整光线并保持片刻。" }
        if text.contains("pose unavailable") { return "未取得头部姿态数据，请暂停后重试；无需继续转头。" }
        if text.contains("look toward") { return "侧转或抬低头幅度较大，请恢复自然正面。" }
        if text.contains("quality is low") { return "脸部采集质量不足，请正面保持片刻，调整光线或距离。" }
        if text.contains("quality") { return "未取得人脸画质评估，请暂停后重试。" }
        if text.contains("alignment") || text.contains("landmark") { return "无法定位眼睛、鼻子或嘴角，请保持无遮挡的正面。" }
        if text.contains("confidence") { return "暂时无法确认人脸位置，请面向摄像头。" }
        return "无法读取有效人脸，请暂停后重新开始。"
    }
}
