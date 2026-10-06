import AVFoundation
import CoreImage
import QuartzCore

/// Adapted from FaceUnlock's AVFoundation capture approach. All capture and ML work
/// stays on one serial queue. Frames carry capture time, not inference completion time.
final class CameraCapture: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    let queue = DispatchQueue(label: "io.github.yuezjo.JoFaceGuard.camera", qos: .userInitiated)
    var onFrame: ((CVPixelBuffer, Double, Int) -> Void)?
    var onFailure: ((String, Int) -> Void)?
    private var output: AVCaptureVideoDataOutput?
    private var generation = 0
    private var lastFrame: Double = 0
    private var notificationTokens: [NSObjectProtocol] = []

    override init() {
        super.init()
        for name in [AVCaptureSession.runtimeErrorNotification, AVCaptureSession.wasInterruptedNotification] {
            notificationTokens.append(NotificationCenter.default.addObserver(forName: name, object: session, queue: nil) { [weak self] _ in
                self?.queue.async { [weak self] in
                    guard let self else { return }
                    self.onFailure?("摄像头已中断，请暂停后重新开始。", self.generation)
                }
            })
        }
    }
    deinit { notificationTokens.forEach(NotificationCenter.default.removeObserver) }

    func start(generation: Int) {
        queue.async {
            self.generation = generation
            self.lastFrame = 0
            do {
                if self.output == nil { try self.configure() }
                if !self.session.isRunning { self.session.startRunning() }
                if !self.session.isRunning { throw GuardError.message("摄像头未能启动") }
            } catch { self.onFailure?(error.localizedDescription, generation) }
        }
    }
    func stop() {
        queue.async {
            self.generation = -1
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    private func configure() throws {
        let devices = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera, .external], mediaType: .video, position: .unspecified).devices
        guard let device = devices.first(where: { $0.deviceType == .builtInWideAngleCamera }) ?? devices.first else {
            throw GuardError.message("没有发现摄像头，请连接摄像头后重试。")
        }
        let input = try AVCaptureDeviceInput(device: device)
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        for old in session.inputs { session.removeInput(old) }
        for old in session.outputs { session.removeOutput(old) }
        if session.canSetSessionPreset(.hd1280x720) { session.sessionPreset = .hd1280x720 }
        guard session.canAddInput(input) else { throw GuardError.message("无法使用摄像头输入") }
        session.addInput(input)
        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else { throw GuardError.message("无法使用摄像头画面") }
        session.addOutput(output)
        if let connection = output.connection(with: .video), connection.isVideoMirroringSupported {
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = false
        }
        self.output = output
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let now = CACurrentMediaTime()
        guard generation >= 0, now - lastFrame >= 0.18 else { return }
        lastFrame = now
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            onFailure?("无法读取摄像头画面", generation); return
        }
        // AVCapture timestamps use the host clock. Invalid or stale timestamps fail closed to uncertainty.
        let captured = CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(sampleBuffer))
        guard captured.isFinite, captured <= now + 0.05, now - captured < 0.5 else {
            onFailure?("摄像头画面延迟过高，请重新开始。", generation); return
        }
        onFrame?(buffer, captured, generation)
    }
}
