# JoFaceGuard

原生 macOS 菜单栏人脸守卫。摄像头连续约 2 秒确认同一个清晰的陌生人后，向 macOS 请求锁屏。Jo、无人、不确定、画面中断和多人同框均不触发锁屏。

这是 [KINN-CH/FaceUnlock](https://github.com/KINN-CH/FaceUnlock) 的独立 MIT fork，保留五点对齐、坐标处理和向量计算，移除了自动解锁、登录密码、密码注入、FileVault 辅助流程和联网更新。模型采用 **OpenCV SFace → Core ML**，避免原 ArcFace/InsightFace 预训练权重的研究用途限制。来源、具体保留内容、候选比较和许可证见 [UPSTREAM.md](docs/UPSTREAM.md)。

## Jo 的使用步骤

1. 将 `JoFaceGuard.zip` 解压后的 App 放入「应用程序」，打开 **JoFaceGuard**。菜单栏会出现人脸图标。初始状态为暂停，摄像头关闭。
2. 点击 **录入 Jo**，由 macOS 提示允许摄像头。光线要足够，画面里只有你；平时戴眼镜就戴着录入。
3. 跟随 5 个姿态提示，每步点 **采集当前姿态**。每步收集 3 张有效特征，合计 15 张。姿态是操作指引，软件不强制证明你完成了每一种姿态；请按提示做。过暗、模糊、过侧或特征不一致会等待清晰画面。取消不会覆盖原资料。
4. 点击 **观察识别**。这个模式绝不自动锁屏。先检查正常坐姿、轻微左右转头、常戴的眼镜和正常房间光线；本人应显示 Jo，暗光应显示不确定，无人不应开始计时。让一位知情的其他人试坐，观察是否稳定显示陌生人。
5. 观察模式连续 5 帧认出 Jo 后，**开启自动锁屏** 按钮可用。开启后，陌生人连续确认约 2 秒才请求锁屏。摄像头启动、检测和推理的耗时另计；这是从第一张合格陌生人画面起算，不是保证入座后正好 2 秒。
6. 菜单栏 **暂停守卫与摄像头** 会立即使待处理结果失效并停止摄像头。锁屏、休眠、切换用户后也会暂停；解锁后需手动重新观察并开启，避免反复锁屏。

关闭设置窗口后，守卫仍可在菜单栏运行；关闭录入窗口会取消录入。退出 App 会停止守卫。不会自动加入登录项。

删除资料：窗口右下角 **删除人脸资料**，确认后删除本机钥匙串中的特征。卸载前建议先删除资料，再退出并将 App 移到废纸篓。直接删除 App 不会自动删除钥匙串资料。

## 运行环境与构建

- macOS 14 或更高、Apple Silicon；本次在 macOS 27.2 / arm64 / Swift 6.4 Command Line Tools 上构建。
- 摄像头权限；不需要辅助功能权限，不读取登录密码。
- 从源码构建需要 Apple Command Line Tools。一次性模型转换另需 Python 3.9–3.12，依赖只装在项目虚拟环境。
- App 运行不需要 Python，也不会下载模型或发出网络请求。

```sh
git clone https://github.com/yuezjo/JoFaceGuard.git
cd JoFaceGuard
make model       # 下载固定版本 SFace，校验 SHA256，转换并核对数值
make test        # 分类、计时、图像质量、对齐测试
make model-test  # Swift 输入与原 ONNX 的数值对照
make release     # 原生编译、签名、模型加载测试，生成 build/JoFaceGuard.zip
make install     # 新安装到 /Applications；已有同名 App 时拒绝覆盖
make run
```

构建工具会在临时目录签名，再输出 ZIP，避免 iCloud Documents 自动附加的 FinderInfo 破坏签名。请将解压后的 App 放在「应用程序」，不要留在同步的 Documents 中运行。仅支持本地 ad-hoc 签名；当前没有 Developer ID 公证。下载到其他 Mac 后 Gatekeeper 可能要求系统设置中的明确允许。重新构建后 macOS 可能重新请求摄像头/钥匙串访问。

有 Developer ID 时可运行 `make release CODESIGN_ID='Developer ID Application: …'`。这不包含公证步骤，也不是 App Store 构建。

## 判断流程

`AVFoundation 原始 BGRA → Vision 人脸/关键点/采集质量 → 原始光照与模糊检查 → 五点 112×112 对齐 → Core ML SFace → 128 维单位向量 → 三档分类 → 连续计时 → macOS 锁屏请求`

| 情况 | 行为 |
| --- | --- |
| 最佳 Jo 样本 cosine ≥ 0.50 | Jo，清零计时 |
| 所有 Jo 样本 cosine ≤ 0.20，且质量合格 | Unknown，允许开始连续确认 |
| 相似度在两者之间、没有有效资料、推理失败 | Uncertain，清零计时 |
| 太暗、过曝、模糊、脸太小/被裁切、姿态太偏、缺少质量分数 | Uncertain，清零计时 |
| 多人同框（包括 Jo + 陌生人） | Uncertain，清零计时 |
| 正常画面没有人脸 | No face，清零计时 |

这些阈值是保守起点，不是经 Jo 实测得到的概率或准确率。SFace 上游的普通人脸验证阈值不能直接证明本场景可靠。

连续陌生人确认使用单调的采集时间，需要 ≥2 秒且至少 8 个有效样本。相邻有效帧间隔不得超过 0.5 秒，每帧处理结果不得迟于采集时间 0.5 秒。人脸框必须重叠，同一段中的特征还必须与上一帧及首帧保持一致，避免不同陌生人拼成两秒。任何中断清零；同一连续片段最多请求一次锁屏。

## 本地数据与限制

- 原始图像只在内存中短暂处理和预览，不保存照片、视频或人脸日志，不上传。15 个特征保存在 `io.github.yuezjo.JoFaceGuard` 的本机钥匙串项目中，关闭 iCloud 同步，设备限定、解锁时可访问。
- 只有摄像头能清楚看到的脸才可能触发。遮挡、背对摄像头、暗光、多人或无法确定身份时，按设计保持不锁。它不替代系统自动锁定、密码或你离开前的手动锁屏。
- 普通二维摄像头不等同 Face ID，没有深度/活体认证；照片或屏幕中的 Jo 可能被认作 Jo。这一版只做陌生脸守卫。
- 需要持续使用摄像头，会亮摄像头指示灯，增加耗电；和视频会议软件争用时可能暂停或显示不确定。
- 锁屏使用动态加载的私有 `SACLockScreenImmediate`，可能随 macOS 更新失效。不可用时禁止开启；请求后需要系统状态/通知确认，不会把函数调用当成已锁屏。失败时暂停并显示错误，可使用 **Control–Command–Q**。
- 当前只自动选择内置摄像头，找不到时选第一个外置摄像头；暂时没有摄像头选择器。外置/连续互通摄像头方向和性能尚未实测。
- **本次没有 Jo 的真实录入、陌生人测试、实际锁屏调用或长时间误锁率测试。** 自动测试证明逻辑与模型输入正确，不证明现场识别效果。详见 [验证记录](docs/VALIDATION.md)。

## 关键代码

- `Sources/JoFaceGuard/FacePipeline.swift`：显式 Vision 采集质量请求、光线/清晰度/姿态门限。
- `Sources/FaceUnlock/Core/FaceAligner.swift`、`FaceGeometry.swift`：保留自上游的五点对齐和像素方向。
- `Sources/JoFaceGuard/EmbeddingModel.swift`：SFace **raw RGB 0…255** 输入；不套用 ArcFace 的归一化。
- `Sources/JoFaceGuard/GuardPolicy.swift`：Jo / Unknown / Uncertain 与两秒连续性规则。
- `Sources/JoFaceGuard/GuardController.swift`：录入、观察、守卫、暂停、休眠与异步结果失效。
- `Sources/JoFaceGuard/ScreenLocker.swift`：只请求锁屏，不解锁。
- `Sources/JoFaceGuard/ProfileStore.swift`：本机钥匙串；完整录入后才替换。

## License

App：MIT，保留 © 2026 Cheolho Kim 的原始版权与许可文本。修改说明见本 README 与 `docs/UPSTREAM.md`。

SFace 模型：Apache 2.0；完整许可为 `Resources/SFACE_LICENSE.txt`。模型从 ONNX 转为 Core ML 的修改和 attribution 保存在 `docs/UPSTREAM.md`，并一同打包到 App。源码仓库不上传权重，构建工具从固定上游版本下载。构建后的安装包含模型和两份许可。
