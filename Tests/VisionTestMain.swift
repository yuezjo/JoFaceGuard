import Foundation
@main enum VisionTestMain {
    static func main() { exit(Int32(QualityTests.run() + AlignmentTests.run())) }
}
