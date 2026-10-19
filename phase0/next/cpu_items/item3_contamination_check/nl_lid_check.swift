// Cross-check for nl_lid: the class method NLLanguageRecognizer.dominantLanguage(for:), one
// label per input line ("nil" if none). usage: nl_lid_check <in.txt> <out.txt>
import Foundation
import NaturalLanguage
let a = CommandLine.arguments
let data = try Data(contentsOf: URL(fileURLWithPath: a[1]))
guard data.last == 0x0A else { exit(3) }
var lines: [String] = []
data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
    let p = raw.bindMemory(to: UInt8.self)
    var start = 0
    for i in 0..<p.count where p[i] == 0x0A {
        let s = String(decoding: UnsafeBufferPointer(rebasing: p[start..<i]), as: UTF8.self)
        start = i + 1
        lines.append(NLLanguageRecognizer.dominantLanguage(for: s)?.rawValue ?? "nil")
    }
}
try (lines.joined(separator: "\n") + "\n").write(toFile: a[2], atomically: true, encoding: .utf8)
FileHandle.standardError.write("nl_lid_check: \(lines.count) strings\n".data(using: .utf8)!)
