// Language ID with Apple's NaturalLanguage framework (the OS's own model; nothing downloaded).
// usage: nl_lid <in.txt> <out.tsv>
//   in.txt : UTF-8 strings separated by "\n" only (a final "\n" is required; no string holds "\n").
//   out.tsv: one row per input string, in order:
//     dominant  top1  p_top1  p_en  p_fr  dominant_2way  p2_en  p2_fr
//   dominant      NLLanguageRecognizer.dominantLanguage over all its languages ("nil" if none)
//   top1, p_top1  the argmax of languageHypotheses(withMaximum: 1000) (ties: smaller code)
//   p_en, p_fr    hypothesis probability of English / French (0 if absent)
//   *_2way        the same with languageConstraints = [.english, .french]
// A fresh recognizer per string (no state can leak between strings).
import Foundation
import NaturalLanguage

let a = CommandLine.arguments
guard a.count == 3 else { FileHandle.standardError.write("usage: nl_lid in out\n".data(using: .utf8)!); exit(2) }
let data = try Data(contentsOf: URL(fileURLWithPath: a[1]))
guard data.last == 0x0A else { FileHandle.standardError.write("input must end with LF\n".data(using: .utf8)!); exit(3) }
FileManager.default.createFile(atPath: a[2], contents: nil)
let out = FileHandle(forWritingAtPath: a[2])!
var buf = ""
buf.reserveCapacity(1 << 22)
var n = 0
func fmt(_ x: Double) -> String { return String(format: "%.9g", x) }
data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
    let p = raw.bindMemory(to: UInt8.self)
    var start = 0
    for i in 0..<p.count where p[i] == 0x0A {
        let s = String(decoding: UnsafeBufferPointer(rebasing: p[start..<i]), as: UTF8.self)
        start = i + 1
        let r = NLLanguageRecognizer()
        r.processString(s)
        let dom = r.dominantLanguage?.rawValue ?? "nil"
        let h = r.languageHypotheses(withMaximum: 1000)
        var top = "nil"; var tp = -1.0
        for (k, v) in h { if v > tp || (v == tp && k.rawValue < top) { tp = v; top = k.rawValue } }
        if h.isEmpty { tp = 0 }
        let r2 = NLLanguageRecognizer()
        r2.languageConstraints = [.english, .french]
        r2.processString(s)
        let dom2 = r2.dominantLanguage?.rawValue ?? "nil"
        let h2 = r2.languageHypotheses(withMaximum: 2)
        buf += "\(dom)\t\(top)\t\(fmt(tp))\t\(fmt(h[.english] ?? 0))\t\(fmt(h[.french] ?? 0))\t\(dom2)\t\(fmt(h2[.english] ?? 0))\t\(fmt(h2[.french] ?? 0))\n"
        n += 1
        if buf.utf8.count > (1 << 22) { out.write(buf.data(using: .utf8)!); buf = "" }
    }
}
out.write(buf.data(using: .utf8)!)
out.closeFile()
FileHandle.standardError.write("nl_lid: \(n) strings\n".data(using: .utf8)!)
