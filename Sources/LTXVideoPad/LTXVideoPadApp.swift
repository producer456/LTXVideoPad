import SwiftUI

@main
struct LTXVideoPadApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                Image(systemName: "video.badge.waveform")
                    .font(.system(size: 60))
                    .foregroundStyle(.blue)

                Text("LTXVideoPad")
                    .font(.largeTitle)
                    .fontWeight(.bold)

                Text("On-device image-to-video generation")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)

                Text("Phase 1: T5 Encoder")
                    .font(.caption)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 6)
                    .background(.blue.opacity(0.15), in: Capsule())
            }
            .navigationTitle("LTXVideoPad")
        }
    }
}
