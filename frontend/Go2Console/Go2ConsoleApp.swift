import SwiftUI

@main
struct Go2ConsoleApp: App {
    @State private var store = ConsoleStore()

    var body: some Scene {
        WindowGroup("Go2 控制台") {
            ConsoleView(store: store)
        }
        #if os(macOS)
        Settings {
            SettingsView(store: store)
                .frame(width: 500)
        }
        #endif
    }
}
