import Foundation

/// REST client for the dog-side `api` service (protocol §1).
struct APIClient {
    let baseURL: URL

    init(host: String) {
        let cleaned = host.trimmingCharacters(in: .whitespaces)
        self.baseURL = URL(string: "http://\(cleaned)") ?? URL(string: "http://127.0.0.1:8200")!
    }

    private func post(_ path: String, body: [String: String]) async throws {
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)
        let (_, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
    }

    func submitTask(_ text: String) async throws {
        try await post("task", body: ["text": text])
    }

    func estop() async throws {
        try await post("estop", body: ["source": "app"])
    }

    func release() async throws {
        try await post("release", body: [:])
    }

    func health() async throws {
        let (data, response) = try await URLSession.shared.data(from: baseURL.appendingPathComponent("health"))
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        _ = data
    }
}
