import SwiftUI

/// Go2Console 的视觉语言：安静的表面、一个强强调色、按事件类型着色的语义色。
/// 表面色一律用 `.primary` 的透明度派生，亮/暗模式自动适配。
enum Theme {
    /// 品牌强调色 —— 信号蓝。
    static let accent = Color(red: 0.22, green: 0.55, blue: 1.00)
    /// 急停红。
    static let danger = Color(red: 1.00, green: 0.30, blue: 0.35)
    /// 警示橙。
    static let caution = Color(red: 1.00, green: 0.62, blue: 0.15)
    /// 正常绿。
    static let ok = Color(red: 0.22, green: 0.78, blue: 0.44)

    // MARK: 渐变

    static let accentGradient = LinearGradient(
        colors: [
            Color(red: 0.31, green: 0.62, blue: 1.00),
            Color(red: 0.16, green: 0.44, blue: 0.96),
        ],
        startPoint: .top,
        endPoint: .bottom
    )

    static let dangerGradient = LinearGradient(
        colors: [
            Color(red: 1.00, green: 0.40, blue: 0.42),
            Color(red: 0.91, green: 0.20, blue: 0.29),
        ],
        startPoint: .top,
        endPoint: .bottom
    )

    static let cautionGradient = LinearGradient(
        colors: [
            Color(red: 1.00, green: 0.69, blue: 0.25),
            Color(red: 0.97, green: 0.52, blue: 0.12),
        ],
        startPoint: .top,
        endPoint: .bottom
    )

    // MARK: 表面（透明度派生，明暗自适应）

    /// 卡片底色。
    static var cardFill: Color { Color.primary.opacity(0.045) }
    /// 卡片描边 / 发丝线。
    static var cardStroke: Color { Color.primary.opacity(0.09) }
    /// 小控件底色。
    static var chipFill: Color { Color.primary.opacity(0.06) }

    // MARK: 圆角

    enum Radius {
        static let chip: CGFloat = 8
        static let card: CGFloat = 12
        static let panel: CGFloat = 16
    }
}
