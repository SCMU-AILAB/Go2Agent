# Go2Console

Go2Agent 的原生控制台 App（macOS + iOS），SwiftUI 单代码库双平台。协议见
`docs/superpowers/specs/2026-10-14-app-protocol.md`。

## 结构

```
frontend/
  Go2Console.xcodeproj/
  Go2Console/            # 共享源码（文件系统同步组，增删文件无需改工程）
    Go2ConsoleApp.swift  # 入口
    Models.swift         # 协议模型（对齐 go2agent.core.types）
    APIClient.swift      # REST：task / estop / release / health
    EventStream.swift    # WS：状态+事件（文本）、帧流（二进制），自动重连
    MockFeed.swift       # 内置演示数据源（无后端开发用）
    ConsoleStore.swift   # @Observable 状态中枢
    ConsoleView.swift    # 主布局 + 急停 + 设置
    CameraView.swift     # 画面 + 编号跟踪框叠加
    TimelineView.swift   # 时间线 + 思考详情展开
  Support/Info.plist     # 含 NSAllowsLocalNetworking（局域网 HTTP）
```

## 构建与运行

用 Xcode 打开 `Go2Console.xcodeproj`，选 Go2Console scheme 直接 ⌘R（一个 target 同时支持
macOS 与 iOS 模拟器）。

命令行：

```sh
cd frontend
xcodebuild -project Go2Console.xcodeproj -scheme Go2Console \
  -destination 'platform=macOS,arch=arm64' -derivedDataPath .build/DerivedData build
```

### 受限环境下宏编译失败的解决办法

若 swift-plugin-server 被系统沙箱策略拦截（报 `macro ... could not be found ...
produced malformed response`），追加进程内宏库加载，绕过外部插件进程：

```sh
PLUGINS=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/usr/lib/swift/host/plugins
xcodebuild ... build OTHER_SWIFT_FLAGS="\
 -load-plugin-library $PLUGINS/libSwiftUIMacros.dylib \
 -load-plugin-library $PLUGINS/libObservationMacros.dylib \
 -load-plugin-library $PLUGINS/libFoundationMacros.dylib \
 -load-plugin-library $PLUGINS/libSwiftMacros.dylib \
 -load-plugin-library $PLUGINS/libPreviewsMacros.dylib"
```

正常终端（Xcode GUI 或无限制 shell）不需要这段参数。

## 演示模式

默认以「演示数据」模式启动：MockFeed 在本地渲染合成场景（10 fps、两个移动的人形目标）、
按脚本回放任务/决策/障碍/丢失等事件，页面开发完全不依赖狗端。设置里切到「连接狗端」并填
`<dog-host>:8200` 即连真机。
