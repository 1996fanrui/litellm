# ISSUE-3: fastest_response 竞速修复 - 用户需求

## 问题描述

1. **竞速永远是第一个 model 赢**：无论 model 列表顺序如何，litellm 的 `fastest_response: true` 竞速总是第一个 model 获胜，不是真正的竞速
2. **Web UI 展示问题**：litellm Web UI 的 request logs 永远显示第一个 model，无法验证竞速结果
3. **延迟不符合预期**：如果第一个 model 是最慢的 provider，整体延迟应该由最快的 provider 决定，但实际延迟由第一个 model 决定

## 验证标准

### 本地 Mock Server 测试方案

搭建本地 mock server 提供 3 个模型，返回固定字符串，通过 sleep 模拟不同推理延迟：

| 模型名 | Server 端 Sleep | 预期行为 |
|--------|----------------|----------|
| mock-slow (第1个) | 3.0s | 不应该赢 |
| mock-medium (第2个) | 2.7s | 不应该赢 |
| mock-fast (第3个) | 2.4s | 应该永远赢 |

### 必须满足的验证点

1. **正确的 winner**：无论 model 列表顺序如何，mock-fast（sleep 2.4s）永远赢
2. **正确的延迟**：总耗时约 2.4s（+ 少量本地网络开销），绝不能达到 2.7s
3. **Web UI 展示**：litellm Web UI 的 request logs 显示赢家是 mock-fast
4. **streaming 和 non-streaming 都要测试**：两种模式都必须满足以上标准

### 约束

- 不使用任何付费 API，全部本地 mock
- 使用 OpenAI Python SDK 发起请求（模拟 chatbot 客户端）
- 自行添加日志验证，自行完成所有测试
