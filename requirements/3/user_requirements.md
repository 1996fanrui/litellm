# ISSUE-3: fastest_response 竞速修复 - 用户需求

## 问题描述

1. **竞速永远是第一个 model 赢**：无论 model 列表顺序如何，litellm 的 `fastest_response: true` 竞速总是第一个 model 获胜，不是真正的竞速
2. **streaming 模式 overhead 过高**：streaming 竞速路径比 non-streaming 多 ~150ms 开销，来自 litellm acompletion 的同步路由处理被串行调度。对于 P99.99 < 800ms 的延迟目标，150ms 本地开销不可接受
3. **Web UI 展示问题**：litellm Web UI 的 request logs 无法验证竞速结果

## 验证标准

### 本地 Mock Server 测试方案

搭建本地 mock server 提供 3 个模型，返回固定字符串，通过 sleep 模拟不同推理延迟：

| 模型名 | Server 端 Sleep | 预期行为 |
|--------|----------------|----------|
| mock-slow (随机位置) | 360ms | 不应该赢 |
| mock-medium (随机位置) | 330ms | 不应该赢 |
| mock-fast (随机位置) | 300ms | 应该永远赢 |

### 必须满足的验证点

1. **正确的 winner**：100 轮测试，每轮随机排列 3 个模型顺序，mock-fast 100% 获胜
2. **正确的延迟**：总耗时 < 329ms（300ms + 最多 29ms 开销），绝不能达到 330ms
3. **streaming 和 non-streaming 都要测试**：两种模式都必须满足以上标准
4. **100 轮全部通过**：不允许有任何失败

### 约束

- 不使用任何付费 API，全部本地 mock
- 使用 OpenAI Python SDK 发起请求（模拟 chatbot 客户端）
- 自行添加日志验证，自行完成所有测试
