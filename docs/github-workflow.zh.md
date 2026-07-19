# GitHub Issue 与 Pull Request 工作流

本文定义 DLEngine 团队统一使用的 GitHub 协作流程。目标是让每项工作都有明确归属、验收条件和关闭方式，并让 Issue、Pull Request（PR）与分支状态始终一致。

## 1. 核心原则

1. **Issue 描述要解决的问题，PR 描述一次可审查的代码变更。**
2. **Epic 管目标，Workstream 管方向，Task/Bug 管交付。**
3. **只有满足全部验收条件的叶子 Issue 才能被 PR 自动关闭。**
4. **一个 PR 只保留一个主要 review 目标。** 正确性、性能优化和大规模重构应尽量拆开。
5. **GitHub Sub-issues 是层级关系的唯一事实来源。** 不在父 Issue 正文中重复维护任务 checkbox。
6. **未记录在 Issue 中的 follow-up 不算已安排。** PR 中发现的遗留工作必须在合入前创建或关联 Issue。

## 2. Issue 类型与层级

### 2.1 Epic：完整目标

Epic 描述跨模块或跨阶段的最终结果，例如支持一个新模型。它应包含背景、最终目标、整体退出条件、正式关联的 Workstream Sub-issues，以及关键依赖和风险。

Epic 通常由多个 Issue 和 PR 共同完成，**任何单个 PR 都不得使用 `Closes` 自动关闭 Epic**。所有子任务完成并验证最终目标后，由负责人手动关闭。

### 2.2 Workstream：独立工作方向

Workstream 是 Epic 下可以独立推进的方向，例如 Pipeline Parallelism、Indexer 或 HiSparse。它应包含范围与非目标、完成标准、下属 Task/Bug Sub-issues，以及已完成和剩余工作的简要索引。

如果一个 Workstream 需要多个 PR，PR 只能写 `Refs #<issue>`。只有 Workstream 足够小、一个 PR 确实完成全部验收条件时，才允许该 PR 写 `Closes`。

### 2.3 Task / Bug：具体交付

Task 或 Bug 是最小的可交付单元，通常由一个 PR 完成。创建时必须说明：

- 问题、目标或复现方法；
- 范围和非目标；
- 验收条件；
- 验证方法；
- Parent Issue、负责人和优先级。

这是最适合通过 `Closes #<issue>` 自动关闭的 Issue 类型。

### 2.4 Investigation / RFC：调查与决策

根因或方案尚不明确时，先创建 Investigation 或 RFC，不要直接创建内容模糊的实现任务。它应产出复现数据或现状分析、候选方案与取舍、最终结论，以及后续实现 Task。调查完成且后续任务已经创建后，手动关闭。

### 2.5 推荐层级

```text
Epic
├── Workstream
│   ├── Task / Bug
│   └── Task / Bug
└── Workstream
    ├── Investigation / RFC
    └── Task / Bug
```

跨方向相关但不属于父子关系的 Issue，在正文中写 `Related to #<issue>`，不要为了展示关联而强行调整层级。

## 3. Issue 创建规范

开始开发前，先搜索现有 Issue：

1. 已存在同一目标：复用并补全原 Issue；
2. 已存在 Epic/Workstream：创建叶子 Issue，并设置为正式 Sub-issue；
3. 没有对应工作：先创建 Issue，再开始长期开发；
4. 很小且显而易见的修复可以直接开 PR，但正文仍须说明问题和验收方式。

推荐 Issue 模板：

```markdown
## Goal

要解决的问题和期望结果。

## Scope

- 本 Issue 包含什么

## Non-goals

- 本 Issue 明确不包含什么

## Acceptance criteria

- [ ] 可验证条件 1
- [ ] 可验证条件 2

## Validation

测试、benchmark、运行环境或复现方式。

## Relationships

- Parent: #<issue>
- Related: #<issue>
```

创建后应立即设置负责人、`type:*` 标签、一个或多个 `area:*` 标签，以及必要的优先级和阻塞状态。没有负责人或没有验收条件的 Issue，不应被视为“正在开发”。

## 4. 分支策略

### 4.1 目标分支

当前仓库以 **`Pure_dp` 作为日常集成分支**：

- 普通功能、修复、性能优化和文档 PR 默认 target `Pure_dp`；
- `Pure_dp -> main` 只用于明确的阶段性发布或同步；
- 除紧急 hotfix 外，功能 PR 不直接 target `main`；
- stacked PR 必须在正文注明依赖关系和临时 base，前置 PR 合入后及时 retarget。

### 4.2 分支命名

```text
feature/<issue>-<short-description>
fix/<issue>-<short-description>
perf/<issue>-<short-description>
docs/<short-description>
```

自动化开发代理可以使用 `agent/<short-description>`。一个分支只服务一个 PR，不要把互不相关的修改追加到已经进入 review 的分支。

## 5. PR 拆分与创建

### 5.1 什么时候拆分

出现以下情况时，优先拆成多个 PR：

- 正确性支持与性能优化；
- 行为修改与大规模重构；
- 不同模块、不同风险或不同验证方式；
- reviewer 无法用一句话说明 PR 的主要 review 目标；
- 部分修改可以独立合入或独立回滚。

拆分后的 PR 可以共同 `Refs` 同一个 Workstream，但每个 PR 应独立描述自己的边界和验证结果。

### 5.2 Draft 与 Ready for review

开发开始后可以尽早创建 Draft PR。满足以下条件后才能转为 Ready for review：

- diff 范围稳定且不包含无关文件；
- 已完成与风险匹配的测试；
- 性能修改提供可复现的 benchmark 与对照数据；
- 已知限制和 follow-up 已写入 PR 或对应 Issue；
- PR 正文与实际代码一致。

### 5.3 PR 正文模板

```markdown
Refs #<workstream-or-related-issue>
Closes #<completed-leaf-issue>

## Why

为什么需要这项修改。

## What changed

- 主要修改 1
- 主要修改 2

## Review boundary

本 PR 需要重点审查什么，以及明确不包含什么。

## Validation

- [ ] Unit tests
- [ ] Integration / end-to-end tests
- [ ] Benchmark（性能 PR 必填）

## Known limitations and follow-ups

- #<follow-up-issue>
```

没有适用的 `Closes` 时删除该行，不要为了自动关闭 Issue 而保留它。

## 6. `Refs`、`Closes` 与关闭语义

以下情况使用 `Refs #N`：

- PR 只完成 Issue 的一部分；
- Issue 是 Epic 或包含多个 PR 的 Workstream；
- PR 与该 Issue 相关，但不是其最终交付；
- PR 依赖另一个 PR 或 Issue。

只有同时满足以下条件才使用 `Closes #N`：

1. `#N` 是叶子 Task/Bug，或确实可以由一个 PR 完成的小型 Workstream；
2. PR 合入后，Issue 的所有验收条件均成立；
3. 没有必须完成但尚未创建 Issue 的遗留工作；
4. 对应验证已经完成，或明确属于合入后执行且有负责人。

推荐一个 PR 只关闭一个主要叶子 Issue，其他关联项使用 `Refs`。

```text
# 错误：单个 PR 关闭仍有剩余工作的上层 Issue
Closes #<epic>
Closes #<workstream-with-remaining-work>

# 正确：引用上层工作，只关闭已完成的叶子任务
Refs #<epic>
Refs #<workstream>
Closes #<completed-leaf-task>
```

## 7. Review、合入与后续工作

### 7.1 Reviewer 检查清单

- Issue 与 PR 的范围是否一致；
- `Refs` / `Closes` 是否符合实际完成度；
- base branch 是否正确，diff 是否包含无关修改；
- 正确性测试是否覆盖关键路径；
- 性能结论是否有基线、环境和原始数据；
- 新增复杂度是否有必要的注释或设计文档；
- 已知问题是否已经创建 follow-up Issue。

### 7.2 合入规则

- 至少获得该模块负责人的 review；
- 必需 CI 通过，分支与目标分支不存在未解决冲突；
- 性能 PR 必须同时检查正确性和资源占用；
- 修改已 review 的提交历史时应通知 reviewer；force push 只允许 `--force-with-lease`。

### 7.3 合入后

PR 作者或 Issue 负责人应确认：

1. 应自动关闭的叶子 Issue 已正确关闭；
2. Epic/Workstream 仍保持正确状态；
3. 验收条件和最终结论已更新；
4. follow-up Issue 已创建、关联并分配负责人；
5. 临时 stacked PR 已 retarget。

## 8. 标签和状态

标签保持精简，每个 Issue 通常使用 2～4 个：

```text
type:epic / type:feature / type:bug / type:perf / type:rfc
area:glm52 / area:indexer / area:pp / area:hisparse
priority:P0 / priority:P1 / priority:P2
status:blocked / status:needs-validation
```

不要创建与 GitHub 原生状态重复的标签，例如 `status:open`。Draft/Ready 使用 PR 原生状态表示。

- **Open + assignee**：已进入计划或正在开发；
- **Open + `status:blocked`**：等待明确的外部条件；
- **Open + `status:needs-validation`**：代码可能完成，但验收尚未完成；
- **Closed**：验收条件已满足、问题不再适用或已被替代；关闭时应说明原因。

## 9. 定期维护

团队每周进行一次简短 triage：

- 检查没有负责人、验收条件或 Parent 的活跃 Issue；
- 检查 base branch 异常、CI 失败或长期 Draft 的 PR；
- 14 天无更新时提醒作者；
- 30 天无更新且没有 owner/计划时，关闭或标记 stale；
- 对被替代、已 revert 或重复的 Issue 补充结论后关闭；
- 检查 Epic 正文是否与正式 Sub-issues 重复或冲突。

关闭长期无进展的 Issue/PR 不代表否定工作；后续可以重新打开，或在保留历史链接的前提下创建范围更清晰的新任务。

## 10. 快速判断

提交前依次回答：

1. 这项工作属于哪个 Epic 或 Workstream？
2. 是否已经有可以复用的 Issue？
3. 这个 PR 是否只有一个主要 review 目标？
4. 它应该 `Refs`，还是确实能够 `Closes` 对应 Issue？
5. base 是否为 `Pure_dp`，或是否已说明例外原因？
6. 测试、benchmark、限制与 follow-up 是否齐全？

如果其中任何一项说不清，应先整理 Issue/PR 边界，再进入 review。
