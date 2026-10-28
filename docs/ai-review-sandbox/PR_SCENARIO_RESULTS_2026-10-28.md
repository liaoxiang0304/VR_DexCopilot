# 独立 PR 场景验收：2026-10-28

共享仓库 IS-Model-Framework/.github 的测试提交 e727d45 已从 PR #24 撤回，分支恢复到 **0d6ca9cebfae75380c9588ba6decfc0b9fdcafa3**。本轮测试代码、工作流、Review、回复、Issue 和测试分支均只放在 VR_DexCopilot。

18 个独立 Draft PR 的实际业务 Check 全部符合声明的预期：**7 个通过、11 个阻断**。所有 PR 保留供核查，不进行合并。

[最终验收运行](https://github.com/liaoxiang0304/VR_DexCopilot/actions/runs/40672879252) · [原始 JSON 结果](pr-scenario-results-20261028.json) · [输入与人工回复收据](pr-scenarios-20261028.json)

## 模拟范围

两个 AI 的身份由 github-actions[bot] 真实发布并签署的两份 Review 模拟，分别带 [SANDBOX AI FIXTURE: CODEX] 与 [SANDBOX AI FIXTURE: INFISCALE] 标记。仅测试身份在 VR_DexCopilot 的沙箱适配器中映射到检查器对应 login；没有假装由真实 Codex/Infiscale 账号发言。

每个 PR 的 Review 状态、行内线程、人工回复、原始作者/编辑者、当前 PR commit 集合和 Issue 状态均实时从 GitHub 读取。没有替换为整份虚构 snapshot。业务 Check 的标题/摘要明确显示 SIMULATED / SANDBOX SIMULATION。

## 逐 PR 结果

| 场景 | PR | 预期 | 实际业务 Check |
| --- | --- | --- | --- |
| 双 AI 完成，无意见 | [#15](https://github.com/liaoxiang0304/VR_DexCopilot/pull/15) | 通过 | [通过](https://github.com/liaoxiang0304/VR_DexCopilot/runs/122758575581) |
| 双 AI 完成，缺人工处理回复 | [#16](https://github.com/liaoxiang0304/VR_DexCopilot/pull/16) | 阻断 | [阻断](https://github.com/liaoxiang0304/VR_DexCopilot/runs/122758625271) |
| Accepted 引用当前 PR 的完整 Fix SHA | [#17](https://github.com/liaoxiang0304/VR_DexCopilot/pull/17) | 通过 | [通过](https://github.com/liaoxiang0304/VR_DexCopilot/runs/122758667992) |
| Rejected 有有效人工理由 | [#18](https://github.com/liaoxiang0304/VR_DexCopilot/pull/18) | 通过 | [通过](https://github.com/liaoxiang0304/VR_DexCopilot/runs/122758723555) |
| Invalid 有有效人工理由 | [#19](https://github.com/liaoxiang0304/VR_DexCopilot/pull/19) | 通过 | [通过](https://github.com/liaoxiang0304/VR_DexCopilot/runs/122758745376) |
| Deferred 指向开放 Issue | [#21](https://github.com/liaoxiang0304/VR_DexCopilot/pull/21) | 通过 | [通过](https://github.com/liaoxiang0304/VR_DexCopilot/runs/122758779502) |
| Accepted 的 Fix 不属于当前 PR | [#22](https://github.com/liaoxiang0304/VR_DexCopilot/pull/22) | 阻断 | [阻断](https://github.com/liaoxiang0304/VR_DexCopilot/runs/122758831094) |
| Deferred 指向已关闭 Issue | [#23](https://github.com/liaoxiang0304/VR_DexCopilot/pull/23) | 阻断 | [阻断](https://github.com/liaoxiang0304/VR_DexCopilot/runs/122758871588) |
| 只有 Bot 代回复 | [#24](https://github.com/liaoxiang0304/VR_DexCopilot/pull/24) | 阻断 | [阻断](https://github.com/liaoxiang0304/VR_DexCopilot/runs/122758912734) |
| Bot 编辑人工模板 | [#25](https://github.com/liaoxiang0304/VR_DexCopilot/pull/25) | 阻断 | [阻断](https://github.com/liaoxiang0304/VR_DexCopilot/runs/122758929830) |
| 仅 Resolve，无处理模板 | [#26](https://github.com/liaoxiang0304/VR_DexCopilot/pull/26) | 阻断 | [阻断](https://github.com/liaoxiang0304/VR_DexCopilot/runs/122758995027) |
| 模板有效，但线程未 Resolve | [#27](https://github.com/liaoxiang0304/VR_DexCopilot/pull/27) | 通过 | [通过](https://github.com/liaoxiang0304/VR_DexCopilot/runs/122758997575) |
| 较新的无效模板覆盖旧有效回复 | [#28](https://github.com/liaoxiang0304/VR_DexCopilot/pull/28) | 阻断 | [阻断](https://github.com/liaoxiang0304/VR_DexCopilot/runs/122759066801) |
| 普通讨论保留旧有效处理回复 | [#29](https://github.com/liaoxiang0304/VR_DexCopilot/pull/29) | 通过 | [通过](https://github.com/liaoxiang0304/VR_DexCopilot/runs/122759085595) |
| 实际 force-push 移除 Accepted Fix | [#30](https://github.com/liaoxiang0304/VR_DexCopilot/pull/30) | 阻断 | [阻断](https://github.com/liaoxiang0304/VR_DexCopilot/runs/122762768769) |
| 真实 Issue 在两次校验间关闭 | [#32](https://github.com/liaoxiang0304/VR_DexCopilot/pull/32) | 阻断 | [阻断](https://github.com/liaoxiang0304/VR_DexCopilot/runs/122759185669) |
| 仅一个 AI 完成的反向对照 | [#33](https://github.com/liaoxiang0304/VR_DexCopilot/pull/33) | 阻断 | [阻断](https://github.com/liaoxiang0304/VR_DexCopilot/runs/122759226364) |
| 模板缺少必填 Reason | [#34](https://github.com/liaoxiang0304/VR_DexCopilot/pull/34) | 阻断 | [阻断](https://github.com/liaoxiang0304/VR_DexCopilot/runs/122759239096) |

表中的“通过/阻断”指固定名称 AI Review / Disposition Check 的本次模拟身份结果。验收 job 绿色表示其断言匹配预期，因此“应阻断”的场景也会有绿色验收 job。

## 可核查的实际状态

- PR #25 的两条人工回复由 liaoxiang0304 创建，GitHub GraphQL editor 为 github-actions，编辑时间分别为 2026-10-28T10:30:42Z 与 10:30:43Z；[第一条回复](https://github.com/liaoxiang0304/VR_DexCopilot/pull/25#discussion_r4399457630)、[第二条回复](https://github.com/liaoxiang0304/VR_DexCopilot/pull/25#discussion_r4399457975)。仅映射 fixture Bot 的审查角色，不改人工回复的真实 editor。
- PR #26 两个线程均实际 resolved，但没有处理模板，业务 Check 阻断。
- PR #27 两个线程均实际 unresolved，模板有效，业务 Check 通过；GitHub 的 conversation resolution 仍单独要求 Resolve。
- PR #30 的 Accepted 引用 6e7adea0b8554fcd81ca9c34e27c66bc96ba7ef8，随后仅对这个测试分支实际 force-push；当前 PR commit 集合只剩 95e9cb2ee80567fdcdf20de4098b79d443ea6e02，引用的 Fix 已不属于当前 PR，业务 Check 阻断。
- PR #32 的真实 Tracking Issue 在第一次校验返回 open 后被关闭，第二次校验读到 closed；日志记录两次验证读取，最终阻断。
- 真实 AI 账号接入、第二人工账户的 write/read/撤权操作，以及实际合并放行不属于这次模拟身份验收。

## 复查

在 Actions 中运行 AI Review Separate PR Scenarios，operation 选择 evaluate，可按这些 PR 的当前真实状态复查。若人工改动了场景状态，结果可能不再符合清单中的预期，应先确认 PR 输入。

已有普通生产 caller 仍使用真实 Bot 配置，因此它在夹具准备时可能留下因真实 Bot 未完成而失败的原生 job。表中链接给出了本轮最终业务 Check，摘要均明确标注身份模拟。全体测试 PR 保持 Draft；测试分支的规则还要求旧的 Disposition Integration Test，不能把本轮业务 Check 通过当作已验证实际合并。
