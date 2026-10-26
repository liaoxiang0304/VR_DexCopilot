# AI Review 门禁隔离验收记录

日期：2026-10-26。仓库：[liaoxiang0304/VR_DexCopilot](https://github.com/liaoxiang0304/VR_DexCopilot)。

**结论：已经完成真实 Actions / Checks API / 合并门禁测试，但尚未全部验收通过，暂不建议直接推广。**

共享脚本与 reusable workflow 均固定到候选提交 `be25a22a1c39b9c64fe8931e858e29829be1ada5`，无需合入公共 CI main。模拟审查者为真实的 `github-actions[bot]`，其评论均明确标注 SANDBOX FIXTURE；本次不声称 Infiscale / Codex 实际完成审查。

## 已验证的场景

| 场景 | 实际结果与证据 |
| --- | --- |
| 没有 AI 完成记录 | 创建 PR 自动失败，不能因零线程放行：[初次运行](https://github.com/liaoxiang0304/VR_DexCopilot/actions/runs/40285069563) |
| 作者行内回复自动复查 | collector → workflow_run 成功：[自动运行](https://github.com/liaoxiang0304/VR_DexCopilot/actions/runs/40285663160) |
| 手动触发 | 原版失败；测试仓库转换输入为数字后成功：[修正后](https://github.com/liaoxiang0304/VR_DexCopilot/actions/runs/40285803976) |
| 普通后续讨论 | 会复查，但沿用有效完成记录和模板：[运行](https://github.com/liaoxiang0304/VR_DexCopilot/actions/runs/40286031460) |
| 有效 Deferred / Issue 关闭 | 关闭后自动全仓复查，PR #3 失败、PR #4 仍被检查且通过；实际合并被拒绝：[运行](https://github.com/liaoxiang0304/VR_DexCopilot/actions/runs/40286270592) |
| Issue 重开 | 未编辑 Deferred 也恢复通过：[运行](https://github.com/liaoxiang0304/VR_DexCopilot/actions/runs/40286378424)；该 sweep 的总状态失败是因为已有 PR #1 无完成记录，不能只看 workflow 总颜色 |
| Accepted 缺 Fix / Fix 被移出 PR | 均失败：[缺 Fix](https://github.com/liaoxiang0304/VR_DexCopilot/actions/runs/40286513076)、[force-push 移除](https://github.com/liaoxiang0304/VR_DexCopilot/actions/runs/40286847889)；有效 Fix 在运行 15 曾通过 |
| resolved 且 outdated 线程 | 实际 GraphQL 状态均为 true；仍要求模板。删除唯一回复后失败，重新回复后通过：[删除](https://github.com/liaoxiang0304/VR_DexCopilot/actions/runs/40287573105)、[恢复](https://github.com/liaoxiang0304/VR_DexCopilot/actions/runs/40287616423) |
| B 处理中，A 曾成功 | 实验固定 Check 仍为 in_progress，实际合并返回 HTTP 405：[运行](https://github.com/liaoxiang0304/VR_DexCopilot/actions/runs/40285932263) |
| B 失败，A 迟到写成功 | 保留 B 的失败，实际合并返回 HTTP 405：[运行](https://github.com/liaoxiang0304/VR_DexCopilot/actions/runs/40285932263) |
| B 取消，A 迟到写成功 | 保留 B 的取消，实际合并返回 HTTP 405：[运行](https://github.com/liaoxiang0304/VR_DexCopilot/actions/runs/40286136866) |
| B 成功，A 迟到写失败 | 保留 B 成功：[运行](https://github.com/liaoxiang0304/VR_DexCopilot/actions/runs/40286025437)；真实 [PR #4](https://github.com/liaoxiang0304/VR_DexCopilot/pull/4) 在两个 Required Checks 通过后合并成功 |

A/B 实验通过待测 `current_check()` 对真实 Checks API 制造迟到写入，并用独立 Check 名 `AI Review / Disposition Integration Test` 验证。它验证 Check 改名、接管和真实合并阻断；不等同于覆盖 GitHub 全部调度竞态。

## 发现的问题

### 1. 手动 PR 编号类型错误，已仅在测试仓库修正

[原始失败运行](https://github.com/liaoxiang0304/VR_DexCopilot/actions/runs/40285281539) 的注释为：`Unexpected value '3'`。PR 编号进入 reusable workflow 时仍为字符串，脚本根本没启动。

需要在公共 CI 的 `docs/examples/ai-review-disposition.yml` 同步这一行；公共 PR #24 本次尚未修改：

```diff
-            || inputs.pr-number
+            || fromJSON(inputs.pr-number || '0')
```

[测试仓库修复提交](https://github.com/liaoxiang0304/VR_DexCopilot/commit/9075f49c568aa79e98c2be6cad76bee669103535) 已实测成功。交付的本地测试包也已同步该修正，因此它含有这处尚未进入公共候选的 caller 修正。

### 2. 同一个 head commit 的两条 PR 尚未通过

[PR #6](https://github.com/liaoxiang0304/VR_DexCopilot/pull/6) 与 [PR #7](https://github.com/liaoxiang0304/VR_DexCopilot/pull/7) 的 head 均为 `2db00048253c38e7090765bec363f07632ec1cdf`。只有 #6 收到测试审查。

#6 的检查成功，#7 的检查失败；然而分别尝试合并两者均得到 HTTP 405：`2 of 2 required status checks are expected`。未发现未审查 PR 被放行，但已审查 PR 也无法合并。**此边界列为未通过，尚不能断言具体根因；不能声称“同 head 多 PR 完全支持”。** 两条 PR 保留供继续定位。

## 尚未验收

- Infiscale / Codex 实际接入、收到开始消息、零问题完成、完成后启动新轮及新轮全部 dismissed 后的历史兜底。
- 不同人工账号的写权限变化与未授权回复。当前实际回复均为 PR 作者；需要第二个受控账号覆盖其余权限场景。
- fork PR；Issue 删除、转移、链接指向 PR；检查末次读取期间恰好发生权限/Issue 变化等剩余边界。本地单元测试不能替代这部分线上验证。

## 当前现场

- [接入 PR #2](https://github.com/liaoxiang0304/VR_DexCopilot/pull/2) 和 [正常合并对照 PR #4](https://github.com/liaoxiang0304/VR_DexCopilot/pull/4) 已合并。
- [PR #3](https://github.com/liaoxiang0304/VR_DexCopilot/pull/3) 保留用于后续真实 Bot 验收；[Issue #5](https://github.com/liaoxiang0304/VR_DexCopilot/issues/5) 已关闭。
- `AI_REVIEW_BOTS` 已从测试身份恢复为 `chatgpt-codex-connector[bot],infiscale-bot`，并通过 Issue 关闭事件重新检查开放 PR。真实 Bot 尚未完成时当前红色是预期行为。
- `AI_REVIEW_SANDBOX=true` 保留，方便继续手动 A/B 实验；fixture / lab 都只用于本测试仓库。
- [规则集](https://github.com/liaoxiang0304/VR_DexCopilot/rules/25862703) 仅约束 `test/ai-review-merge-target`，没有 bypass actor；同时要求正式 Check、实验 Check 和解决会话。`main` 未加入这些门禁。
- 只新增测试工作流、验收文档和专用分支测试文件。原 `README.md`、`flexiv_team14.py` 的 blob SHA 均与测试前一致；既有 PR #1 的内容未改动，但全仓复查会给它增加检查结果。

验收依据为具体 Check 和实际合并 API 返回，不能把预期失败的 workflow 总颜色当成验收失败，也不能用实验 job 的绿色代替正式 Check。

## 参考

- [GitHub：Ruleset 参数与检查来源](https://docs.github.com/en/rest/repos/rules#create-a-repository-ruleset)
- [GitHub：GITHUB_TOKEN 产生事件的触发限制](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)
