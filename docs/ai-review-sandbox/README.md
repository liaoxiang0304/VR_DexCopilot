# AI Review 门禁：隔离仓库验收包

本包只在专用测试仓库中使用。已于 2026-10-26 完成部分真实 API / 门禁验证，真实 AI 验收仍待接入。

本 caller 已额外修正手动 PR 编号的数字转换；该修正尚未同步公共候选。实测记录见测试仓库 docs/ai-review-sandbox/RESULTS.md。共享 head 多 PR 尚未通过，不应据此直接推广。

待测版本：[`be25a22a1c39b9c64fe8931e858e29829be1ada5`](https://github.com/IS-Model-Framework/.github/commit/be25a22a1c39b9c64fe8931e858e29829be1ada5)。

两个 caller job 的 workflow 引用和 `shared-ref` 均已改为这个提交。因此无需先合入公共 CI 仓库的 main；生产环境仍可按原方案使用 main。若候选代码修复后产生新提交，需要同步改这些引用并重新验收。

## 1. 准备专用测试仓库

1. 选择一个你能配置 Actions、Bot 访问和仓库规则的专用仓库。使用无业务价值的测试代码和测试 Issue。
2. 将本包的 `.github/workflows/` 整个目录复制到测试仓库，并提交到其默认分支。不要只放在实验分支：issue_comment、issues、workflow_run、workflow_dispatch 都依赖默认分支存在对应 workflow。
3. 设置仓库 Variable `AI_REVIEW_BOTS`，值为 `chatgpt-codex-connector[bot],infiscale-bot`。还需确保两个真实 Bot 能访问测试仓库并实际运行；这个变量本身不会安装 Bot。
4. 设置仓库 Variable `AI_REVIEW_SANDBOX=true`，用于启用下面的手动 A/B 实验。生产 caller 不读取这个变量。
5. 新建一个包含少量代码改动的测试 PR。先不增加 Required Status Check，观察触发、日志和具体 Check 输出是否正确。

`ai-review-disposition.yml` 和 `ai-review-events.yml` 沿用候选版本的接入示例，额外固定了测试引用，并修正手动编号转换。`ai-review-latest-result-lab.yml` 是额外的测试夹具，只能放测试仓库；它不改生产脚本，也不伪造真实 Bot 身份。

## 2. 在测试仓库启用真正的合并门禁

在一个健康测试 PR 上，让真实 Bot 完成，并补齐有效模板，使 `AI Review / Disposition Check` 至少成功一次。

然后只在测试仓库创建 Active ruleset，目标为其测试用默认分支：

- Require a pull request before merging。
- Required Status Check：`AI Review / Disposition Check`，来源选择 GitHub Actions。
- Require conversation resolution before merging。

不要选择 Actions job 的成功状态。人工 approval 或其他已有条件也会影响合并；验收每一项时，分别观察门禁原因。测试“能否合并”时使用没有绕过权限的账号，并确保其他合并条件已满足。

## 3. 真实操作验收清单

每项记录 PR 链接、最新 head SHA、Actions run 链接、具体 Check 的结论和合并区显示。没有实际运行的项目不要标为通过。

| 编号 | 实际操作 | 应观察到的结果 |
| --- | --- | --- |
| 1 | 创建 PR，AI 尚未完成 | Check 不通过，不能因为线程数量为零就提前通过 |
| 2 | 已通过后请求真实 AI 开始新一轮 | 新开始消息后阻断；该轮完成后重新检查，不沿用旧轮提前放行 |
| 3 | 真实 Bot 有意见但缺模板；再新增、编辑、删除有效模板 | 缺模板失败；有效模板通过；改坏或删掉唯一有效模板后失败；普通聊天不重置审查轮次 |
| 4 | 仅 resolve 线程，不写模板；再写有效模板但 unresolve | 前者 Check 失败；后者 Check 可通过，但原生 conversation resolution 阻止合并。Resolve/Unresolve 不需要重跑本 Check |
| 5 | 线程 outdated 或原意见被编辑 | outdated 仍需模板；原意见编辑和普通后续讨论不要求额外重复确认 |
| 6 | 用本仓库开放 Issue 回复 Deferred，再从 GitHub UI 关闭 Issue | 自动出现全仓复查；仍引用这个已关闭 Issue 的 Deferred 失败。另一个开放 PR 也应被检查，不能在第一个失败后停止 |
| 7 | 将上项回复改为 Rejected 并写理由，或改为 Accepted 并引用当前 PR 的修复 commit | 自动复查后恢复。跨仓库 Issue、实际为 PR 的链接、已删除 Issue 均不能通过 |
| 8 | Accepted 引用修复 commit，再 force-push 将该提交移出 PR | 最新 head 上的 Check 失败；不能误用旧 head 的成功结果 |
| 9 | 修复后正常 push，且 Bot 没有实际开始新一轮 | 只复查处理结果，无需强制 AI 重审。若 Bot 自己开始新一轮，则等待完成是正确行为 |
| 10 | 非作者、无写权限账号回复模板 | 不可替代授权回复；PR 作者或仍有写权限的人工回复者可处理 |

零问题完成、真实 Bot 的开始/完成格式必须至少各验证一次。不要由人工复制 Bot 文本来代替这项测试：检查会匹配评论作者。

对于 dismissed，可先在测试仓库用一个受控的第二账号作为 `AI_REVIEW_BOTS`，提交可撤销的正式 Review，测试：无有效历史依据时撤销后失败；保留更早有效 Review 时允许兜底。真实 Bot 的 COMMENTED Review 若没有可用的 dismiss 操作，不应硬套这个 UI 场景。改回真实 Bot 配置后，使用 Run workflow 发起新检查。

再补两项独立性检查：不同 head 的两个 PR 不应串用结果；如果团队会对同一个 head 开多个 PR，也要测试“一条缺回复、另一条已通过”，不能让成功结果错误放行缺回复的 PR。团队接收 fork PR 时，还需用真实 fork 走一遍 collector → workflow_run 链路。

用 GitHub UI 或真实 Bot / App 执行评论、Review、关闭 Issue 等动作。用 GITHUB_TOKEN 自动制造这些事件通常不会启动后续 workflow，容易把事件抑制误判成脚本故障。

## 4. 用真实 Checks API 验证“只认 B”

自然触发两次 CI 能测试取消旧任务，但不容易稳定制造迟到结果。本包另有手动实验，在 GitHub Actions 中使用真实 GITHUB_TOKEN 和待测脚本的 `current_check()`，人为制造 A 在 B 结束后才写结果。

实验只使用独立名字 `AI Review / Disposition Integration Test`，不覆盖真实 evaluator 的 Check。它测试实际 Checks API、旧记录改名和最新结果；不代替上面的真实 Bot、事件和权限验收。

1. 选择一个实际 `AI Review / Disposition Check` 已通过、其他合并条件也满足的测试 PR。
2. Actions → **AI Review Latest Result Lab** → Run workflow，填写 PR 编号，先运行 **B-success**。第一次用于创建成功的实验 Check。
3. 在测试 ruleset 中，另外添加 `AI Review / Disposition Integration Test` 为 Required Status Check，来源同样选 GitHub Actions。
4. 用新的 Run workflow 依次运行下表场景。可将 `pause-seconds` 设为 30，以便在 B 仍处理中时查看合并区。

| 场景 | 程序故意制造的顺序 | 具体实验 Check / 合并区的预期 |
| --- | --- | --- |
| B-failure | B 失败，然后 A 迟到写成功 | 保持 B 的失败，不能合并 |
| B-cancelled | B 取消，然后 A 迟到写成功 | 保持 B 的取消，不能回退到 A 放行 |
| B-success | B 成功，然后 A 迟到写失败 | 保持 B 的成功，A 的历史记录不能阻塞合并 |
| 暂停期间 | A 曾成功，B 刚接管但仍处理中 | 不能沿用 A 的成功合并 |

实验 job 变绿表示“断言符合预期”；它创建的实验 Check 则按所选 B 场景变红、绿或取消。B-failure 时“实验 job 绿、实验 Check 红”正是预期，不能只看 job 颜色。

程序会在 A 迟到写入后、再次处理 A 之前，检查固定名称下只剩 B 且结论未变。仍需人工核对真实 merge box，确认 ruleset 正在阻止或允许合并；仅 API 断言通过不能替代这一项。

## 5. 进入正式仓库前

隔离仓库中的主要场景通过并留下证据后，再选择一个真实仓库试运行。先确认其现有规则未要求这个 Check；观察正常 PR 的触发、结果和故障恢复，再将它加入 Required Status Checks。公共 CI 的 main 可以在隔离验收后再合入。

正式仓库只复制实际 caller 和 collector，不复制 lab workflow。此前测试结论只对记录的候选提交成立；候选代码有实质变化时，重跑受影响场景。

## 参考

- [GitHub：复用 workflow 可以引用提交 SHA](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows)
- [GitHub：事件与默认分支要求](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)
- [GitHub：GITHUB_TOKEN 触发事件的限制](https://docs.github.com/en/actions/concepts/security/github_token)
- [GitHub：创建仓库 ruleset](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/creating-rulesets-for-a-repository)
