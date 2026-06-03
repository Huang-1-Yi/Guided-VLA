# Contributing to GuidedVLA

欢迎提交贡献、bug report、feature request 和文档改进。GuidedVLA 遵循仓库许可证发布；额外的第三方条款已在 README 和相关 submodules 中说明。

## Issues and feature requests

请使用 GitHub issues 提交 bug 和 feature request：

- Issues: https://github.com/GuidedVLA/GuidedVLA/issues
- Discussions: https://github.com/GuidedVLA/GuidedVLA/discussions

提交 bug 时，请包含：

- 你的 OS、Python 版本、CUDA/GPU 信息（如相关）以及安装命令
- 复现问题所需的精确命令或代码
- 完整 traceback 或错误输出
- 相关 config 名称、checkpoint 路径、数据集格式或 submodule 状态

提交 feature request 时，请包含动机、预期 workflow，以及足够的上下文，方便维护者评估实现和维护成本。

## Pull requests

打开 pull request 前：

- 确保 PR 有清晰的标题和描述。
- 使用 `pre-commit install` 安装 hooks。
- 运行 `pre-commit run --all-files`；至少应运行 `ruff check .`、`ruff format .` 和相关测试。
- 保持改动范围清晰。尽可能将机械性 cleanup、行为变更和大型 refactor 拆分为不同 PR。

对于模型、数据加载或训练相关改动，请包含用于验证的 config 和命令。对于面向用户的 workflow，请在同一个 PR 中更新相关 README 或 docs 页面。
