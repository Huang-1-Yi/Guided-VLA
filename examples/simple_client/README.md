# Simple Client

这是一个最小客户端示例，用于向服务器发送 observation，并打印推理速率。

可以通过 `--env` 参数指定要使用的运行环境。运行以下命令可以查看可用选项：

```bash
uv run examples/simple_client/main.py --help
```

## 使用 Docker

```bash
export SERVER_ARGS="--env ALOHA_SIM"
docker compose -f examples/simple_client/compose.yml up --build
```

## 不使用 Docker

终端窗口 1：

```bash
uv run examples/simple_client/main.py --env DROID
```

终端窗口 2：

```bash
uv run scripts/serve_policy.py --env DROID
```
