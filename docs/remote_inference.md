# 远程运行 openpi 模型

我们提供了远程运行 openpi 模型的工具。这适用于在机器人外部、更强的 GPU 上执行推理，也有助于将机器人环境和 policy 环境分离，例如避免机器人软件依赖带来的环境冲突。

## 启动远程 policy server

要启动远程 policy server，可以直接运行以下命令：

```bash
uv run scripts/serve_policy.py --env=[DROID | ALOHA | LIBERO]
```

`env` 参数指定要加载哪个 $\pi_0$ checkpoint。在内部，该脚本会执行类似下面的命令；你也可以用这种形式为自己训练的 checkpoint 启动 policy server。下面以 DROID 环境为例：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

该命令会启动一个 policy server，为 `config` 和 `dir` 参数指定的 policy 提供服务。policy 会运行在指定端口上，默认端口为 8000。

## 从机器人代码查询远程 policy server

我们提供了一个依赖很少的客户端工具，可以很容易嵌入到任意机器人代码库中。

首先，在机器人环境中安装 `openpi-client` 包：

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

然后，可以在机器人代码中使用该客户端查询远程 policy server。下面是一个示例：

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]

    # Execute the actions in the environment.
    ...

```

这里的 `host` 和 `port` 参数指定远程 policy server 的 IP 地址和端口。你也可以将它们作为命令行参数传给机器人代码，或者直接写入机器人代码库。`observation` 是包含观测和 prompt 的字典，遵循当前所服务 policy 的输入规范。关于如何为不同环境构造该字典，可以参考 [simple client example](examples/simple_client/main.py) 中的具体示例。
