### Docker 设置

本仓库中的所有 example 都提供了普通运行方式和 Docker 运行方式。Docker 不是必需的，但推荐使用；它可以简化软件安装、提供更稳定的环境，并且对于依赖 ROS 的 example，也可以避免在本机安装 ROS 导致环境变得混乱。

- 基础 Docker 安装说明见[这里](https://docs.docker.com/engine/install/)。
- Docker 必须以 [rootless mode](https://docs.docker.com/engine/security/rootless/) 安装。
- 如果要使用 GPU，还必须安装 [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。
- 通过 `snap` 安装的 docker 版本与 NVIDIA container toolkit 不兼容，会导致无法访问 `libnvidia-ml.so`（见该 [issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/154)）。可以使用 `sudo snap remove docker` 卸载 snap 版本。
- Docker Desktop 也与 NVIDIA runtime 不兼容（见该 [issue](https://github.com/NVIDIA/nvidia-container-toolkit/issues/229)）。可以使用 `sudo apt remove docker-desktop` 卸载 Docker Desktop。

如果从零开始，并且 host 机器是 Ubuntu 22.04，可以使用便捷脚本 `scripts/docker/install_docker_ubuntu22.sh` 和 `scripts/docker/install_nvidia_container_toolkit.sh` 完成上述设置。

使用以下命令构建 Docker image 并启动 container：

```bash
docker compose -f scripts/docker/compose.yml up --build
```

如果要为某个特定 example 构建并运行 Docker image，请使用以下命令：

```bash
docker compose -f examples/<example_name>/compose.yml up --build
```

其中 `<example_name>` 是你想运行的 example 名称。

第一次运行任意 example 时，Docker 会构建 images。这个过程可能需要一些时间；后续运行会更快，因为 images 会被缓存。
