# 大模型一体机

当前研发版本：v0.1.0。目标为国产服务器硬件盘点与真实推理验证，HTML管理界面从v0.2开发。

当前服务器已通过v0.1开发环境验收：[验收报告](docs/v0.1验收报告.md)、[兼容矩阵](docs/兼容矩阵_v0.1.md)。认证范围为Qwen3-0.6B、设备0、单并发；模型能力限制在报告中单独列出。

已验证镜像：`quay.io/ascend/vllm-ascend@sha256:27405f9787f5ff16d3f10cfa56b3a567828de43bb60aeb0f025f0640fb8fdc28`。在下方命令中替换示例镜像；模型目录可使用服务器已有的 `/home/jzyh/ai-appliance/shared/models/Qwen3-0.6B`，无需重复下载。

仅使用Python 3.10+标准库。无需OpenAI API或外部商业模型账号。远程登录凭据不进入项目。

```bash
python3 -m unittest discover -s tests -v
sudo python3 -m appliance.cli doctor --output /path/to/reports/inventory.json
python3 -m appliance.cli bootstrap-download --directory /path/to/models/Qwen3-0.6B
sudo python3 -m appliance.cli start --model /path/to/models/Qwen3-0.6B --image registry/image@sha256:VERIFIED_DIGEST
python3 -m appliance.cli smoke --output /path/to/reports/smoke.json
sudo python3 -m appliance.cli stop --logs /path/to/reports/engine.log
```

执行目录应为版本目录。NPU/Docker操作在当前环境需要sudo；应用不授予普通用户Docker组权限，不修改设备权限。推理端口仅绑定127.0.0.1，用于开发验证。

下载器是v0.1的验证工具，使用来源提供的逐文件固定revision及SHA-256；不等同于v0.3包含断点续传、离线包和HTML界面的产品安装器。清单校验保证传输完整性，模型认证需要另行通过真实推理测试。

运行容器只映射指定NPU及必要管理设备，权重只读挂载，使用固定镜像摘要、保守上下文和单并发。停止工具只处理带本项目标签的固定名称容器；不会清理其他容器、镜像或模型。

v0.1只验证设备0、单实例及空闲主机上的运行路径。检测到其他NPU任务时拒绝启动，不自动停止其他任务。其他设备映射和多实例调度留待后续版本验证。`smoke --watch-container`可在模型退出后立即报错；`--reject-repetition`针对固定长文本诊断题检查异常重复，不是完整语义评测。

建议通过systemd启动完整验收，避免SSH断开中断测试。先拉取已固定摘要的镜像，再在版本目录执行：

```bash
sudo python3 scripts/launch_qualification.py \
  --image registry/image@sha256:VERIFIED_DIGEST \
  --model /path/to/models/Qwen3-0.6B \
  --reports /path/to/reports/new-run \
  --log /path/to/logs/new-run.log
```

命令返回systemd任务名、报告路径和日志路径。每次使用新的报告目录保留历史证据。测试结束会停止本次模型以验证资源释放；源码、模型和报告继续保存在服务器。执行`python3 scripts/build_release.py`生成源码包、SHA-256及逐文件清单，包中不含权重、凭据和诊断数据。

远程部署目录约定：`/home/jzyh/ai-appliance/releases/v0.1.0`；共享模型、日志及报告保存在`/home/jzyh/ai-appliance/shared`。不同版本使用独立发布目录，用户权重不随更新覆盖。
