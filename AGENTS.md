# Repository Instructions

1. 涉及互联网或外部 HTTP(S) 的网络操作时，必须先设置以下代理环境变量：

   ```bash
   export http_proxy=http://127.0.0.1:15409 https_proxy=http://127.0.0.1:15409 HTTP_PROXY=http://127.0.0.1:15409 HTTPS_PROXY=http://127.0.0.1:15409
   ```

   Ray 相关操作是例外，包括 `ray start/status/stop`、Ray Client/GCS 通信，以及连接 Ray 集群的 NanoDeploy 进程。这些操作不走 HTTP 代理；运行前必须清除代理环境变量：

   ```bash
   unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
   ```

2. 编辑任何 C++ 文件后，如果想运行整个项目，必须运行以下命令重新安装项目：

   ```bash
   pip install -v -e .
   ```

3. 只修改 NanoDeploy 的代码，不修改任何外部依赖库的代码。

4. 如果有阶段性的思考结果或者调研结果，可以落盘到 /mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-July/docs-dev 下面，方便翻找。

5. 允许开启 Subagents.

6. LoongServe 的开源代码在 /mnt/nvme1n1/ml_research/linbinbin1/LoongServe

7. LoongServe 论文的 LaTeX 源码在 /mnt/nvme1n1/ml_research/linbinbin1/LoongServe/paper-tex-src

8. 及时提交你的修改，方便进行回溯。撰写合适的 msg。

9. 涉及 GPU 的操作都需要先申请提权

10. 每次进行上下文压缩之前，将当前的任务进度写下来罗盘到文档中（/mnt/nvme1n1/ml_research/linbinbin1/NanoDeploy-July/docs-dev以日期为文件夹，写个 Progress.md），附带上时间和任务目标概述等。压缩完成后，进行任务前先读取任务进度避免重复劳动。
