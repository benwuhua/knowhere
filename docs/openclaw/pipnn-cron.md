# PiPNN-DiskANN OpenClaw Cron 配置（跨服务器复用）

## 目标配置

- 名称：`pipnn-builder`
- 频率：每 1 小时
- 时段：24 小时
- 超时：20 分钟
- 提交策略：自动 commit + push
- 门禁：远端 `Debug build` + `[pipnn]` 测试
- 网络：通过 `socks5` 访问远端 x86 构建机

## 一次性初始化（在运行 OpenClaw 的服务器）

```bash
cd <KNOWHERE_REPO_ROOT>

# 1) 安装技能（可选，但推荐）
scripts/remote/install-skill.sh

# 2) 准备远端构建配置
mkdir -p ~/.config/knowhere-x86-remote
cp scripts/remote/remote.env.example ~/.config/knowhere-x86-remote/remote.env

# 3) 编辑 remote.env
#    REMOTE_HOST=knowhere-x86-hk-proxy
#    WITH_PIPNN=True
#    WITH_DISKANN=True
#    WITH_UT=True

# 4) 准备 OpenClaw cron 配置
mkdir -p ~/.config/knowhere-openclaw
cp scripts/openclaw/openclaw-agent.env.example \
   ~/.config/knowhere-openclaw/openclaw-agent.env
```

## Socks5 与 SSH 别名

在 `~/.ssh/config` 中配置可复用别名（示例）：

```sshconfig
Host knowhere-x86-hk-proxy
  HostName 94.74.108.167
  User root
  Port 22
  IdentityFile ~/.ssh/agent/knowhere-x86-hk
  IdentitiesOnly yes
  ProxyCommand python3 <KNOWHERE_REPO_ROOT>/scripts/remote/socks5_proxy.py --proxy-host <SOCKS5_HOST> --proxy-port <SOCKS5_PORT> --username '<SOCKS5_USER>' --password '<SOCKS5_PASS>' %h %p
```

> 不在 prompt 中写代理账号，统一放在 SSH 配置或本机安全配置中。

## Cron 运行入口

统一使用：

```bash
<KNOWHERE_REPO_ROOT>/scripts/openclaw/run-pipnn-cron.sh
```

该脚本负责：
- 单实例锁（`flock`）防重入
- 预检 `scripts/remote/check-env.sh`
- 执行 OpenClaw 命令（来自 `openclaw-agent.env` 的 `OPENCLAW_RUN_CMD`）
- 自动抓取远端日志 `scripts/remote/fetch-logs.sh`
- 追加写入 `RESULT.md`

## 系统 Crontab 示例

```cron
0 * * * * OPENCLAW_ENV_FILE=$HOME/.config/knowhere-openclaw/openclaw-agent.env /bin/bash <KNOWHERE_REPO_ROOT>/scripts/openclaw/run-pipnn-cron.sh
```

## 关键文件

- Agent prompt：`docs/openclaw/pipnn-agent-prompt.md`
- Prompt 模板：`docs/openclaw/pipnn-agent-prompt.template.md`
- 任务队列模板：`docs/openclaw/TASK_QUEUE.md`
- 运行脚本：`scripts/openclaw/run-pipnn-cron.sh`
- OpenClaw 环境模板：`scripts/openclaw/openclaw-agent.env.example`

## 日常检查

- `RESULT.md`：`<OPENCLAW_MEMORY_DIR>/RESULT.md`
- 任务队列：`<OPENCLAW_MEMORY_DIR>/TASK_QUEUE.md`
- 本地 cron 日志：`<OPENCLAW_WORKSPACE>/logs/cron_*.log`
- 远端构建日志：执行 `scripts/remote/fetch-logs.sh` 后查看临时日志目录
