# sandrone QQ 聊天机器人

面向小型 VPS 的轻量 QQ 机器人：通过腾讯官方 `qqbot-agent-sdk` 接收好友/群聊消息，调用 GPT-5.6 Luna，并使用 SQLite 持久保存聊天记录和明确的长期记忆。回复、固有角色记忆与滚动摘要采用《原神》桑多涅人设。

## 功能

- 只处理 `ALLOWED_GROUP_IDS` 指定的群；当前部署仅服务一个沙箱群
- 全量群消息进入记忆；@ sandrone 时正常回复，并有两种克制的主动发言
- 北京时间 06:00–10:00，每位成员当天首次发言会收到结合公开印象的短早安；10:00 后
  当天首次出现则改成普通个性化问候，绝不再说“早安”
- 每累计 20 条群消息评估一次是否适合自然插话；成功主动发言后有 20 分钟硬冷却，
  有定向 @、引用、图片上下文或事实指代不清时保持安静，候选句还会经过一次一致性复核
- 约 30% 的普通聊天回复确定性携带一个 Unicode Emoji；模型未主动使用时由发送前策略按语境补一个，其余回复仍明确抑制，避免比例失控
- 接收贴在 Sandrone 消息上的 QQ 表情并写入持久互动反馈：近期表情会轻微影响她下一次自然回复的情绪，长期累计只作为很轻的关系线索；撤销表情会撤销当前反馈，不会因贴表情立即刷屏。能否收到仍取决于平台实际下发
- GPT-5.6 Luna Responses API；也可切换 OpenAI 兼容的 Chat Completions
- 按需网页检索：支持 `/搜索`、明确检索请求和明显时效问题，回答附来源
- 网页结果按群缓存 30 分钟并限频；外部资料带时间进入短期上下文，不覆盖角色固有记忆
- SQLite WAL 原始时间线 + 自动滚动摘要 + 显式长期记忆，服务重启后仍保留
- 每位群成员按平台 `openid` 建立独立、可修正的双层印象卡；默认每 8 条发言更新一次。长期层最多 600 字，不因单条消息轻易改变但可被持续反证修正；短期层最多 400 字，只保留最近一两天的动态互动
- 群成员排名会读取保留区与印象卡中的完整发言名单，低发言量成员也必须逐一列出
- 群聊图片先由 VPS 限时下载并校验，再以内嵌 Data URL 交给视觉模型，避免上游拉取
  QQ 临时 CDN 地址超时；下载单图上限 8 MiB，最多 4 张。供后续编辑使用的原图有独立限额缓存
- 群聊识图采用作品中立、证据优先策略，不会因桑多涅人设而把普通图片强猜成《原神》角色；
  只有独特特征或文字上下文明显吻合时才给出具体身份，不确定时保留候选与置信度
- `/画图 描述` 或自然语言画图请求会按立绘、头像或场景选择竖版、方形或横版尺寸
- 生图前结合最近对话消解“你/她/他”等指代，发送前视觉复核成图并写入记忆
- 画面包含桑多涅时自动携带两张固定身份参考图；身份复核失败会校准重画一次，仍失败则不发送
- 原图编辑：引用图片说“把图中的原石数量改成 65432”等会上传真实原图到 `/images/edits`，不经过新图构图扩写、不混入固定桑多涅参考。编辑使用 high 质量、PNG 输出；保留原图尺寸和比例，禁止中心裁剪，不保证未修改处像素完全一致。
- 原图选择优先级：引用附件 > 本条附件 > 同一群、同一发言者最近 10 分钟的单张原图或成图。多图用途不明、引用缺少原图或链接失效时追问，不退回文字生图。带明确多图参考要求时最多 4 张，第一张为底图。
- 编辑结果与原图视觉对照，数字/文字修改逐字核对；失败最多从原图重试一次，仍失败或校验不可用就不发送。三分钟冷却、取消、单任务锁与生图共用。
- 原图缓存位于生成目录下 `sources/`，24 小时过期（访问/启动时清理）、最多 128 张、96 MiB；索引不存签名 URL，按群和发言者隔离，源图去除 EXIF。实际编辑期间固定副本，避免缓存淘汰影响任务。缓存失败不影响已收到消息的记录或正常回复。
- 图片全局冷却默认 180 秒，冷却拒绝和错误提示保持桑多涅口吻
- 生图任务阶段持久化到 SQLite；服务重启会明确标记中断，进度回复不依赖模型猜测
- 生图请求者和最高指挥可用“取消/别画了”真正中止尚未发送的任务，其他成员不能越权取消
- 同一时间只运行一张图，避免慢任务越过冷却后与下一张并发串线
- 最高指挥使用经过 QQ 事件身份绑定的权限标记，不能靠聊天文本冒充
- 共享群记忆只能由最高指挥使用 `/新对话` 清空
- 每个会话串行处理，避免并发回复打乱上下文
- 消息 ID 去重，Gateway 重连时避免重复扣费与重复回复
- `/新对话`、`/记住`、`/记忆`、`/忘记` 等隐私可控命令
- systemd 自动重启与 300 MB 内存上限，不依赖 Docker、Redis 或向量数据库
- 上传签名 URL 不写入 INFO 日志；`.env`、数据库和 WAL 默认使用私有权限

## 本地运行

需要 Python 3.10–3.13。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
# 编辑 .env，填写 QQ_APP_ID、QQ_APP_SECRET、OPENAI_API_KEY
.\.venv\Scripts\python -m talk_bot.main
```

运行测试：

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m compileall -q src tests
```

如本机已安装 `ruff`，可额外执行 `ruff check .`，它不是运行或测试所必需的依赖。

## 配置说明

- `OPENAI_BASE_URL` 必须包含 API 根路径，官方服务为 `https://api.openai.com/v1`。
- `OPENAI_API_MODE=responses` 是默认值。若中转只实现 Chat Completions，可改成 `chat_completions`。
- `HISTORY_MESSAGES` 是每次发给模型的最近消息数，默认 30；数据库会保留最多 4000 条，且绝不因容量上限删除尚未进入摘要的消息。滚动摘要最多 2400 个中文字符；显式长期事实单条最多 1000 字、默认读取最近 60 条。
- 普通短问句、图片追问和承接问句保留配置内的完整近期上下文及滚动记忆，不再按“20 字以内”裁成当前一句，也不再将承接问句限制为 8 条。仅明确的自我评价、纯表情和简短情绪采用窄上下文；归属校验共享主回复实际获得的历史、事实与印象证据，不能因自己缺少上下文否认记忆。
- `SUMMARY_TRIGGER_MESSAGES` 控制累计多少条新消息后更新一次滚动摘要，默认 20。
- `SUMMARY_BATCH_MESSAGES` 控制每次摘要最多追赶多少条消息，默认 40；不得小于触发条数。
- `PROFILE_TRIGGER_MESSAGES` 控制每位成员累计多少条新发言后更新印象，默认 8。
- `WEB_SEARCH_ENABLED` 控制网页检索；默认开启，仅在 Responses API 模式且中转支持
  `web_search` 工具时可用。
- `WEB_SEARCH_CACHE_SECONDS` 默认 1800；普通成员检索冷却默认 30 秒，全群默认 10 秒。
- `IMAGE_MODEL` 默认为 `gpt-image-2`；`IMAGE_COOLDOWN_SECONDS` 默认为 180。
- `SANDRONE_REFERENCE_IMAGES` 是逗号分隔的桑多涅参考图；默认使用项目内的脸部与服装参考。
- `BOT_OWNER_IDS` 是最高指挥身份列表，应同时配置 QQ 号与平台实际提供的 `openid`。
- `ALLOWED_GROUP_IDS` 限制唯一服务群，应同时配置群号与平台实际提供的群 `openid`。
- 群聊上下文按群共享，长期记忆按当前发言者隔离；私聊会话和群聊记忆不会混用。
- 当前 QQ 适配只解析图片，不能直接观看群视频或收听语音；应改发关键截图、字幕或转写。
- 全量群消息还要求群主在手机 QQ 的机器人设置中授予“获取群内全部消息”；否则平台只会推送 @ 消息。
- `.env` 和 `data/` 已被 Git 忽略。不要把 AppSecret 或 API Key 提交到仓库。

## VPS 部署

推荐路径 `/opt/sandrone-bot`，使用独立低权限用户运行：

```bash
sudo useradd --system --home /opt/sandrone-bot --shell /usr/sbin/nologin sandrone
sudo mkdir -p /opt/sandrone-bot/data
sudo chown -R sandrone:sandrone /opt/sandrone-bot
sudo -u sandrone python3 -m venv /opt/sandrone-bot/.venv
sudo -u sandrone /opt/sandrone-bot/.venv/bin/pip install /opt/sandrone-bot
sudo cp /opt/sandrone-bot/deploy/sandrone-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now sandrone-bot
sudo systemctl status sandrone-bot
sudo journalctl -u sandrone-bot -f
```

服务文件仅允许 `/opt/sandrone-bot/data` 写入，并设置 `MemoryMax=300M`。`.env` 建议权限设为 `600`。

若没有免密 sudo，可部署到 `~/services/sandrone-bot` 并使用
`deploy/sandrone-bot.user.service`。用户服务要跨 SSH 注销和重启持续运行，还需管理员执行一次
`sudo loginctl enable-linger <用户名>`；不能启用 linger 时，不应把“当前运行”误当作持续服务已经完成。


## 展示版说明

本仓库为脱敏源码快照，包含现有自动化测试；不附带生产配置、数据库、聊天记录、部署凭据或第三方角色参考图片。请自行配置平台凭据。测试使用模拟接口，不表示已完成本轮 QQ 平台联调。
图像功能的角色参考图需要自行提供获授权图片，并配置 `SANDRONE_REFERENCE_IMAGES`。文本聊天不需要这些图片。
