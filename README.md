# AstrBot Silence Guard

一个 AstrBot 回复守门插件：在消息进入正常 LLM 回复前，先判断“这句话是不是不该继续回”。明确场景直接用规则拦截，模糊场景才调用 AstrBot 已接入的模型做分类。

## 能做什么

- 用户说“闭嘴”“别回我”“不用回复”时，不发确认，直接静默。
- 用户说“闭嘴 10 分钟”时，在指定时间内停止回复。
- 机器人刚说“晚安/拜拜”，用户也回“晚安/拜拜”时，不再继续礼貌循环。
- 最近连续对话里，用户只回“嗯嗯/好吧/算了/不用了”这类模糊收尾时，可调用平台内模型判断是否不回复。
- 群聊未指向机器人的普通消息默认不主动干预。

## 安装

把这个仓库放进 AstrBot 的插件目录后，确保最终插件目录名是 `astrbot_plugin_silence_guard`，然后在 AstrBot 管理面板启用插件。根目录下的 `main.py` 会作为加载入口。

## 推荐配置

```json
{
  "smart_mode": true,
  "judge_mode": "ambiguous_only",
  "judge_provider_id": "",
  "judge_timeout_seconds": 3,
  "judge_max_tokens": 96,
  "judge_context_turns": 6,
  "judge_min_confidence": 0.72,
  "fallback_when_uncertain": "reply"
}
```

`judge_provider_id` 留空时，会默认使用 AstrBot 当前正在使用的 LLM 提供商。

## 规则优先

插件不会每条消息都请求模型。流程是：

```text
明确静默/恢复/收尾规则 -> 直接处理
短确认、算了、不用了等模糊场景 -> 调用平台内模型
模型不确定或失败 -> 默认继续回复
```

## 例子

```text
用户：闭嘴
机器人：不回复，并进入默认 10 分钟静默

用户：可以说话了
机器人：恢复正常流程

机器人：晚安，好梦
用户：晚安
机器人：不回复

机器人：这个问题可以这样处理……
用户：嗯嗯
插件：规则认为模糊，调用平台内模型判断是否只是收尾确认
```

## 文件说明

- `main.py`：AstrBot 插件入口，负责监听事件、停止事件、记录机器人回复。
- `silence_logic.py`：纯规则、状态和时长解析。
- `deepseek_judge.py`：基于 AstrBot Provider 的分类判断器。
- `_conf_schema.json`：AstrBot 管理面板配置项。

## 安全说明

插件不会在日志中输出敏感密钥。开启 `debug_log` 后只记录判断结果、原因和会话标识。
