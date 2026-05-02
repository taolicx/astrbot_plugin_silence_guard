# AstrBot Silence Guard 技术规格

## 目标

做一个 AstrBot 插件，在消息进入正常 LLM 回复流程前判断“是否应该不回复”。插件优先用本地规则处理明确场景，只在短确认、收尾、含糊拒绝等模糊场景调用 AstrBot 已接入的模型分类，以控制成本。

## 首版范围

- 支持私聊和群聊。
- 支持闭嘴、别回、不用回复等强静默指令。
- 支持指定静默时长，例如“闭嘴 10 分钟”。
- 支持“晚安/拜拜”互相收尾后不再继续回复。
- 支持平台内模型智能判断模糊收尾消息。
- 提供 AstrBot `_conf_schema.json` 可视化配置。

## 非目标

- 不替换 AstrBot 原有聊天模型。
- 不主动生成任何回复。
- 不保存长期数据库状态，首版只使用运行期内存状态。

## 技术方案

1. 使用 `@filter.event_message_type(EventMessageType.ALL, priority=100)` 监听消息。
2. 本地规则先判断强命令、唤醒词、互道晚安、短确认等。
3. 规则返回 `UNCERTAIN` 时，调用 AstrBot 已接入的 provider 做分类判断，要求 JSON 输出。
4. 通过 `_special: select_provider` 让用户在配置里选择 provider；留空时默认使用当前正在使用的 LLM。
5. 命中 `NO_REPLY` 或 `MUTE` 时调用 `event.should_call_llm(True)` 和 `event.stop_event()`。
6. 使用 `@filter.after_message_sent()` 记录机器人最近回复，作为下次判断上下文。

## 风险与取舍

- 为避免误伤，模型不确定时默认继续回复。
- 群聊中未指向机器人的普通消息默认不主动干预。
- 运行期内存状态在 AstrBot 重启后会丢失，但这对“闭嘴几分钟”和“晚安收尾”这类短状态影响较小。
