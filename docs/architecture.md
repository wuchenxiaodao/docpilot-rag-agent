# DocPilot 请求架构

## 核心调用链
graph LR
U["用户"] --> UI["Streamlit UI"]
UI --> C["AgentClient"]
C --> API["FastAPI Service"]
API --> R["Agent Registry"]
R --> A["LangGraph Agent"]
A --> T["Tools / RAG"]
A --> M["Model"]
T --> A
M --> A
A --> API
API --> C
C --> UI

## 各模块职责

- **Streamlit UI**：接收用户输入，并显示普通或流式响应。
- **AgentClient**：把 Python 数据转换为 HTTP 请求，并读取后端响应。
- **FastAPI Service**：提供接口、接收请求，并使用 Pydantic 校验数据。
- **Agent Registry**：根据 `agent_id` 找到对应的 Agent。
- **LangGraph Agent**：组织模型、工具和 RAG 的处理流程。
- **Tools / RAG**：查询外部工具或知识库。
- **Model**：生成回答内容。

## 流式调用
st.chat_input()
→ AgentClient.astream()
→ StreamInput
→ model_dump()
→ Python dict
→ httpx 编码 JSON 并 POST
→ FastAPI 路由
→ Pydantic 校验
→ message_generator()
→ Agent.astream()
→ StreamingResponse / SSE
→ response.aiter_lines()
→ yield parsed
→ Streamlit 显示

## 流式与非流式的区别

- `astream()`：后端生成一部分，客户端接收并显示一部分。
- `ainvoke()`：等待后端生成完整结果，再一次性返回。
- 流式响应以 `[DONE]` 作为明确的结束信号。