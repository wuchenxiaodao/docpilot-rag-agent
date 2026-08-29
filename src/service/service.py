import asyncio
import hashlib
import inspect
import json
import logging
import os
import time
import warnings
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated, Any
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from core.auth import Principal, load_api_keys, resolve_principal
from core.ratelimit import rate_limiter
from langchain_core._api import LangChainBetaWarning
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langfuse import Langfuse  # type: ignore[import-untyped]
from langfuse.langchain import (
    CallbackHandler,  # type: ignore[import-untyped]
)
from langgraph.types import Command, Interrupt
from langsmith import Client as LangsmithClient
from langsmith import uuid7

from agents import DEFAULT_AGENT, AgentGraph, get_agent, get_all_agent_info, load_agent
from agents.ingestion import MAX_UPLOAD_BYTES, SUPPORTED_EXTENSIONS, ingest_file
from agents.tools import get_chroma_store
from cache import (
    acquire_lock,
    invalidate_all,
    lookup,
    redis_token_bucket,
    release_lock,
    store,
    store_embedding,
)
from core import settings
from core.settings import DatabaseType
from db import (
    DocumentStatus,
    MessageRole,
    append_message,
    create_session,
    get_session_by_id,
    init_tables,
    list_documents,
    register_document,
    session_scope,
    transition_status,
)
from memory import initialize_database, initialize_store
from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    DocumentListResponse,
    DocumentOut,
    Feedback,
    FeedbackResponse,
    IngestResponse,
    ServiceMetadata,
    StreamInput,
    ThreadInfo,
    UserInput,
)
from service.agui import router as agui_router
from service.utils import (
    convert_message_content_to_string,
    langchain_to_chat_message,
    remove_tool_calls,
)

warnings.filterwarnings("ignore", category=LangChainBetaWarning)
logger = logging.getLogger(__name__)


def custom_generate_unique_id(route: APIRoute) -> str:
    """Generate idiomatic operation IDs for OpenAPI client generation."""
    return route.name


# Per-user API keys are read from a (gitignored) JSON file or inline env var and
# cached so a request never re-reads the file. The cache key includes the file
# mtime, so editing api_keys.json is picked up without a restart.
_API_KEY_CACHE: tuple[tuple[str | None, float, str | None], dict[str, Principal]] | None = None


def _get_api_keys() -> dict[str, Principal]:
    """Load and cache the per-user API-key store from settings."""
    global _API_KEY_CACHE
    path = settings.AUTH_API_KEYS_FILE
    json_str = settings.AUTH_API_KEYS_JSON
    mtime: float | None = None
    if path:
        try:
            mtime = float(os.path.getmtime(path))
        except OSError:
            mtime = -1.0
    cache_key = (path, mtime if mtime is not None else 0.0, json_str)
    if _API_KEY_CACHE is not None and _API_KEY_CACHE[0] == cache_key:
        return _API_KEY_CACHE[1]
    store = load_api_keys(path, json_str)
    _API_KEY_CACHE = (cache_key, store)
    return store


def verify_bearer(
    request: Request,
    http_auth: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(HTTPBearer(description="Please provide AUTH_SECRET or API key.", auto_error=False)),
    ],
) -> None:
    """Resolve the caller to a Principal and stash it on the request.

    Backward compatible: with neither AUTH_SECRET nor API keys configured every
    request is anonymous (no 401). With AUTH_SECRET only, the shared secret
    authenticates the request but does not pin a user id (clients keep
    supplying their own, as before). With API keys, the key identifies a user
    whose id the server pins downstream.
    """
    auth_secret = settings.AUTH_SECRET.get_secret_value() if settings.AUTH_SECRET else None
    api_keys = _get_api_keys()
    token = http_auth.credentials if http_auth else None
    principal = resolve_principal(token, api_keys, auth_secret)
    if principal is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    request.state.principal = principal


def _rate_key(principal: Principal | None, request: Request) -> str:
    """Bucket key for rate limiting: per-user id when identified, else client IP."""
    if principal is not None and principal.source == "api_key":
        return f"u:{principal.user_id}"
    # X-Forwarded-For left to a reverse proxy; behind bare uvicorn use the peer.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


def rate_limit(
    request: Request,
) -> None:
    """Per-caller limiter. No-op when RATE_LIMIT_PER_MIN is 0.

    Redis 令牌桶优先（多进程共享一个桶）；Redis 不可用时降级到进程内
    固定窗口——限流是防御性功能，宁可降级放行也不能让接口 500。
    """
    limit = int(settings.RATE_LIMIT_PER_MIN or 0)
    if limit <= 0:
        return
    principal = getattr(request.state, "principal", None)
    key = _rate_key(principal, request)
    try:
        allowed, retry_after = redis_token_bucket.check(key, limit, window_sec=60.0)
    except Exception as e:
        logger.warning(f"Redis rate limit degraded to in-memory: {e}")
        allowed, retry_after = rate_limiter.check(key, limit, window_sec=60.0)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again shortly.",
            headers={"Retry-After": str(retry_after)},
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Configurable lifespan that initializes the appropriate database checkpointer, store,
    and agents with async loading - for example for starting up MCP clients.
    """
    try:
        # 业务表（MySQL）：秒传登记、状态机、对话记录。启动时建表幂等。
        try:
            init_tables()
        except Exception as e:
            # MySQL 不在线时服务仍可启动（文档入库功能降级），不阻断 agent 主链路
            logger.warning(f"MySQL business tables unavailable, /ingest metadata degraded: {e}")
        # Initialize both checkpointer (for short-term memory) and store (for long-term memory)
        async with initialize_database() as saver, initialize_store() as store:
            # Set up both components
            if hasattr(saver, "setup"):  # ignore: union-attr
                await saver.setup()
            # Only setup store for Postgres as InMemoryStore doesn't need setup
            if hasattr(store, "setup"):  # ignore: union-attr
                await store.setup()

            if not settings.AUTH_SECRET and not _get_api_keys():
                logger.warning(
                    "AUTH_SECRET is not configured and no API keys are loaded — all API "
                    "endpoints are unauthenticated. Set AUTH_SECRET (or AUTH_API_KEYS) in "
                    "your environment to enable authentication."
                )

            # Configure agents with both memory components and async loading
            agents = get_all_agent_info()
            for a in agents:
                try:
                    await load_agent(a.key)
                    logger.info(f"Agent loaded: {a.key}")
                except Exception as e:
                    logger.error(f"Failed to load agent {a.key}: {e}")
                    # Continue with other agents rather than failing startup

                agent = get_agent(a.key)
                # Set checkpointer for thread-scoped memory (conversation history)
                agent.checkpointer = saver
                # Set store for long-term memory (cross-conversation knowledge)
                agent.store = store
            yield
    except Exception as e:
        logger.error(f"Error during database/store/agents initialization: {e}")
        raise


app = FastAPI(lifespan=lifespan, generate_unique_id_function=custom_generate_unique_id)
router = APIRouter(dependencies=[Depends(verify_bearer)])
# AG-UI protocol endpoints inherit the same bearer auth - see service/agui.py
router.include_router(agui_router)


@router.get("/info")
async def info() -> ServiceMetadata:
    models = list(settings.AVAILABLE_MODELS)
    models.sort()
    return ServiceMetadata(
        agents=get_all_agent_info(),
        models=models,
        default_agent=DEFAULT_AGENT,
        default_model=settings.DEFAULT_MODEL,
    )


async def _handle_input(
    user_input: UserInput,
    agent: AgentGraph,
    principal: Principal | None = None,
) -> tuple[dict[str, Any], UUID]:
    """
    Parse user input and handle any required interrupt resumption.
    Returns kwargs for agent invocation and the run_id.

    When the caller authenticated with a per-user API key, the server pins
    ``user_id`` to that principal's id so a client cannot impersonate another
    user. The shared-secret and anonymous modes keep honoring the client's
    ``user_id`` (prior behavior), since the secret does not identify a user.
    """
    run_id = uuid7()
    thread_id = user_input.thread_id or str(uuid4())
    if principal is not None and principal.is_identified:
        user_id = principal.user_id
    else:
        user_id = user_input.user_id or str(uuid4())

    configurable = {"thread_id": thread_id, "user_id": user_id}
    if user_input.model is not None:
        configurable["model"] = user_input.model

    callbacks: list[Any] = []
    if settings.LANGFUSE_TRACING:
        # Initialize Langfuse CallbackHandler for Langchain (tracing)
        langfuse_handler = CallbackHandler()

        callbacks.append(langfuse_handler)

    if user_input.agent_config:
        # Check for reserved keys (including 'model' even if not in configurable)
        reserved_keys = {"thread_id", "user_id", "model"}
        if overlap := reserved_keys & user_input.agent_config.keys():
            raise HTTPException(
                status_code=422,
                detail=f"agent_config contains reserved keys: {overlap}",
            )
        configurable.update(user_input.agent_config)

    config = RunnableConfig(
        configurable=configurable,
        run_id=run_id,
        callbacks=callbacks,
    )

    # Check for interrupts that need to be resumed
    state = await agent.aget_state(config=config)
    interrupted_tasks = [
        task for task in state.tasks if hasattr(task, "interrupts") and task.interrupts
    ]

    input: Command | dict[str, Any]
    if interrupted_tasks:
        # assume user input is response to resume agent execution from interrupt
        input = Command(resume=user_input.message)
    else:
        input = {"messages": [HumanMessage(content=user_input.message)]}

    kwargs = {
        "input": input,
        "config": config,
    }

    return kwargs, run_id


@router.post(
    "/{agent_id}/invoke",
    operation_id="invoke_with_agent_id",
    dependencies=[Depends(rate_limit)],
)
@router.post("/invoke", dependencies=[Depends(rate_limit)])
async def invoke(
    request: Request,
    user_input: UserInput,
    agent_id: str = DEFAULT_AGENT,
) -> ChatMessage:
    """
    Invoke an agent with user input to retrieve a final response.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.
    When authenticated with a per-user API key, the server overrides user_id
    with the caller's identity.
    """
    # NOTE: Currently this only returns the last message or interrupt.
    # In the case of an agent outputting multiple AIMessages (such as the background step
    # in interrupt-agent, or a tool step in research-assistant), it's omitted. Arguably,
    # you'd want to include it. You could update the API to return a list of ChatMessages
    # in that case.
    agent: AgentGraph = get_agent(agent_id)
    principal = getattr(request.state, "principal", None)
    kwargs, run_id = await _handle_input(user_input, agent, principal)

    try:
        response_events: list[tuple[str, Any]] = await agent.ainvoke(**kwargs, stream_mode=["updates", "values"])  # type: ignore # fmt: skip
        response_type, response = response_events[-1]
        if response_type == "values":
            # Normal response, the agent completed successfully
            output = langchain_to_chat_message(response["messages"][-1])
        elif response_type == "updates" and "__interrupt__" in response:
            # The last thing to occur was an interrupt
            # Return the value of the first interrupt as an AIMessage
            output = langchain_to_chat_message(
                AIMessage(content=response["__interrupt__"][0].value)
            )
        else:
            raise ValueError(f"Unexpected response type: {response_type}")

        output.run_id = str(run_id)
        return output
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


async def message_generator(
    user_input: StreamInput,
    agent_id: str = DEFAULT_AGENT,
    principal: Principal | None = None,
) -> AsyncGenerator[str, None]:
    """
    Generate a stream of messages from the agent.

    This is the workhorse method for the /stream endpoint.
    """
    agent: AgentGraph = get_agent(agent_id)
    kwargs, run_id = await _handle_input(user_input, agent, principal)
    # 对话记录入库（MySQL）用：thread_id 即 chat_sessions.id
    thread_id = kwargs["config"]["configurable"]["thread_id"]
    turn_started = time.perf_counter()
    citations: list[dict] | None = None
    final_answer: str | None = None

    # QA 缓存：rag-assistant 的答案只依赖知识库内容，缓存是安全的。
    # 其他 agent（chatbot 等）答案依赖会话上下文，不缓存。
    if agent_id == "rag-assistant":
        try:
            cached = await asyncio.to_thread(
                lookup, user_input.message, _embed_question
            )
        except Exception as e:
            logger.warning(f"QA cache lookup skipped (Redis unavailable?): {e}")
            cached = None
        if cached is not None:
            logger.info("QA cache hit (%s) for thread %s", cached.get("hit"), thread_id)
            hit_message = ChatMessage(type="ai", content=cached["answer"])
            yield f"data: {json.dumps({'type': 'message', 'content': hit_message.model_dump()})}\n\n"
            yield "data: [DONE]\n\n"
            try:
                await _persist_chat_turn_async(
                    thread_id, user_input.message, cached["answer"],
                    cached.get("citations"), time.perf_counter() - turn_started,
                )
            except Exception as e:
                logger.warning(f"Chat persistence skipped (MySQL unavailable?): {e}")
            return

    try:
        # Process streamed events from the graph and yield messages over the SSE stream.
        async for stream_event in agent.astream(
            **kwargs, stream_mode=["updates", "messages", "custom"], subgraphs=True
        ):
            if not isinstance(stream_event, tuple):
                continue
            # Handle different stream event structures based on subgraphs
            if len(stream_event) == 3:
                # With subgraphs=True: (node_path, stream_mode, event)
                _, stream_mode, event = stream_event
            else:
                # Without subgraphs: (stream_mode, event)
                stream_mode, event = stream_event
            new_messages = []
            if stream_mode == "updates":
                for node, updates in event.items():
                    # A simple approach to handle agent interrupts.
                    # In a more sophisticated implementation, we could add
                    # some structured ChatMessage type to return the interrupt value.
                    if node == "__interrupt__":
                        interrupt: Interrupt
                        for interrupt in updates:
                            new_messages.append(AIMessage(content=interrupt.value))
                        continue
                    updates = updates or {}
                    update_messages = updates.get("messages", [])
                    # special cases for using langgraph-supervisor library
                    if "supervisor" in node or "sub-agent" in node:
                        # the only tools that come from the actual agent are the handoff and handback tools
                        if isinstance(update_messages[-1], ToolMessage):
                            if "sub-agent" in node and len(update_messages) > 1:
                                # If this is a sub-agent, we want to keep the last 2 messages - the handback tool, and it's result
                                update_messages = update_messages[-2:]
                            else:
                                # If this is a supervisor, we want to keep the last message only - the handoff result. The tool comes from the 'agent' node.
                                update_messages = [update_messages[-1]]
                        else:
                            update_messages = []
                    new_messages.extend(update_messages)

            if stream_mode == "custom":
                new_messages = [event]
                # DocPilot 引用链：CustomData.dispatch 后是 role='custom' 的
                # ChatMessage，content 为 [ {docpilot_citations: [...] } ]
                custom_content = getattr(event, "content", None)
                if (
                    isinstance(custom_content, list)
                    and custom_content
                    and isinstance(custom_content[0], dict)
                    and "docpilot_citations" in custom_content[0]
                ):
                    citations = custom_content[0]["docpilot_citations"]

            # LangGraph streaming may emit tuples: (field_name, field_value)
            # e.g. ('content', <str>), ('tool_calls', [ToolCall,...]), ('additional_kwargs', {...}), etc.
            # We accumulate only supported fields into `parts` and skip unsupported metadata.
            # More info at: https://langchain-ai.github.io/langgraph/cloud/how-tos/stream_messages/
            processed_messages = []
            current_message: dict[str, Any] = {}
            for message in new_messages:
                if isinstance(message, tuple):
                    key, value = message
                    # Store parts in temporary dict
                    current_message[key] = value
                else:
                    # Add complete message if we have one in progress
                    if current_message:
                        processed_messages.append(_create_ai_message(current_message))
                        current_message = {}
                    processed_messages.append(message)

            # Add any remaining message parts
            if current_message:
                processed_messages.append(_create_ai_message(current_message))

            for message in processed_messages:
                try:
                    chat_message = langchain_to_chat_message(message)
                    chat_message.run_id = str(run_id)
                except Exception as e:
                    logger.error(f"Error parsing message: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Unexpected error'})}\n\n"
                    continue
                # LangGraph re-sends the input message, which feels weird, so drop it
                if chat_message.type == "human" and chat_message.content == user_input.message:
                    continue
                # 记录最终 AI 回答（后面的会覆盖前面的，留下最后一条）
                if chat_message.type == "ai":
                    final_answer = convert_message_content_to_string(chat_message.content)
                yield f"data: {json.dumps({'type': 'message', 'content': chat_message.model_dump()})}\n\n"

            if stream_mode == "messages":
                if not user_input.stream_tokens:
                    continue
                msg, metadata = event
                if "skip_stream" in metadata.get("tags", []):
                    continue
                # For some reason, astream("messages") causes non-LLM nodes to send extra messages.
                # Drop them.
                if not isinstance(msg, AIMessageChunk):
                    continue
                content = remove_tool_calls(msg.content)
                if content:
                    # Empty content in the context of OpenAI usually means
                    # that the model is asking for a tool to be invoked.
                    # So we only print non-empty content.
                    yield f"data: {json.dumps({'type': 'token', 'content': convert_message_content_to_string(content)})}\n\n"
    except Exception as e:
        logger.error(f"Error in message generator: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': 'Internal server error'})}\n\n"
    finally:
        yield "data: [DONE]\n\n"

    # 本轮对话落 MySQL（尽力而为：库挂了只记日志，不影响 SSE 已发出的内容）
    if final_answer is not None:
        latency_ms = (time.perf_counter() - turn_started) * 1000
        try:
            await asyncio.to_thread(
                _persist_chat_turn,
                thread_id,
                user_input.message,
                final_answer,
                citations,
                latency_ms,
            )
        except Exception as e:
            logger.warning(f"Chat persistence skipped (MySQL unavailable?): {e}")

        # QA 缓存写入：答案 + 问题嵌入（语义级查找用），Redis 挂了同样降级跳过
        if agent_id == "rag-assistant":
            try:
                await asyncio.to_thread(
                    _store_qa_cache, user_input.message, final_answer, citations
                )
            except Exception as e:
                logger.warning(f"QA cache store skipped: {e}")


async def _persist_chat_turn_async(
    thread_id: str,
    user_message: str,
    assistant_message: str,
    citations: list[dict] | None,
    latency_sec: float,
) -> None:
    await asyncio.to_thread(
        _persist_chat_turn,
        thread_id,
        user_message,
        assistant_message,
        citations,
        latency_sec * 1000,
    )


def _embed_question(question: str) -> list[float]:
    """复用 Chroma 检索的同一嵌入模型（进程内已缓存实例），保证
    缓存向量与检索向量同空间，可直接做余弦比较。"""
    return get_chroma_store().embeddings.embed_query(question)


def _store_qa_cache(
    question: str, answer: str, citations: list[dict] | None
) -> None:
    store(question, answer, citations)
    try:
        vector = _embed_question(question)
    except Exception as e:
        # 语义级是增强能力：嵌入失败不影响精确级缓存
        logger.warning(f"QA embedding store skipped: {e}")
        return
    store_embedding(question, vector)


def _persist_chat_turn(
    thread_id: str,
    user_message: str,
    assistant_message: str,
    citations: list[dict] | None,
    latency_ms: float,
) -> None:
    """一轮问答 = 两条消息（user + assistant）。会话不存在则按 thread_id 建。"""
    with session_scope() as db:
        if get_session_by_id(db, thread_id) is None:
            title = user_message[:50] or "新会话"
            create_session(db, title=title, session_id=thread_id)
        append_message(db, thread_id, MessageRole.USER, user_message)
        append_message(
            db,
            thread_id,
            MessageRole.ASSISTANT,
            assistant_message,
            citations={"items": citations} if citations else None,
            latency_ms=round(latency_ms, 1),
        )


def _create_ai_message(parts: dict) -> AIMessage:
    sig = inspect.signature(AIMessage)
    valid_keys = set(sig.parameters)
    filtered = {k: v for k, v in parts.items() if k in valid_keys}
    return AIMessage(**filtered)


def _sse_response_example() -> dict[int | str, Any]:
    return {
        status.HTTP_200_OK: {
            "description": "Server Sent Event Response",
            "content": {
                "text/event-stream": {
                    "example": "data: {'type': 'token', 'content': 'Hello'}\n\ndata: {'type': 'token', 'content': ' World'}\n\ndata: [DONE]\n\n",
                    "schema": {"type": "string"},
                }
            },
        }
    }


@router.post(
    "/{agent_id}/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
    operation_id="stream_with_agent_id",
    dependencies=[Depends(rate_limit)],
)
@router.post(
    "/stream",
    response_class=StreamingResponse,
    responses=_sse_response_example(),
    dependencies=[Depends(rate_limit)],
)
async def stream(
    request: Request,
    user_input: StreamInput,
    agent_id: str = DEFAULT_AGENT,
) -> StreamingResponse:
    """
    Stream an agent's response to a user input, including intermediate messages and tokens.

    If agent_id is not provided, the default agent will be used.
    Use thread_id to persist and continue a multi-turn conversation. run_id kwarg
    is also attached to all messages for recording feedback.
    Use user_id to persist and continue a conversation across multiple threads.
    When authenticated with a per-user API key, the server overrides user_id
    with the caller's identity.

    Set `stream_tokens=false` to return intermediate messages but not token-by-token.
    """
    principal = getattr(request.state, "principal", None)
    return StreamingResponse(
        message_generator(user_input, agent_id, principal),
        media_type="text/event-stream",
    )


@router.post("/ingest", operation_id="ingest_document", dependencies=[Depends(rate_limit)])
async def ingest_document(file: Annotated[UploadFile, File(description="PDF or DOCX to index")]) -> IngestResponse:
    """
    Upload a PDF/DOCX into the DocPilot knowledge base.

    The file is chunked, embedded and written to the configured Chroma DB.
    Re-uploading a file with the same name replaces its previous chunks.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(tuple(SUPPORTED_EXTENSIONS)):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)}MB).",
        )

    # 分布式锁：同名文件并发上传时只放一个进解析（解析+嵌入数秒，
    # 两个并发会互相删除对方的 chunk 造成数据错乱）
    lock_name = "docpilot:lock:ingest:" + hashlib.md5(filename.encode("utf-8")).hexdigest()
    lock_token: str | None = None
    lock_state_known = False
    try:
        lock_token = await asyncio.to_thread(
            acquire_lock, lock_name, settings.INGEST_LOCK_TTL_SECONDS
        )
        lock_state_known = True
    except Exception as e:
        # Redis 不可用：降级放行（与 MySQL 降级同一原则，主功能优先）
        logger.warning(f"Ingest lock unavailable (Redis down?): {e}")
    if lock_state_known and lock_token is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The same file is currently being processed. Try again shortly.",
        )

    try:
        # MySQL 登记拿 document_id；READY 状态的同哈希同文件名直接秒传返回
        try:
            with session_scope() as db:
                doc, created = register_document(db, filename, content)
                if not created:
                    return IngestResponse(
                        status="success",
                        filename=doc.filename,
                        chunks_added=doc.chunks_added,
                        chunks_deleted=doc.chunks_deleted,
                        deduplicated=True,
                        document_id=doc.id,
                    )
                document_id = doc.id
        except Exception as e:
            # MySQL 不可用时退回纯 Chroma 老路径，功能不中断
            logger.warning(f"Document registration skipped (MySQL unavailable): {e}")
            document_id = None

        # 状态机：pending -> processing
        if document_id is not None:
            with session_scope() as db:
                transition_status(db, document_id, DocumentStatus.PROCESSING)

        try:
            # 嵌入是阻塞的 GPU/CPU 工作，放线程池避免卡住事件循环
            result = await asyncio.to_thread(ingest_file, filename, content)
        except ValueError as e:
            if document_id is not None:
                with session_scope() as db:
                    transition_status(db, document_id, DocumentStatus.FAILED, error_message=str(e))
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

        # 状态机：processing -> ready
        if document_id is not None:
            with session_scope() as db:
                transition_status(
                    db,
                    document_id,
                    DocumentStatus.READY,
                    chunks_added=result.chunks_added,
                    chunks_deleted=result.chunks_deleted,
                )

        # Cache-Aside 的删缓存侧：知识库变了，QA 缓存整体失效。
        # 宁可让下一个问题重算（慢但正确），不能回吐基于旧库的答案。
        try:
            invalidated = await asyncio.to_thread(invalidate_all)
            if invalidated:
                logger.info("QA cache invalidated after ingest: %d keys", invalidated)
        except Exception as e:
            logger.warning(f"QA cache invalidation skipped (Redis down?): {e}")

        return IngestResponse(
            status="success",
            filename=result.filename,
            chunks_added=result.chunks_added,
            chunks_deleted=result.chunks_deleted,
            deduplicated=False,
            document_id=document_id,
        )
    finally:
        if lock_token is not None:
            try:
                await asyncio.to_thread(release_lock, lock_name, lock_token)
            except Exception as e:
                # 释放失败只能等 TTL 过期，日志留痕
                logger.warning(f"Ingest lock release failed (will expire): {e}")


@router.get("/documents", operation_id="list_documents")
async def documents(
    status_filter: Annotated[str | None, Query(alias="status")] = None,
) -> DocumentListResponse:
    """List uploaded documents and their ingestion status from MySQL metadata."""
    valid = {s.value for s in DocumentStatus}
    if status_filter is not None and status_filter not in valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid status filter. Valid values: {sorted(valid)}",
        )
    filter_enum = DocumentStatus(status_filter) if status_filter else None
    try:
        with session_scope() as db:
            docs = list_documents(db, status=filter_enum)
            items = [
                DocumentOut(
                    id=d.id,
                    filename=d.filename,
                    size_bytes=d.size_bytes,
                    status=d.status.value if hasattr(d.status, "value") else str(d.status),
                    chunks_added=d.chunks_added,
                    chunks_deleted=d.chunks_deleted,
                    error_message=d.error_message,
                    created_at=d.created_at.isoformat() if d.created_at else None,
                    updated_at=d.updated_at.isoformat() if d.updated_at else None,
                )
                for d in docs
            ]
            return DocumentListResponse(documents=items, total=len(items))
    except Exception as e:
        logger.warning(f"Document listing unavailable (MySQL down?): {e}")
        return DocumentListResponse(documents=[], total=0)


@router.post("/feedback")
async def feedback(feedback: Feedback) -> FeedbackResponse:
    """
    Record feedback for a run to LangSmith.

    This is a simple wrapper for the LangSmith create_feedback API, so the
    credentials can be stored and managed in the service rather than the client.
    See: https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
    """
    client = LangsmithClient()
    kwargs = feedback.kwargs or {}
    client.create_feedback(
        run_id=feedback.run_id,
        key=feedback.key,
        score=feedback.score,
        **kwargs,
    )
    return FeedbackResponse()


async def _recent_thread_ids_sqlite(limit: int) -> list[str]:
    """按写入新旧枚举 thread_id（sqlite 实现）。

    AsyncSqliteSaver 不支持无 thread_id 的跨线程 alist（langgraph 内部
    search_where 直接 KeyError），所以直接只读查询 checkpoints 表：
    rowid 顺序即写入顺序，MAX(rowid) 就是各线程的最新写入。
    """
    import aiosqlite

    db_path = settings.SQLITE_DB_PATH
    if not os.path.exists(db_path):
        return []
    async with aiosqlite.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        async with conn.execute(
            "SELECT thread_id FROM checkpoints GROUP BY thread_id "
            "ORDER BY MAX(rowid) DESC LIMIT ?",
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
    return [r[0] for r in rows]


@router.get("/threads", operation_id="list_threads", response_model=list[ThreadInfo])
async def list_threads(limit: Annotated[int, Query(ge=1, le=200)] = 50) -> list[ThreadInfo]:
    """
    List conversation threads, newest first (latest checkpoint per thread).

    All agents share one checkpointer, so threads are not attributed to a
    specific agent. user_id is not persisted in checkpoints, so per-user
    filtering is not available yet (see roadmap item 10: user accounts).
    """
    agent: AgentGraph = get_agent(DEFAULT_AGENT)
    checkpointer = getattr(agent, "checkpointer", None)
    if checkpointer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Checkpointer not initialized",
        )
    if settings.DATABASE_TYPE != DatabaseType.SQLITE:
        # postgres saver 支持无 thread_id 的 alist，尚未实现；先按部署形态收边界
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Thread listing is currently supported for the sqlite checkpointer only",
        )

    thread_ids = await _recent_thread_ids_sqlite(limit)

    threads: list[ThreadInfo] = []
    for tid in thread_ids:
        tup = await checkpointer.aget_tuple({"configurable": {"thread_id": tid}})
        if tup is None:
            continue
        checkpoint = tup.checkpoint or {}
        messages = checkpoint.get("channel_values", {}).get("messages", [])
        preview = ""
        for m in messages:
            if getattr(m, "type", "") == "human":
                preview = str(m.content)[:80]
                break
        threads.append(
            ThreadInfo(
                thread_id=tid,
                updated_at=str(checkpoint.get("ts", "")) or None,
                preview=preview,
                message_count=len(messages),
            )
        )
    return threads


@router.post("/{agent_id}/history", operation_id="history_with_agent_id")
@router.post("/history")
async def history(input: ChatHistoryInput, agent_id: str = DEFAULT_AGENT) -> ChatHistory:
    """
    Get chat history for a thread and agent.

    If agent_id is not provided, the default agent will be used.
    """
    agent: AgentGraph = get_agent(agent_id)
    try:
        state_snapshot = await agent.aget_state(
            config=RunnableConfig(configurable={"thread_id": input.thread_id})
        )
        messages: list[AnyMessage] = state_snapshot.values["messages"]
        chat_messages: list[ChatMessage] = [langchain_to_chat_message(m) for m in messages]
        return ChatHistory(messages=chat_messages)
    except Exception as e:
        logger.error(f"An exception occurred: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")


@app.get("/health")
async def health_check():
    """Health check endpoint."""

    health_status = {"status": "ok"}

    if settings.LANGFUSE_TRACING:
        try:
            langfuse = Langfuse()
            health_status["langfuse"] = "connected" if langfuse.auth_check() else "disconnected"
        except Exception as e:
            logger.error(f"Langfuse connection error: {e}")
            health_status["langfuse"] = "disconnected"

    return health_status


app.include_router(router)
