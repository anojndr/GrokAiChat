import os
import time
import json
import uuid
import base64
import tempfile
import asyncio
from typing import List, Optional, Union, Literal, Dict, Any, AsyncGenerator
import logging

import aiohttp
import aiofiles
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from grok import Grok, GrokMessages, DEFAULT_USER_AGENT, AllCredentialsLimitedError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

credential_sets = []
for i in range(1, 6):
    cookie = os.getenv(f"COOKIES_{i}")
    csrf_token = os.getenv(f"X_CSRF_TOKEN_{i}")
    bearer_token = os.getenv(f"BEARER_TOKEN_{i}")

    if all([cookie, csrf_token, bearer_token]):
        credential_sets.append({
            "cookies": cookie,
            "csrf": csrf_token,
            "bearer": bearer_token
        })
        logger.info(f"Loaded credential set #{i}")
    elif any([cookie, csrf_token, bearer_token]):
        logger.warning(f"Incomplete credential set #{i} found in .env. Skipping.")

if not credential_sets:
    logger.error("FATAL: No complete Grok credential sets found in .env file.")
    logger.error("Please set COOKIES_1, X_CSRF_TOKEN_1, BEARER_TOKEN_1 (and optionally _2, _3, etc.)")

grok_client: Optional[Grok] = None
if credential_sets:
    try:
        grok_client = Grok(credential_sets=credential_sets)
        logger.info(f"Grok client initialized with {len(credential_sets)} credential set(s).")
    except ValueError as e:
        logger.error(f"ERROR initializing Grok client: {e}")
        grok_client = None
else:
    logger.error("Grok client not initialized because no credentials were loaded.")


app = FastAPI(
    title="Grok OpenAI-Compatible API",
    description="An adapter to use Grok AI via an OpenAI-compatible interface with credential rotation.",
    version="0.3.0",
)

class ImageUrl(BaseModel):
    url: str

class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str

class ImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl

ContentPart = Union[TextContentPart, ImageContentPart]

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[ContentPart]]

class ChatCompletionRequest(BaseModel):
    model: Literal["grok-3"] = Field(..., description="The only supported model is 'grok-3'")
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

class ResponseMessage(BaseModel):
    role: Literal["assistant"]
    content: Optional[str] = None

class Choice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: Optional[Literal["stop", "length", "function_call", "content_filter"]] = "stop"

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: UsageInfo = Field(default_factory=UsageInfo)

class DeltaMessage(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None

class ChunkChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]] = None

class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChunkChoice]
    usage: Optional[UsageInfo] = None


async def download_image(url: str) -> Optional[bytes]:
    """Downloads image from URL or decodes base64 data URI."""
    if url.startswith("data:image"):
        try:
            header, encoded = url.split(",", 1)
            return base64.b64decode(encoded)
        except Exception as e:
            logger.error(f"Error decoding base64 image: {e}")
            return None
    else:
        try:
            async with aiohttp.ClientSession() as session:
                headers = {'User-Agent': DEFAULT_USER_AGENT}
                async with session.get(url, timeout=20, headers=headers) as response:
                    response.raise_for_status()
                    return await response.read()
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")
            return None

def prepare_grok_prompt(messages: List[ChatMessage]) -> str:
    """
    Concatenates message history and the latest user message content
    into a single string.
    """
    prompt_parts = []
    for msg in messages:
        role = msg.role
        content = msg.content

        if isinstance(content, str):
            text_content = content
        elif isinstance(content, list):
            text_content = "\n".join(
                part.text for part in content if isinstance(part, TextContentPart) and part.text
            )
        else:
            text_content = ""

        if text_content:
             prompt_parts.append(f"{role.capitalize()}: {text_content}")

    return "\n\n".join(prompt_parts)

async def process_images(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
    """
    Finds image URLs in *all* user messages within the provided list,
    downloads them, uploads to Grok, and returns the attachment info. (Async)
    Handles potential AllCredentialsLimitedError during upload.
    """
    file_attachments = []
    image_tasks = []
    temp_files = []
    processed_urls = set()

    for msg in messages:
        if msg.role == 'user' and isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, ImageContentPart):
                    image_url = part.image_url.url
                    if image_url in processed_urls:
                        continue
                    processed_urls.add(image_url)

                    logger.info(f"Processing image URL: {image_url[:100]}...")
                    image_data = await download_image(image_url)
                    if image_data:
                        temp_file_path = None
                        try:
                            async with aiofiles.tempfile.NamedTemporaryFile("wb", suffix=".jpg", delete=False) as temp_file:
                                await temp_file.write(image_data)
                                temp_file_path = temp_file.name

                            if temp_file_path:
                                temp_files.append(temp_file_path)
                                logger.info(f"Image saved temporarily to: {temp_file_path}")

                                if grok_client:
                                    image_tasks.append(grok_client.upload_file(temp_file_path))
                                else:
                                    logger.warning("Grok client not available, skipping image upload.")

                        except Exception as e:
                            logger.error(f"Error creating/writing temporary image file: {e}")
                            if temp_file_path and os.path.exists(temp_file_path):
                                 try: os.unlink(temp_file_path)
                                 except OSError: pass

    if image_tasks:
        try:
            logger.info(f"Uploading {len(image_tasks)} image(s)...")
            upload_results = await asyncio.gather(*image_tasks, return_exceptions=False)
            for result in upload_results:
                if isinstance(result, list) and result:
                    file_attachments.extend(result)
                elif result is not None:
                    logger.warning(f"Unexpected or empty upload result: {result}")
        except AllCredentialsLimitedError:
             logger.error("Image upload failed: All credentials hit rate limit.")
             raise
        except Exception as e:
             logger.error(f"Error gathering image upload results: {e}")
             raise HTTPException(status_code=500, detail=f"Failed during image processing: {e}")

    cleanup_tasks = []
    for temp_path in temp_files:
        async def _delete_file(path):
            try:
                exists = await asyncio.to_thread(os.path.exists, path)
                if exists:
                    await asyncio.to_thread(os.remove, path)
                    logger.info(f"Cleaned up temporary file: {path}")
                else:
                    logger.debug(f"Temporary file already removed or not found: {path}")
            except OSError as e:
                logger.warning(f"Failed to clean up temporary file {path}: {e}")
        cleanup_tasks.append(_delete_file(temp_path))
    if cleanup_tasks:
        logger.info(f"Attempting cleanup of {len(cleanup_tasks)} temporary file(s)...")
        await asyncio.gather(*cleanup_tasks)
        logger.info("Temporary file cleanup complete.")


    return file_attachments



async def stream_grok_response(model: str, payload: dict, chunk_id_base: str, is_reasoning_request: bool) -> AsyncGenerator[str, None]:
    """
    Generates OpenAI-compatible SSE chunks from Grok's streaming response.
    Handles potential AllCredentialsLimitedError from the underlying client method.
    """
    created_time = int(time.time())
    chunk_index = 0
    finish_reason = None

    if not grok_client:
         logger.error("stream_grok_response called but grok_client is None")
         error_chunk = ChatCompletionChunk(
             id=chunk_id_base, model=model, choices=[ChunkChoice(index=0, delta=DeltaMessage(content="Error: Grok client not initialized"), finish_reason="stop")]
         )
         yield f"data: {error_chunk.model_dump_json()}\n\n"
         yield "data: [DONE]\n\n"
         return

    try:
        async for line in grok_client.send_stream(payload):
            line = line.strip()
            if not line:
                continue

            try:
                parsed_line = json.loads(line)
                result_data = parsed_line.get("result", {})
                is_thinking = result_data.get("isThinking", False)
                delta_content = result_data.get("message")
                response_type = result_data.get("responseType")

                if response_type == "limiter":
                    logger.warning("Limiter response received unexpectedly during stream generation. Stopping stream.")
                    finish_reason = "stop"
                    break

                if is_reasoning_request and is_thinking:
                    continue
                else:
                    if delta_content is not None:
                        chunk = ChatCompletionChunk(
                            id=chunk_id_base,
                            object="chat.completion.chunk",
                            created=created_time,
                            model=model,
                            choices=[
                                ChunkChoice(
                                    index=0,
                                    delta=DeltaMessage(role="assistant", content=delta_content),
                                    finish_reason=None,
                                )
                            ],
                        )
                        chunk_index += 1
                        yield f"data: {chunk.model_dump_json()}\n\n"

            except json.JSONDecodeError:
                logger.warning(f"Received non-JSON line from stream: {line}")
                continue
            except Exception as e:
                 logger.error(f"Error processing stream line: {line}, Error: {e}")

        if finish_reason is None:
            finish_reason = "stop"

        final_chunk = ChatCompletionChunk(
            id=chunk_id_base,
            object="chat.completion.chunk",
            created=created_time,
            model=model,
            choices=[
                ChunkChoice(
                    index=0,
                    delta=DeltaMessage(),
                    finish_reason=finish_reason,
                )
            ],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"

    except AllCredentialsLimitedError as e:
         logger.error(f"Stream generation failed: {e}")
         try:
             error_content = f"Error: All Grok accounts are currently rate-limited. Please try again later."
             error_chunk = ChatCompletionChunk(
                 id=chunk_id_base, model=model, choices=[ChunkChoice(index=0, delta=DeltaMessage(content=error_content), finish_reason="stop")]
             )
             yield f"data: {error_chunk.model_dump_json()}\n\n"
         except Exception as inner_e:
             logger.error(f"Failed to yield rate limit error chunk: {inner_e}")

    except ConnectionError as e:
         logger.error(f"Connection error during streaming: {e}")
         try:
             error_content = f"Error communicating with Grok service: {e}"
             error_chunk = ChatCompletionChunk(
                 id=chunk_id_base, model=model, choices=[ChunkChoice(index=0, delta=DeltaMessage(content=error_content), finish_reason="stop")]
             )
             yield f"data: {error_chunk.model_dump_json()}\n\n"
         except Exception as inner_e:
             logger.error(f"Failed to yield connection error chunk: {inner_e}")
    except Exception as e:
         logger.error(f"Unexpected error during streaming generation: {e.__class__.__name__}: {e}")
         import traceback
         traceback.print_exc()
         try:
             error_content = f"Unexpected Server Error during stream: {e.__class__.__name__}"
             error_chunk = ChatCompletionChunk(
                 id=chunk_id_base, model=model, choices=[ChunkChoice(index=0, delta=DeltaMessage(content=error_content), finish_reason="stop")]
             )
             yield f"data: {error_chunk.model_dump_json()}\n\n"
         except Exception as inner_e:
             logger.error(f"Failed to yield unexpected error chunk: {inner_e}")
    finally:
        yield "data: [DONE]\n\n"



@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completion endpoint using Grok AI.
    Supports streaming/non-streaming and credential rotation.
    """
    if not grok_client:
        raise HTTPException(status_code=503, detail="Grok client is not initialized. Check credentials and server logs.")

    request_id = f"chatcmpl-{uuid.uuid4()}"

    try:
        conversation_id = await grok_client.create_new_conversation()

        file_attachments = await process_images(request.messages)

        grok_prompt = prepare_grok_prompt(request.messages)
        if not grok_prompt:
             raise HTTPException(status_code=400, detail="No text content found in messages.")

        payload = grok_client.create_message_payload(
            model_name=request.model,
            conversation_id=conversation_id
        )
        grok_client.add_user_message_to_payload(
            request_payload=payload,
            message=grok_prompt,
            file_attachments=file_attachments
        )

        lower_prompt = grok_prompt.lower()
        if "deepsearch" in lower_prompt:
            payload["isDeepsearch"] = True
            logger.info("Deepsearch mode activated by keyword.")
        if "reasoning" in lower_prompt:
            payload["isReasoning"] = True
            logger.info("Reasoning mode activated by keyword.")

        is_reasoning_request = payload.get("isReasoning", False)

        if request.stream:
            logger.info(f"Initiating stream for request {request_id}...")
            return StreamingResponse(
                stream_grok_response(request.model, payload, request_id, is_reasoning_request),
                media_type="text/event-stream"
            )
        else:
            logger.info(f"Processing non-streaming request {request_id}...")
            raw_response_text = await grok_client.send(payload)

            parsed_response = GrokMessages(raw_response_text)
            if parsed_response.is_limiter_response():
                 logger.error("Received limiter response unexpectedly in non-streaming result.")
                 raise AllCredentialsLimitedError("Rate limit encountered even after retries.")

            full_message = parsed_response.get_full_message()

            response = ChatCompletionResponse(
                id=request_id,
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        message=ResponseMessage(role="assistant", content=full_message),
                        finish_reason="stop",
                    )
                ],
                usage=UsageInfo()
            )
            logger.info(f"Completed non-streaming request {request_id}.")
            return response

    except AllCredentialsLimitedError as e:
        logger.error(f"Request failed for {request_id}: {e}")
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "message": "All available Grok accounts are currently rate-limited. Please try again later or add more credentials.",
                    "type": "rate_limit_error",
                    "param": None,
                    "code": None
                }
            }
        )
    except (ConnectionError, IOError, aiohttp.ClientError) as e:
        logger.error(f"API Communication Error for {request_id}: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to communicate with Grok API or related service: {e}")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected Server Error for {request_id}: {e.__class__.__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.get("/v1/models")
async def list_models():
    """Provides a list of available Grok models in OpenAI format."""
    grok_models = ["grok-3"]
    data = [
        {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "xai"
        } for model_id in grok_models
    ]
    return {"object": "list", "data": data}

@app.get("/")
async def read_root():
    return {"message": "Grok OpenAI-Compatible API with credential rotation is running. Use the /v1/chat/completions endpoint."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting server on {host}:{port}")
    if grok_client is None:
        logger.critical("FATAL: Grok client failed to initialize. Server cannot function correctly.")
    uvicorn.run("main:app", host=host, port=port, reload=True)