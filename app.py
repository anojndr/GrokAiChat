import os
import time
import json
import uuid
import re
import tempfile
import logging
import requests
from typing import List, Dict, Any, Optional, Union, Tuple, Generator, AsyncGenerator
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from functools import lru_cache
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("grok-api")

# Import the Grok modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from grok import Grok, GrokMessages

# Pydantic models for request/response validation
class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class Message(BaseModel):
    role: str
    content: Union[str, List[MessageContent]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False

class Delta(BaseModel):
    content: Optional[str] = None

class Choice(BaseModel):
    index: int
    delta: Optional[Delta] = None
    message: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

class ModelObject(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str

class ModelList(BaseModel):
    object: str
    data: List[ModelObject]

class ErrorResponse(BaseModel):
    error: Dict[str, Any]

# Load environment variables
load_dotenv()

# Model mappings
MODEL_MAPPINGS = {
    "gpt-3.5-turbo": "grok-1",
    "gpt-4": "grok-2",
    "gpt-4-turbo": "grok-3",
    "gpt-4-vision": "grok-3",
    "gpt-4-vision-preview": "grok-3",
    "grok-1": "grok-1",
    "grok-2": "grok-2",
    "grok-3": "grok-3"
}

# Rate limit patterns for efficient detection
RATE_LIMIT_PATTERNS = [
    "You've reached your limit of", 
    "limit of 15 Grok questions", 
    "limit of 10 Grok DeepSearch questions",
    "responseType\": \"limiter\"",
    "upsellType\": \"free_grok"
]

class CredentialManager:
    """Manages and rotates API credentials"""
    
    def __init__(self):
        """Initialize the credential manager with environment variables"""
        self.cookies_list = []
        self.x_csrf_token_list = []
        self.bearer_token_list = []
        self.current_index = 0
        self._load_credentials()
        
        if not self.cookies_list:
            raise ValueError("No valid credential sets found. Please check your environment variables.")
            
        logger.info(f"Loaded {len(self.cookies_list)} credential sets")
        
    def _load_credentials(self):
        """Load credentials from environment variables"""
        for i in range(1, 11):  # Support up to 10 credential sets
            # Try both naming formats (with and without index)
            cookie_key = f"COOKIES_{i}" if i > 1 else "COOKIES"
            csrf_key = f"X_CSRF_TOKEN_{i}" if i > 1 else "X_CSRF_TOKEN"
            bearer_key = f"BEARER_TOKEN_{i}" if i > 1 else "BEARER_TOKEN"
            
            cookie = os.getenv(cookie_key)
            csrf = os.getenv(csrf_key)
            bearer = os.getenv(bearer_key)
            
            if cookie and csrf and bearer:
                self.cookies_list.append(cookie)
                self.x_csrf_token_list.append(csrf)
                self.bearer_token_list.append(bearer)
    
    def get_current_credentials(self) -> Tuple[str, str, str]:
        """Get the current set of credentials"""
        return (
            self.bearer_token_list[self.current_index],
            self.x_csrf_token_list[self.current_index],
            self.cookies_list[self.current_index]
        )
    
    def rotate(self) -> Tuple[str, str, str]:
        """Rotate to the next set of credentials and return them"""
        if len(self.cookies_list) > 1:
            self.current_index = (self.current_index + 1) % len(self.cookies_list)
            logger.info(f"Rotated to credential set {self.current_index + 1}/{len(self.cookies_list)}")
        return self.get_current_credentials()
    
    def get_count(self) -> int:
        """Return the number of credential sets available"""
        return len(self.cookies_list)
    
    def get_current_index(self) -> int:
        """Return the current credential set index (0-based)"""
        return self.current_index

class GrokClientManager:
    """Manages Grok client instances and handles credential rotation"""
    
    def __init__(self, credential_manager: CredentialManager):
        """Initialize with a credential manager"""
        self.credential_manager = credential_manager
        self._create_client()
        
    def _create_client(self) -> None:
        """Create a new Grok client with current credentials"""
        bearer, csrf, cookies = self.credential_manager.get_current_credentials()
        
        # Get retry settings from environment variables
        retry_count = int(os.getenv("GROK_RETRY_COUNT", "2"))
        retry_backoff = float(os.getenv("GROK_RETRY_BACKOFF", "1.5"))
        
        self.client = Grok(
            bearer, 
            csrf, 
            cookies, 
            retry_count=retry_count,
            retry_backoff=retry_backoff
        )
        
    def get_client(self) -> Grok:
        """Get the current Grok client instance"""
        return self.client
        
    def rotate(self) -> Grok:
        """Rotate credentials and return a new Grok client"""
        self.credential_manager.rotate()
        self._create_client()
        return self.client

# Max retries for API requests
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Initialize credential manager and Grok client manager
credential_manager = CredentialManager()
grok_client_manager = GrokClientManager(credential_manager)

# Get the port number from environment variable with default
PORT = int(os.getenv("PORT", "5000"))

# Request buffer size for streaming responses
STREAM_BUFFER_SIZE = int(os.getenv("STREAM_BUFFER_SIZE", "10"))

# Initialize FastAPI app
app = FastAPI(title="GrokAI OpenAI-compatible API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@lru_cache(maxsize=100)
def get_grok_model(openai_model: str) -> str:
    """Maps OpenAI model names to Grok model names with caching for performance."""
    return MODEL_MAPPINGS.get(openai_model, "grok-3")  # Default to grok-3

def is_rate_limited(response_text: str) -> bool:
    """Check if the response indicates a rate limit using efficient pattern matching."""
    if not response_text:
        return False
        
    # Quick check for common patterns
    for pattern in RATE_LIMIT_PATTERNS:
        if pattern in response_text:
            return True
    
    # More thorough check with JSON parsing but only if necessary
    try:
        for line in response_text.splitlines():
            if not line.strip():
                continue
                
            try:
                parsed = json.loads(line)
                result = parsed.get("result", {})
                
                # Check for rate limit indicators in parsed JSON structure
                if result.get("responseType") == "limiter":
                    return True
                
                # Check for upsell object which indicates premium requirement
                if "upsell" in result:
                    return True
                
                # Check for rate limit message in parsed message content
                message = result.get("message", "")
                if message and ("You've reached your limit" in message or "limit of" in message):
                    return True
            except json.JSONDecodeError:
                # Skip lines that aren't valid JSON
                continue
    except Exception as e:
        logger.error(f"Error checking for rate limits: {str(e)}")
    
    return False

def detect_rate_limit_pattern(response_text: str) -> List[str]:
    """Detect which rate limit pattern was triggered for better logging."""
    patterns = {
        "standard_limit": "You've reached your limit of 15 Grok questions",
        "deepsearch_limit": "You've reached your limit of 10 Grok DeepSearch questions",
        "limiter_response": "responseType\": \"limiter\"",
        "upsell_detected": "upsellType\": \"free_grok"
    }
    
    detected = []
    for name, pattern in patterns.items():
        if pattern in response_text:
            detected.append(name)
    
    return detected if detected else ["unknown_rate_limit"]

def filter_thinking_traces(response_text: str) -> str:
    """Filter out thinking traces from the response efficiently."""
    if not response_text:
        return ""
        
    filtered_lines = []
    
    for line in response_text.splitlines():
        if not line.strip():
            continue
            
        try:
            parsed = json.loads(line)
            result_data = parsed.get("result", {})
            
            # Skip thinking traces
            if result_data.get("messageTag") == "thinking_trace" or result_data.get("isThinking") is True:
                continue
                
            filtered_lines.append(line)
        except json.JSONDecodeError:
            # Keep non-JSON lines as they might be important
            filtered_lines.append(line)
            
    return "\n".join(filtered_lines)

async def download_and_upload_image(grok_client: Grok, image_url: str) -> Optional[Dict]:
    """
    Downloads an image from a URL or processes a data URI and uploads it to Grok.
    Improved to handle various error cases more gracefully.
    """
    temp_file_path = None
    
    try:
        # Check if it's a data URI
        if image_url.startswith('data:'):
            # Handle data URI
            import base64
            # Parse the data URI to get the metadata and the base64 content
            try:
                metadata, base64_data = image_url.split(',', 1)
                
                # Determine the file extension from the metadata
                mime_type = metadata.split(';')[0].split(':')[1]
                extension = '.' + mime_type.split('/')[1]  # For example, '.png' from 'image/png'
                
                # Generate a random filename with the appropriate extension
                filename = f"image_{uuid.uuid4().hex}{extension}"
                
                # Decode the base64 data
                image_data = base64.b64decode(base64_data)
                
                # Save to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
                    temp_file.write(image_data)
                    temp_file_path = temp_file.name
            except Exception as e:
                logger.error(f"Error processing data URI: {str(e)}")
                return None
        else:
            # Handle regular URL with error handling
            try:
                async def fetch_with_timeout():
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, 
                        lambda: requests.get(image_url, stream=True)
                    )
                
                response = await fetch_with_timeout()
                
                if response.status_code != 200:
                    logger.error(f"Failed to download image from {image_url}: {response.status_code}")
                    return None
                
                # Get the filename from the URL or use a random name
                parsed_url = urlparse(image_url)
                path = parsed_url.path
                filename = os.path.basename(path)
                if not filename or '.' not in filename:
                    filename = f"image_{uuid.uuid4().hex}.jpg"
                
                # Save the image to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)
                    temp_file_path = temp_file.name
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error downloading image: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error downloading image: {str(e)}")
                return None
        
        # Upload the image to Grok
        if temp_file_path:
            upload_result = grok_client.upload_file(temp_file_path)
            return upload_result
        
        return None
    except Exception as e:
        logger.error(f"Error in download_and_upload_image: {str(e)}")
        return None
    finally:
        # Always clean up the temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file_path}: {str(e)}")

async def prepare_messages(messages: List[Message], grok_client: Grok) -> Tuple[str, List]:
    """
    Prepares messages in a format suitable for Grok, with support for image URLs.
    Returns a tuple of (message_text, file_attachments)
    """
    # Initialize file_attachments list at the beginning
    file_attachments = []
    
    # For Grok, we'll include previous messages in the content for context
    context_messages = []
    for i, msg in enumerate(messages[:-1]):
        if msg.role == "system":
            context_messages.append(f"System: {msg.content}")
        elif msg.role == "user":
            # Handle new message format where content could be a list
            if isinstance(msg.content, list):
                # Extract text parts from the content list
                text_parts = []
                for item in msg.content:
                    if item.type == 'text':
                        text_parts.append(item.text)
                    elif item.type == 'image_url':
                        # Process image URL from previous messages too
                        image_url = item.image_url.get('url')
                        if image_url:
                            uploaded_file = await download_and_upload_image(grok_client, image_url)
                            if uploaded_file:
                                if isinstance(uploaded_file, list):
                                    file_attachments.extend(uploaded_file)
                                else:
                                    file_attachments.append(uploaded_file)
                msg_content = " ".join(text_parts)
            else:
                msg_content = msg.content
            context_messages.append(f"User: {msg_content}")
        elif msg.role == "assistant":
            context_messages.append(f"Assistant: {msg.content}")
    
    context_text = "\n\n".join(context_messages)
    
    # Get the last message
    last_message = messages[-1]
    
    # Handle new format for the last message
    if isinstance(last_message.content, list):
        # This is the new format with potential images
        text_parts = []
        for item in last_message.content:
            if item.type == 'text':
                text_parts.append(item.text)
            elif item.type == 'image_url':
                # Process image URL - download and upload it to Grok
                image_url = item.image_url.get('url')
                if image_url:
                    uploaded_file = await download_and_upload_image(grok_client, image_url)
                    if uploaded_file:
                        if isinstance(uploaded_file, list):
                            file_attachments.extend(uploaded_file)  # Extend with the list of uploaded files
                        else:
                            file_attachments.append(uploaded_file)  # Append a single uploaded file
        last_message_content = " ".join(text_parts)
    else:
        last_message_content = last_message.content
    
    if context_text:
        # If there's context, include it in the message
        message_text = f"{context_text}\n\n{last_message_content}"
    else:
        # Otherwise, just return the content
        message_text = last_message_content
    
    return message_text, file_attachments

def check_special_keywords(message: str) -> Tuple[str, bool, bool, bool]:
    """
    Check if the message contains special keywords and returns the appropriate flags.
    Returns a tuple of (modified_message, is_deepsearch, is_reasoning, is_deeper_search)
    """
    is_deepsearch = False
    is_reasoning = False
    is_deeper_search = False
    modified_message = message
    
    # Convert to lowercase for case-insensitive searching
    message_lower = message.lower()
    
    # Check for "deepersearch" keyword anywhere in the message
    if "deepersearch" in message_lower:
        is_deepsearch = True  # Set deepsearch to true for deeper search mode
        is_deeper_search = True
        # Remove all instances of "deepersearch" (case insensitive)
        modified_message = re.sub(r'(?i)deepersearch\s*', '', modified_message).strip()
    # Check for regular "deepsearch" keyword
    elif "deepsearch" in message_lower:
        is_deepsearch = True
        # Remove all instances of "deepsearch" (case insensitive)
        modified_message = re.sub(r'(?i)deepsearch\s*', '', modified_message).strip()
    
    # Check for "reasoning" keyword anywhere in the message
    if "reasoning" in message_lower:
        is_reasoning = True
        # Remove all instances of "reasoning" (case insensitive)
        modified_message = re.sub(r'(?i)reasoning\s*', '', modified_message).strip()
    
    return modified_message, is_deepsearch, is_reasoning, is_deeper_search

def format_response_to_openai(grok_response: str, model: str, messages: List[Message], is_deepsearch: bool = False) -> Dict:
    """
    Formats a Grok response to match the OpenAI API format.
    Improved to handle filtering more efficiently.
    """
    # Filter the response if needed, but always check rate limits on the full response
    grok_response_filtered = filter_thinking_traces(grok_response)
    
    # Parse Grok response
    response_obj = GrokMessages(grok_response_filtered)
    full_message = response_obj.get_full_message()
    
    # Create response in OpenAI format
    response_id = f"chatcmpl-{uuid.uuid4()}"
    created_timestamp = int(time.time())
    
    # Token counting - more accurate approach but still approximation
    prompt_tokens = 0
    for msg in messages:
        if isinstance(msg.content, str):
            prompt_tokens += len(msg.content) // 4
        elif isinstance(msg.content, list):
            for item in msg.content:
                if item.type == 'text' and item.text:
                    prompt_tokens += len(item.text) // 4
    
    completion_tokens = len(full_message) // 4
    total_tokens = prompt_tokens + completion_tokens
    
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created_timestamp,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_message
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    }

def format_streaming_chunk(text: str, response_id: str, model: str) -> str:
    """Format a text chunk for streaming in OpenAI format."""
    data = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": text
                },
                "finish_reason": None
            }
        ]
    }
    return f"data: {json.dumps(data)}\n\n"

async def process_streaming_grok_response(
    grok_message_tokens: Generator, 
    response_id: str, 
    model: str
) -> AsyncGenerator[str, None]:
    """
    Process Grok message tokens as they arrive and yield formatted chunks for streaming.
    """
    for message_token in grok_message_tokens:
        # Skip any special tokens or empty tokens
        if not message_token or isinstance(message_token, str) and message_token.startswith("__"):
            continue
            
        # Try to parse JSON if it looks like a JSON string
        if isinstance(message_token, str) and message_token.strip().startswith("{"):
            try:
                parsed = json.loads(message_token)
                result_data = parsed.get("result", {})
                
                # Skip thinking traces
                if result_data.get("messageTag") == "thinking_trace" or result_data.get("isThinking") is True:
                    continue
            except json.JSONDecodeError:
                # Not JSON, so keep it
                pass
            
        # Format the token as an OpenAI compatible streaming chunk
        yield format_streaming_chunk(message_token, response_id, model)
    
    # Send the final chunk with finish_reason
    yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
    
    # End the stream
    yield "data: [DONE]\n\n"

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """List available models endpoint."""
    current_time = int(time.time())
    models = [
        {"id": "grok-1", "object": "model", "created": current_time, "owned_by": "Grok"},
        {"id": "grok-2", "object": "model", "created": current_time, "owned_by": "Grok"},
        {"id": "grok-3", "object": "model", "created": current_time, "owned_by": "Grok"}
    ]
    return {"object": "list", "data": models}

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, responses={500: {"model": ErrorResponse}})
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Improved with better error handling, credential rotation, and performance optimizations.
    """
    try:
        # Get request parameters
        messages = request.messages
        model = request.model
        stream = request.stream
        
        # Map model to Grok model
        grok_model = get_grok_model(model)
        
        # Maximum number of retry attempts - use all available credential sets
        max_retries = min(MAX_RETRIES, credential_manager.get_count())
        retries = 0
        success = False
        
        while not success and retries < max_retries:
            # Get the current Grok client
            grok_client = grok_client_manager.get_client()
            
            # Create a new conversation
            try:
                grok_client.create_conversation()
            except Exception as e:
                logger.error(f"Error creating conversation: {str(e)}, rotating credentials (attempt {retries + 1})")
                grok_client_manager.rotate()
                retries += 1
                continue
            
            # Check for keywords in the last message
            last_message = messages[-1]
            last_message_content = ""
            
            # Extract the content from the last message
            if isinstance(last_message.content, list):
                text_parts = []
                for item in last_message.content:
                    if item.type == 'text':
                        text_parts.append(item.text)
                last_message_content = " ".join(text_parts)
            else:
                last_message_content = last_message.content
            
            # Check for special keywords only in the latest query
            modified_query, is_deepsearch, is_reasoning, is_deeper_search = check_special_keywords(last_message_content)
            
            # Create a copy of the messages with the modified last message
            modified_messages = [Message(role=msg.role, content=msg.content) for msg in messages]
            if isinstance(modified_messages[-1].content, list):
                # For multimodal messages, update just the text parts
                new_content = []
                for item in modified_messages[-1].content:
                    new_item = MessageContent(**item.dict())
                    if new_item.type == 'text':
                        new_item.text = modified_query
                        # Only modify the first text part, leave others unchanged
                        new_content.append(new_item)
                        break
                    else:
                        new_content.append(new_item)
                
                # Add remaining items
                for item in modified_messages[-1].content[len(new_content):]:
                    new_content.append(item)
                
                modified_messages[-1].content = new_content
            else:
                modified_messages[-1].content = modified_query
            
            # Process messages to fit Grok's format with potential image attachments
            final_message, file_attachments = await prepare_messages(modified_messages, grok_client)
            
            # Prepare the request data with special flags if keywords detected
            request_data = grok_client.create_message(
                grok_model,
                isDeepsearch=is_deepsearch,
                isReasoning=is_reasoning,
                is_deeper_search=is_deeper_search
            )
            
            # Add the user message with any file attachments
            grok_client.add_user_message(request_data, final_message, file_attachments=file_attachments)
            
            # Generate a response ID
            response_id = f"chatcmpl-{uuid.uuid4()}"
            
            if stream:
                # Define the streaming generator function for streaming responses
                async def stream_generator() -> AsyncGenerator[str, None]:
                    nonlocal success, retries
                    nonlocal grok_client  # Make grok_client accessible from outer scope
                    
                    try:
                        # Get the streaming response
                        stream = grok_client.send_streaming(request_data, filter_final_only=is_deepsearch)
                        rate_limited = False
                        error_occurred = False
                        
                        # Use a more efficient buffer - fixed size deque
                        from collections import deque
                        response_buffer = deque(maxlen=STREAM_BUFFER_SIZE)
                        
                        for message_token in stream:
                            # Check for special tokens
                            if message_token == "__RATE_LIMITED__":
                                logger.warning("Rate limited in streaming mode")
                                rate_limited = True
                                break
                            elif message_token.startswith("__ERROR__") if isinstance(message_token, str) else False:
                                logger.error(f"Error in streaming mode: {message_token}")
                                error_occurred = True
                                break
                            
                            # Skip thinking traces - efficient check
                            is_thinking_trace = False
                            if isinstance(message_token, str) and message_token.strip():
                                try:
                                    if message_token.strip().startswith("{"):
                                        parsed = json.loads(message_token)
                                        result_data = parsed.get("result", {})
                                        if result_data.get("messageTag") == "thinking_trace" or result_data.get("isThinking") is True:
                                            is_thinking_trace = True
                                except Exception:
                                    # Not JSON or couldn't parse, assume it's safe to keep
                                    pass
                            
                            if not is_thinking_trace:
                                # Add to buffer
                                response_buffer.append(message_token)
                                
                                # If buffer has items, yield immediately to improve responsiveness
                                if response_buffer:
                                    yield format_streaming_chunk(response_buffer.popleft(), response_id, model)
                        
                        # If rate limited or error, try again with new credentials
                        if rate_limited or error_occurred:
                            retries += 1
                            if retries < max_retries:
                                logger.info(f"Connection issue, rotating credentials (attempt {retries})")
                                grok_client_manager.rotate()
                                grok_client = grok_client_manager.get_client()
                                
                                # Create a new conversation with new credentials
                                try:
                                    grok_client.create_conversation()
                                    new_request_data = grok_client.create_message(
                                        grok_model,
                                        isDeepsearch=is_deepsearch,
                                        isReasoning=is_reasoning,
                                        is_deeper_search=is_deeper_search
                                    )
                                    grok_client.add_user_message(new_request_data, final_message, file_attachments=file_attachments)
                                    
                                    # Try with the new credentials
                                    new_stream = grok_client.send_streaming(new_request_data, filter_final_only=is_deepsearch)
                                    for token in new_stream:
                                        # Skip special tokens and thinking traces - efficient version
                                        if isinstance(token, str) and (
                                            token == "__RATE_LIMITED__" or 
                                            token.startswith("__ERROR__")
                                        ):
                                            continue
                                        
                                        # Skip thinking traces
                                        is_thinking_trace = False
                                        if isinstance(token, str) and token.strip():
                                            try:
                                                if token.strip().startswith("{"):
                                                    parsed = json.loads(token)
                                                    result_data = parsed.get("result", {})
                                                    if result_data.get("messageTag") == "thinking_trace" or result_data.get("isThinking") is True:
                                                        is_thinking_trace = True
                                            except Exception:
                                                pass
                                        
                                        if not is_thinking_trace:
                                            yield format_streaming_chunk(token, response_id, model)
                                    
                                    # Send final tokens
                                    yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
                                    yield "data: [DONE]\n\n"
                                    success = True
                                except Exception as e:
                                    logger.error(f"Error in retry stream: {str(e)}")
                                    # If all retries failed, return any remaining buffer
                                    while response_buffer:
                                        yield format_streaming_chunk(response_buffer.popleft(), response_id, model)
                                    yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
                                    yield "data: [DONE]\n\n"
                            else:
                                # Out of retries, return any remaining buffer
                                while response_buffer:
                                    yield format_streaming_chunk(response_buffer.popleft(), response_id, model)
                                yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
                                yield "data: [DONE]\n\n"
                        else:
                            # No rate limiting, return remaining buffer
                            while response_buffer:
                                yield format_streaming_chunk(response_buffer.popleft(), response_id, model)
                            
                            # Send the final chunk
                            yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
                            yield "data: [DONE]\n\n"
                            success = True
                    except Exception as e:
                        logger.error(f"Error in streaming: {str(e)}")
                        # Return error message
                        error_msg = f"Error: {str(e)}"
                        yield format_streaming_chunk(error_msg, response_id, model)
                        yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
                        yield "data: [DONE]\n\n"
                
                # Return streaming response
                return StreamingResponse(
                    stream_generator(),
                    media_type="text/event-stream"
                )
            else:
                # Non-streaming mode with improved retry logic
                while retries < max_retries:
                    try:
                        # Send the message and get the response
                        grok_response = grok_client.send(request_data)
                        
                        # Check if we hit a rate limit
                        if is_rate_limited(grok_response):
                            limit_types = detect_rate_limit_pattern(grok_response)
                            logger.warning(f"Rate limited ({', '.join(limit_types)}), rotating credentials (attempt {retries + 1})")
                            grok_client_manager.rotate()
                            grok_client = grok_client_manager.get_client()
                            retries += 1
                            
                            # Create a new conversation with new credentials
                            try:
                                grok_client.create_conversation()
                                request_data = grok_client.create_message(
                                    grok_model,
                                    isDeepsearch=is_deepsearch,
                                    isReasoning=is_reasoning,
                                    is_deeper_search=is_deeper_search
                                )
                                grok_client.add_user_message(request_data, final_message, file_attachments=file_attachments)
                            except Exception as create_error:
                                logger.error(f"Error creating new conversation: {str(create_error)}")
                                # We'll retry in the next loop iteration
                            continue
                        
                        # Format the response to match OpenAI's format
                        openai_response = format_response_to_openai(grok_response, model, messages, is_deepsearch=is_deepsearch)
                        success = True
                        return openai_response
                    
                    except Exception as e:
                        logger.error(f"Error: {str(e)}")
                        retries += 1
                        if retries < max_retries:
                            grok_client_manager.rotate()
                            grok_client = grok_client_manager.get_client()
                        else:
                            break
                
                # If we've exhausted all retries
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": {
                            "message": f"Failed after {retries} attempts with different credentials",
                            "type": "connection_error",
                            "code": 500
                        }
                    }
                )
    
    except Exception as e:
        logger.error(f"Unhandled exception in chat_completions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "api_error",
                    "code": 500
                }
            }
        )

@app.get("/")
async def home():
    """Home endpoint with improved status information."""
    return {
        "status": "ok",
        "message": "GrokAI OpenAI-compatible API is running",
        "endpoints": {
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions"
        },
        "credentials": {
            "sets_available": credential_manager.get_count(),
            "current_set": credential_manager.get_current_index() + 1
        },
        "retry_settings": {
            "max_retries": MAX_RETRIES,
            "grok_retry_count": int(os.getenv("GROK_RETRY_COUNT", "2")),
            "grok_retry_backoff": float(os.getenv("GROK_RETRY_BACKOFF", "1.5"))
        },
        "stream_buffer_size": STREAM_BUFFER_SIZE,
        "version": "1.1.0"
    }

@app.on_event("startup")
async def startup_event():
    """Runs when the API server starts up."""
    logger.info(f"Starting GrokAI OpenAI-compatible API on port {PORT}")
    logger.info(f"Using {credential_manager.get_count()} credential sets")
    logger.info(f"Max retries: {MAX_RETRIES}")

# Uvicorn startup script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)