import os
import time
import json
import uuid
import re
import tempfile
import requests
from urllib.parse import urlparse
from flask import Flask, request, Response, jsonify, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv

# Import the Grok modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from grok import Grok, GrokMessages

# Load environment variables
load_dotenv()

# Credential sets structure - each index corresponds to a set of related credentials
COOKIES_LIST = []
X_CSRF_TOKEN_LIST = []
BEARER_TOKEN_LIST = []

# Get environment variables and parse multiple credential sets
for i in range(1, 6):  # Get 5 sets of credentials
    # Try both naming formats (with and without index)
    cookie_key = f"COOKIES_{i}" if i > 1 else "COOKIES"
    csrf_key = f"X_CSRF_TOKEN_{i}" if i > 1 else "X_CSRF_TOKEN"
    bearer_key = f"BEARER_TOKEN_{i}" if i > 1 else "BEARER_TOKEN"
    
    cookie = os.getenv(cookie_key)
    csrf = os.getenv(csrf_key)
    bearer = os.getenv(bearer_key)
    
    if cookie and csrf and bearer:
        COOKIES_LIST.append(cookie)
        X_CSRF_TOKEN_LIST.append(csrf)
        BEARER_TOKEN_LIST.append(bearer)

# Ensure we have at least one set of credentials
if not COOKIES_LIST:
    raise ValueError("No valid credential sets found. Please check your environment variables.")

PORT = int(os.getenv("PORT", 5000))

# Set API timeout settings - can be configured via environment variables
API_CONNECT_TIMEOUT = float(os.getenv("API_CONNECT_TIMEOUT", "10.0"))  # Default 10 seconds
API_READ_TIMEOUT = float(os.getenv("API_READ_TIMEOUT", "30.0"))        # Default 30 seconds
API_TIMEOUT = (API_CONNECT_TIMEOUT, API_READ_TIMEOUT)

app = Flask(__name__)
CORS(app)

# Current credential index
current_cred_index = 0

# Initialize the Grok client with the first set of credentials
grok_client = Grok(BEARER_TOKEN_LIST[current_cred_index], 
                  X_CSRF_TOKEN_LIST[current_cred_index], 
                  COOKIES_LIST[current_cred_index],
                  timeout=API_TIMEOUT)

# Model mappings
MODEL_MAPPINGS = {
    "gpt-3.5-turbo": "grok-1",
    "gpt-4": "grok-2",
    "gpt-4-turbo": "grok-3",
    "gpt-4-vision": "grok-3",  # Assuming grok-3 can handle images
    "gpt-4-vision-preview": "grok-3",  # Map vision model to grok-3
    "grok-1": "grok-1",
    "grok-2": "grok-2",
    "grok-3": "grok-3"
}

def filter_thinking_traces(line_text):
    """
    Check if a line contains thinking trace data that should be filtered out.
    
    Args:
        line_text: A JSON line from the Grok response
        
    Returns:
        bool: True if the line should be kept, False if it should be filtered out
    """
    try:
        parsed = json.loads(line_text)
        result_data = parsed.get("result", {})
        
        # Filter out thinking traces by messageTag
        if result_data.get("messageTag") == "thinking_trace":
            return False
            
        # Filter out thinking traces by isThinking flag
        if result_data.get("isThinking") == True:
            return False
            
        # Keep all other lines
        return True
    except:
        # If we can't parse it, keep it (better safe than sorry)
        return True

def rotate_credentials():
    """Rotate to the next set of credentials."""
    global current_cred_index, grok_client
    
    # Move to next credential set
    current_cred_index = (current_cred_index + 1) % len(COOKIES_LIST)
    
    # Reinitialize grok client with new credentials
    grok_client = Grok(BEARER_TOKEN_LIST[current_cred_index], 
                      X_CSRF_TOKEN_LIST[current_cred_index], 
                      COOKIES_LIST[current_cred_index],
                      timeout=API_TIMEOUT)
    
    print(f"Rotated to credential set {current_cred_index + 1}/{len(COOKIES_LIST)}")
    return grok_client

def detect_rate_limit_pattern(response_text):
    """Detect which rate limit pattern was triggered."""
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

def is_rate_limited(response_text):
    """Check if the response indicates a rate limit."""
    try:
        # Check for common rate limit patterns in the raw response text
        if any(pattern in response_text for pattern in [
            "You've reached your limit of", 
            "limit of 15 Grok questions", 
            "limit of 10 Grok DeepSearch questions",
            "responseType\": \"limiter\"",
            "upsellType\": \"free_grok"
        ]):
            return True
        
        # Try parsing the response as JSON lines
        for line in response_text.splitlines():
            if line.strip():
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
        print(f"Error checking for rate limits: {str(e)}")
    
    return False

def get_grok_model(openai_model):
    """Maps OpenAI model names to Grok model names."""
    return MODEL_MAPPINGS.get(openai_model, "grok-3")  # Default to grok-3

def download_and_upload_image(grok_client, image_url):
    """
    Downloads an image from a URL or processes a data URI and uploads it to Grok.
    
    Args:
        grok_client: The Grok client instance
        image_url: URL of the image to download or data URI containing image data
        
    Returns:
        The response from Grok's upload_file method or None if failed
    """
    try:
        # Check if it's a data URI
        if image_url.startswith('data:'):
            # Handle data URI
            import base64
            # Parse the data URI to get the metadata and the base64 content
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
        else:
            # Handle regular URL
            import requests
            response = requests.get(image_url, stream=True)
            if response.status_code != 200:
                print(f"Failed to download image from {image_url}: {response.status_code}")
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
        
        # Upload the image to Grok
        upload_result = grok_client.upload_file(temp_file_path)
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        return upload_result
    except Exception as e:
        print(f"Error downloading and uploading image: {str(e)}")
        return None

def prepare_messages(messages, grok_client):
    """
    Prepares messages in a format suitable for Grok, with support for image URLs.
    
    Args:
        messages: List of message objects from the OpenAI-compatible request
        grok_client: The Grok client instance for handling image uploads
        
    Returns:
        Tuple of (message_text, file_attachments)
    """
    # For Grok, we'll include previous messages in the content for context
    context_messages = []
    for i, msg in enumerate(messages[:-1]):
        if msg["role"] == "system":
            context_messages.append(f"System: {msg['content']}")
        elif msg["role"] == "user":
            # Handle new message format where content could be a list
            if isinstance(msg['content'], list):
                # Extract text parts from the content list
                text_parts = []
                for item in msg['content']:
                    if item['type'] == 'text':
                        text_parts.append(item['text'])
                msg_content = " ".join(text_parts)
            else:
                msg_content = msg['content']
            context_messages.append(f"User: {msg_content}")
        elif msg["role"] == "assistant":
            context_messages.append(f"Assistant: {msg['content']}")
    
    context_text = "\n\n".join(context_messages)
    
    # Get the last message
    last_message = messages[-1]
    
    # Handle new format for the last message
    file_attachments = []
    if isinstance(last_message.get('content'), list):
        # This is the new format with potential images
        text_parts = []
        for item in last_message['content']:
            if item['type'] == 'text':
                text_parts.append(item['text'])
            elif item['type'] == 'image_url':
                # Process image URL - download and upload it to Grok
                image_url = item['image_url']['url']
                uploaded_file = download_and_upload_image(grok_client, image_url)
                if uploaded_file:
                    file_attachments = uploaded_file  # Set to the response from upload_file
        last_message_content = " ".join(text_parts)
    else:
        last_message_content = last_message['content']
    
    if context_text:
        # If there's context, include it in the message
        message_text = f"{context_text}\n\n{last_message_content}"
    else:
        # Otherwise, just return the content
        message_text = last_message_content
    
    return message_text, file_attachments

def check_special_keywords(message):
    """
    Check if the message contains special keywords and returns the appropriate flags.
    Removes the keywords from the message.
    
    Args:
        message: The user's message string
        
    Returns:
        Tuple of (modified_message, is_deepsearch, is_reasoning)
    """
    is_deepsearch = False
    is_reasoning = False
    modified_message = message
    
    # Convert to lowercase for case-insensitive searching
    message_lower = message.lower()
    
    # Check for "deepsearch" keyword anywhere in the message
    if "deepsearch" in message_lower:
        is_deepsearch = True
        # Remove all instances of "deepsearch" (case insensitive)
        modified_message = re.sub(r'(?i)deepsearch\s*', '', modified_message).strip()
    
    # Check for "reasoning" keyword anywhere in the message
    if "reasoning" in message_lower:
        is_reasoning = True
        # Remove all instances of "reasoning" (case insensitive)
        modified_message = re.sub(r'(?i)reasoning\s*', '', modified_message).strip()
    
    return modified_message, is_deepsearch, is_reasoning

def format_response_to_openai(grok_response, model, is_deepsearch=False):
    """
    Formats a Grok response to match the OpenAI API format.
    
    Args:
        grok_response: The raw response from Grok
        model: The model name to include in the response
        is_deepsearch: If True, filter for messages with messageTag="final"
    """
    # NOTE: Rate limit checks should always happen on the full response
    # before filtering for final tags (this is done in the main function)
    
    # Filter the response to exclude messages tagged as "thinking_trace" and handle deepsearch
    filtered_lines = []
    for line in grok_response.splitlines():
        if line.strip():
            try:
                parsed = json.loads(line)
                result_data = parsed.get("result", {})
                message_tag = result_data.get("messageTag")
                
                # Skip entries with messageTag set to "thinking_trace"
                if message_tag == "thinking_trace":
                    continue
                
                # Skip entries with isThinking flag set to True
                if result_data.get("isThinking") == True:
                    continue
                
                # For deepsearch responses, only include messages tagged as "final"
                if is_deepsearch:
                    if message_tag == "final":
                        filtered_lines.append(line)
                else:
                    filtered_lines.append(line)
            except json.JSONDecodeError:
                # Skip malformed JSON
                continue
    
    # Join the filtered lines back into a single string
    grok_response_filtered = "\n".join(filtered_lines)
    
    # Always use the filtered response (even if empty, this is better than showing thinking traces)
    grok_response = grok_response_filtered
    
    # Parse Grok response
    response_obj = GrokMessages(grok_response)
    full_message = response_obj.get_full_message()
    
    # Create response in OpenAI format
    response_id = f"chatcmpl-{uuid.uuid4()}"
    created_timestamp = int(time.time())
    
    # Very basic token counting (just characters/4)
    prompt_tokens = sum(len(m.get('content', '')) for m in request.json.get('messages', [])) // 4
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

def format_streaming_chunk(text, response_id, model):
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

def process_and_stream_grok_response(grok_message_tokens, response_id, model):
    """
    Process Grok message tokens as they arrive and yield formatted chunks for streaming.
    """
    for message_token in grok_message_tokens:
        # Skip any special tokens or empty tokens
        if not message_token or message_token.startswith("__"):
            continue
            
        # Format the token as an OpenAI compatible streaming chunk
        yield format_streaming_chunk(message_token, response_id, model)
    
    # Send the final chunk with finish_reason
    yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
    
    # End the stream
    yield "data: [DONE]\n\n"

@app.route('/v1/models', methods=['GET'])
def list_models():
    """List available models endpoint."""
    models = [
        {"id": "grok-1", "object": "model", "created": int(time.time()), "owned_by": "Grok"},
        {"id": "grok-2", "object": "model", "created": int(time.time()), "owned_by": "Grok"},
        {"id": "grok-3", "object": "model", "created": int(time.time()), "owned_by": "Grok"}
    ]
    return jsonify({"object": "list", "data": models})

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    OpenAI-compatible chat completions endpoint.
    """
    try:
        data = request.json
        
        # Get request parameters
        messages = data.get('messages', [])
        model = data.get('model', 'grok-3')
        stream = data.get('stream', False)
        
        # Map model to Grok model
        grok_model = get_grok_model(model)
        
        # Maximum number of retry attempts
        max_retries = len(COOKIES_LIST)
        retries = 0
        success = False
        
        while not success and retries < max_retries:
            # Create a new conversation
            try:
                grok_client.create_conversation()
            except requests.exceptions.Timeout:
                # Specific handling for timeout errors
                print(f"Timeout creating conversation, rotating credentials (attempt {retries + 1})")
                rotate_credentials()
                retries += 1
                continue
            except Exception as e:
                print(f"Error creating conversation: {str(e)}, rotating credentials (attempt {retries + 1})")
                rotate_credentials()
                retries += 1
                continue
            
            # First, check for keywords in just the last message (latest query)
            last_message = messages[-1]
            last_message_content = ""
            
            # Extract the content from the last message
            if isinstance(last_message.get('content'), list):
                text_parts = []
                for item in last_message['content']:
                    if item['type'] == 'text':
                        text_parts.append(item['text'])
                last_message_content = " ".join(text_parts)
            else:
                last_message_content = last_message.get('content', '')
            
            # Check for special keywords only in the latest query
            modified_query, is_deepsearch, is_reasoning = check_special_keywords(last_message_content)
            
            # Create a copy of the messages with the modified last message
            modified_messages = messages.copy()
            if isinstance(modified_messages[-1].get('content'), list):
                # For multimodal messages, update just the text parts
                for i, item in enumerate(modified_messages[-1]['content']):
                    if item['type'] == 'text':
                        modified_messages[-1]['content'][i]['text'] = modified_query
                        # Only modify the first text part, leave others unchanged
                        break
            else:
                modified_messages[-1]['content'] = modified_query
            
            # Process messages to fit Grok's format with potential image attachments
            final_message, file_attachments = prepare_messages(modified_messages, grok_client)
            
            # Prepare the request data with special flags if keywords detected
            request_data = grok_client.create_message(
                grok_model,
                isDeepsearch=is_deepsearch,
                isReasoning=is_reasoning
            )
            
            # Add the user message with any file attachments
            grok_client.add_user_message(request_data, final_message, file_attachments=file_attachments)
            
            # Generate a response ID
            response_id = f"chatcmpl-{uuid.uuid4()}"
            
            if stream:
                # Create a wrapper function to handle rotation
                def stream_with_rotation():
                    nonlocal success, retries
                    
                    # Initialize the stream inside the function to avoid variable scope issues
                    current_stream = grok_client.send_streaming(request_data, filter_final_only=is_deepsearch)
                    
                    try:
                        # Try to stream from the current client
                        rate_limited = False
                        timeout_occurred = False
                        
                        # Buffer for accumulating response lines to check for rate limits
                        response_buffer = []
                        
                        for message_token in current_stream:
                            # Check for special tokens from the Grok class
                            if message_token == "__RATE_LIMITED__":
                                print(f"Rate limited in streaming mode (detected by Grok class)")
                                rate_limited = True
                                break
                            elif message_token == "__TIMEOUT__":
                                print(f"Timeout in streaming mode (detected by Grok class)")
                                timeout_occurred = True
                                break
                            elif message_token.startswith("__ERROR__"):
                                print(f"Error in streaming mode: {message_token}")
                                timeout_occurred = True  # Treat as timeout for retry purposes
                                break
                            
                            # Skip thinking traces
                            if isinstance(message_token, str) and message_token.strip():
                                try:
                                    # Try to parse as JSON to check for thinking traces
                                    parsed = json.loads(message_token)
                                    result_data = parsed.get("result", {})
                                    if result_data.get("messageTag") == "thinking_trace" or result_data.get("isThinking") == True:
                                        continue
                                except:
                                    # Not JSON or couldn't parse, assume it's safe to keep
                                    pass
                            
                            # Add to buffer for checking
                            response_buffer.append(message_token)
                            
                            # If buffer is large enough, start yielding
                            if len(response_buffer) > 5:
                                yield format_streaming_chunk(response_buffer.pop(0), response_id, model)
                        
                        # If rate limited or timeout, try again with a new credential set
                        if rate_limited or timeout_occurred:
                            retries += 1
                            if retries < max_retries:
                                print(f"Connection issue, rotating credentials (attempt {retries})")
                                rotate_credentials()
                                
                                # Create a new conversation with new credentials
                                try:
                                    grok_client.create_conversation()
                                    new_request_data = grok_client.create_message(
                                        grok_model,
                                        isDeepsearch=is_deepsearch,
                                        isReasoning=is_reasoning
                                    )
                                    grok_client.add_user_message(new_request_data, final_message, file_attachments=file_attachments)
                                    
                                    # Start a new stream with the new request
                                    new_stream = grok_client.send_streaming(new_request_data, filter_final_only=is_deepsearch)
                                    
                                    # Process the new stream directly (no recursion)
                                    try:
                                        for message_token in new_stream:
                                            # Check for special tokens
                                            if message_token == "__RATE_LIMITED__" or message_token == "__TIMEOUT__" or message_token.startswith("__ERROR__"):
                                                print(f"Error in new stream: {message_token}")
                                                raise Exception(f"Error in new stream: {message_token}")
                                            
                                            # Skip thinking traces
                                            if isinstance(message_token, str) and message_token.strip():
                                                try:
                                                    # Try to parse as JSON to check for thinking traces
                                                    parsed = json.loads(message_token)
                                                    result_data = parsed.get("result", {})
                                                    if result_data.get("messageTag") == "thinking_trace" or result_data.get("isThinking") == True:
                                                        continue
                                                except:
                                                    # Not JSON or couldn't parse, assume it's safe to keep
                                                    pass
                                            
                                            yield format_streaming_chunk(message_token, response_id, model)
                                        
                                        # Send the final chunk after processing all tokens
                                        yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
                                        yield "data: [DONE]\n\n"
                                        success = True
                                    except Exception as e:
                                        print(f"Error in new stream after rotation: {str(e)}")
                                        # Try again with another credential set if available
                                        if retries < max_retries - 1:
                                            retries += 1
                                            rotate_credentials()
                                            # Create another conversation with the next set of credentials
                                            try:
                                                grok_client.create_conversation()
                                                newer_request_data = grok_client.create_message(
                                                    grok_model,
                                                    isDeepsearch=is_deepsearch,
                                                    isReasoning=is_reasoning
                                                )
                                                grok_client.add_user_message(newer_request_data, final_message, file_attachments=file_attachments)
                                                
                                                # Try with the newest credentials
                                                newest_stream = grok_client.send_streaming(newer_request_data, filter_final_only=is_deepsearch)
                                                for token in newest_stream:
                                                    # Check for special tokens
                                                    if token == "__RATE_LIMITED__" or token == "__TIMEOUT__" or token.startswith("__ERROR__"):
                                                        print(f"Error in newest stream: {token}")
                                                        # If we hit another issue, just send what we have and end
                                                        break
                                                    
                                                    # Skip thinking traces
                                                    if isinstance(token, str) and token.strip():
                                                        try:
                                                            # Try to parse as JSON to check for thinking traces
                                                            parsed = json.loads(token)
                                                            result_data = parsed.get("result", {})
                                                            if result_data.get("messageTag") == "thinking_trace" or result_data.get("isThinking") == True:
                                                                continue
                                                        except:
                                                            # Not JSON or couldn't parse, assume it's safe to keep
                                                            pass
                                                    
                                                    yield format_streaming_chunk(token, response_id, model)
                                                
                                                yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
                                                yield "data: [DONE]\n\n"
                                            except Exception as newest_error:
                                                # Out of options, return an error
                                                error_msg = f"Error after multiple retries: {str(newest_error)}"
                                                yield format_streaming_chunk(error_msg, response_id, model)
                                                yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
                                                yield "data: [DONE]\n\n"
                                        else:
                                            # Out of retries, return an error message
                                            error_msg = f"Error after {max_retries} retries: {str(e)}"
                                            yield format_streaming_chunk(error_msg, response_id, model)
                                            yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
                                            yield "data: [DONE]\n\n"
                                except Exception as e:
                                    print(f"Failed to create new conversation: {str(e)}")
                                    # Out of options, return an error
                                    error_msg = f"Failed to connect after multiple attempts: {str(e)}"
                                    yield format_streaming_chunk(error_msg, response_id, model)
                                    yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
                                    yield "data: [DONE]\n\n"
                            else:
                                # Out of retries, return what we have
                                for chunk in response_buffer:
                                    # Skip thinking traces one more time
                                    if isinstance(chunk, str) and chunk.strip():
                                        try:
                                            # Try to parse as JSON to check for thinking traces
                                            parsed = json.loads(chunk)
                                            result_data = parsed.get("result", {})
                                            if result_data.get("messageTag") == "thinking_trace" or result_data.get("isThinking") == True:
                                                continue
                                        except:
                                            # Not JSON or couldn't parse, assume it's safe to keep
                                            pass
                                    yield format_streaming_chunk(chunk, response_id, model)
                                yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
                                yield "data: [DONE]\n\n"
                        else:
                            # No rate limiting, return remaining buffer
                            for chunk in response_buffer:
                                # Skip thinking traces one more time
                                if isinstance(chunk, str) and chunk.strip():
                                    try:
                                        # Try to parse as JSON to check for thinking traces
                                        parsed = json.loads(chunk)
                                        result_data = parsed.get("result", {})
                                        if result_data.get("messageTag") == "thinking_trace" or result_data.get("isThinking") == True:
                                            continue
                                    except:
                                        # Not JSON or couldn't parse, assume it's safe to keep
                                        pass
                                yield format_streaming_chunk(chunk, response_id, model)
                            
                            # Send the final chunk
                            yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
                            yield "data: [DONE]\n\n"
                            
                            success = True
                    except Exception as e:
                        print(f"Error in streaming: {str(e)}")
                        # If there's an error, try with the next credentials
                        retries += 1
                        if retries < max_retries:
                            rotate_credentials()
                            
                            # Create a new conversation with new credentials
                            try:
                                grok_client.create_conversation()
                                retry_request_data = grok_client.create_message(
                                    grok_model,
                                    isDeepsearch=is_deepsearch,
                                    isReasoning=is_reasoning
                                )
                                grok_client.add_user_message(retry_request_data, final_message, file_attachments=file_attachments)
                                
                                # Create a new stream for the retry
                                retry_stream = grok_client.send_streaming(retry_request_data, filter_final_only=is_deepsearch)
                                
                                # Process the retry stream
                                try:
                                    for token in retry_stream:
                                        # Check for special tokens
                                        if token == "__RATE_LIMITED__" or token == "__TIMEOUT__" or token.startswith("__ERROR__"):
                                            print(f"Error in retry stream: {token}")
                                            # If we hit another issue, just end the stream
                                            break
                                        
                                        # Skip thinking traces one more time
                                        if isinstance(token, str) and token.strip():
                                            try:
                                                # Try to parse as JSON to check for thinking traces
                                                parsed = json.loads(token)
                                                result_data = parsed.get("result", {})
                                                if result_data.get("messageTag") == "thinking_trace" or result_data.get("isThinking") == True:
                                                    continue
                                            except:
                                                # Not JSON or couldn't parse, assume it's safe to keep
                                                pass
                                        
                                        yield format_streaming_chunk(token, response_id, model)
                                    
                                    yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
                                    yield "data: [DONE]\n\n"
                                except Exception as retry_error:
                                    # Out of retries or another error
                                    error_msg = f"Error during retry: {str(retry_error)}"
                                    yield format_streaming_chunk(error_msg, response_id, model)
                                    yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
                                    yield "data: [DONE]\n\n"
                            except Exception as create_error:
                                # Failed to create new conversation
                                error_msg = f"Failed to create new conversation: {str(create_error)}"
                                yield format_streaming_chunk(error_msg, response_id, model)
                                yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
                                yield "data: [DONE]\n\n"
                        else:
                            # Out of retries, return an error message
                            error_msg = f"Error after {max_retries} retries: {str(e)}"
                            yield format_streaming_chunk(error_msg, response_id, model)
                            yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
                            yield "data: [DONE]\n\n"
                
                # Return the streaming response
                return Response(
                    stream_with_context(stream_with_rotation()),
                    content_type='text/event-stream'
                )
            else:
                # Non-streaming mode
                while retries < max_retries:
                    try:
                        # Send the message and get the response
                        grok_response = grok_client.send(request_data)
                        
                        # Check if we hit a rate limit
                        if is_rate_limited(grok_response):
                            limit_types = detect_rate_limit_pattern(grok_response)
                            print(f"Rate limited ({', '.join(limit_types)}), rotating credentials (attempt {retries + 1})")
                            rotate_credentials()
                            retries += 1
                            
                            # Create a new conversation with new credentials
                            try:
                                grok_client.create_conversation()
                                request_data = grok_client.create_message(
                                    grok_model,
                                    isDeepsearch=is_deepsearch,
                                    isReasoning=is_reasoning
                                )
                                grok_client.add_user_message(request_data, final_message, file_attachments=file_attachments)
                            except Exception as create_error:
                                print(f"Error creating new conversation: {str(create_error)}")
                                # We'll retry in the next loop iteration
                            continue
                        
                        # Format the response to match OpenAI's format
                        openai_response = format_response_to_openai(grok_response, model, is_deepsearch=is_deepsearch)
                        success = True
                        return jsonify(openai_response)
                    
                    except requests.exceptions.Timeout:
                        print(f"Timeout error, rotating credentials (attempt {retries + 1})")
                        retries += 1
                        if retries < max_retries:
                            rotate_credentials()
                        else:
                            break
                    
                    except Exception as e:
                        print(f"Error: {str(e)}")
                        retries += 1
                        if retries < max_retries:
                            rotate_credentials()
                        else:
                            break
                
                # If we're here, we've exhausted all retries
                return jsonify({
                    "error": {
                        "message": f"Failed after {max_retries} attempts with different credentials",
                        "type": "connection_error",
                        "code": 500
                    }
                }), 500
    
    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "api_error",
                "code": 500
            }
        }), 500

@app.route('/', methods=['GET'])
def home():
    """Home endpoint."""
    return jsonify({
        "status": "ok",
        "message": "GrokAI OpenAI-compatible API is running",
        "endpoints": {
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions"
        },
        "credentials": {
            "sets_available": len(COOKIES_LIST),
            "current_set": current_cred_index + 1
        },
        "timeout_settings": {
            "connect_timeout": API_CONNECT_TIMEOUT,
            "read_timeout": API_READ_TIMEOUT
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)