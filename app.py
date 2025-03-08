import os
import time
import json
import uuid
import re
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

app = Flask(__name__)
CORS(app)

# Current credential index
current_cred_index = 0

# Initialize the Grok client with the first set of credentials
grok_client = Grok(BEARER_TOKEN_LIST[current_cred_index], 
                  X_CSRF_TOKEN_LIST[current_cred_index], 
                  COOKIES_LIST[current_cred_index])

# Model mappings
MODEL_MAPPINGS = {
    "gpt-3.5-turbo": "grok-1",
    "gpt-4": "grok-2",
    "gpt-4-turbo": "grok-3",
    "gpt-4-vision": "grok-3",  # Assuming grok-3 can handle images
    "grok-1": "grok-1",
    "grok-2": "grok-2",
    "grok-3": "grok-3"
}

def rotate_credentials():
    """Rotate to the next set of credentials."""
    global current_cred_index, grok_client
    
    # Move to next credential set
    current_cred_index = (current_cred_index + 1) % len(COOKIES_LIST)
    
    # Reinitialize grok client with new credentials
    grok_client = Grok(BEARER_TOKEN_LIST[current_cred_index], 
                      X_CSRF_TOKEN_LIST[current_cred_index], 
                      COOKIES_LIST[current_cred_index])
    
    print(f"Rotated to credential set {current_cred_index + 1}/{len(COOKIES_LIST)}")
    return grok_client

def is_rate_limited(response_text):
    """Check if the response indicates a rate limit."""
    try:
        # Check for the specific rate limit error message
        if "You've reached your limit of 15 Grok questions per 2 hours for now" in response_text:
            return True
        
        # Try parsing the response as JSON
        for line in response_text.splitlines():
            if line.strip():
                parsed = json.loads(line)
                result = parsed.get("result", {})
                
                # Check for rate limit message in parsed message content
                message = result.get("message", "")
                if "You've reached your limit of 15 Grok questions per 2 hours for now" in message:
                    return True
    except (json.JSONDecodeError, KeyError):
        pass
    
    return False

def get_grok_model(openai_model):
    """Maps OpenAI model names to Grok model names."""
    return MODEL_MAPPINGS.get(openai_model, "grok-3")  # Default to grok-3

def prepare_messages(messages):
    """Prepares messages in a format suitable for Grok."""
    # For Grok, we just need the last message content
    # But we'll include previous messages in the content for context
    
    context_messages = []
    for i, msg in enumerate(messages[:-1]):
        if msg["role"] == "system":
            context_messages.append(f"System: {msg['content']}")
        elif msg["role"] == "user":
            context_messages.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            context_messages.append(f"Assistant: {msg['content']}")
    
    context_text = "\n\n".join(context_messages)
    
    # Get the last message
    last_message = messages[-1]
    
    if context_text:
        # If there's context, include it in the message
        return f"{context_text}\n\n{last_message['content']}"
    else:
        # Otherwise, just return the content
        return last_message['content']

def format_response_to_openai(grok_response, model):
    """
    Formats a Grok response to match the OpenAI API format.
    """
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
            grok_client.create_conversation()
            
            # Prepare the request data
            request_data = grok_client.create_message(grok_model)
            
            # Process messages to fit Grok's format
            final_message = prepare_messages(messages)
            
            # Add the user message
            grok_client.add_user_message(request_data, final_message)
            
            # Generate a response ID
            response_id = f"chatcmpl-{uuid.uuid4()}"
            
            if stream:
                # For streaming, we need to use a generator and check for rate limits
                grok_message_stream = grok_client.send_streaming(request_data)
                
                # Create a wrapper function to handle rotation
                def stream_with_rotation():
                    nonlocal success, retries
                    try:
                        # Try to stream from the current client
                        rate_limited = False
                        
                        # Buffer for accumulating response lines to check for rate limits
                        response_buffer = []
                        
                        for message_token in grok_message_stream:
                            # Check if this chunk contains a rate limit message
                            if "You've reached your limit of 15 Grok questions per 2 hours for now" in message_token:
                                rate_limited = True
                                break
                            
                            # Add to buffer for checking
                            response_buffer.append(message_token)
                            
                            # If buffer is large enough, start yielding
                            if len(response_buffer) > 5:
                                yield format_streaming_chunk(response_buffer.pop(0), response_id, model)
                        
                        # If rate limited, try again with a new credential set
                        if rate_limited:
                            retries += 1
                            if retries < max_retries:
                                print(f"Rate limited, rotating credentials (attempt {retries})")
                                rotate_credentials()
                                
                                # Recursive call to try with new credentials
                                for chunk in stream_with_rotation():
                                    yield chunk
                            else:
                                # Out of retries, return what we have
                                for chunk in response_buffer:
                                    yield format_streaming_chunk(chunk, response_id, model)
                                yield format_streaming_chunk("", response_id, model).replace('"finish_reason": null', '"finish_reason": "stop"')
                                yield "data: [DONE]\n\n"
                        else:
                            # No rate limiting, return remaining buffer
                            for chunk in response_buffer:
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
                            for chunk in stream_with_rotation():
                                yield chunk
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
                            print(f"Rate limited, rotating credentials (attempt {retries + 1})")
                            rotate_credentials()
                            retries += 1
                            
                            # Create a new conversation with new credentials
                            grok_client.create_conversation()
                            request_data = grok_client.create_message(grok_model)
                            grok_client.add_user_message(request_data, final_message)
                            continue
                        
                        # Format the response to match OpenAI's format
                        openai_response = format_response_to_openai(grok_response, model)
                        success = True
                        return jsonify(openai_response)
                    
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
                        "type": "rate_limit_error",
                        "code": 429
                    }
                }), 429
    
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
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=True)