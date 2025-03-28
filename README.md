# Grok OpenAI-Compatible API

An asynchronous Python library and FastAPI server for interacting with Grok AI using X's user account API. This project provides an OpenAI-compatible `/v1/chat/completions` endpoint, enabling integration with existing tools, along with features like automatic credential rotation to handle rate limits.

**Note:** This uses the unofficial X user account API, not the paid enterprise API. Grok AI access is required (currently available to X Premium subscribers).

## Features

-   🚀 **OpenAI-Compatible API:** Run a local server (`main.py`) that mimics the OpenAI `/v1/chat/completions` endpoint for Grok.
-   🔄 **Credential Rotation:** Automatically cycles through multiple X account credentials (`.env` configured) to mitigate rate limits.
-   ⚡ **Asynchronous:** Built with `asyncio`, `aiohttp`, and `FastAPI` for efficient non-blocking operations.
-   📄 **Streaming Support:** Handles OpenAI-style Server-Sent Events (SSE) for streaming responses.
-   🖼️ **Image Support:** Send images (via URL or base64 data URI) in chat messages, compatible with the OpenAI vision format.
-   🤖 **Grok Integration:** Leverages the underlying `grok.py` library for core interactions.
-   ⚙️ **Configurable:** Easily configure multiple accounts via `.env`.
-   🔍 **Special Modes:** Activate `deepsearch` or `reasoning` modes by including keywords in your prompt.

## Prerequisites

-   Python 3.8 or higher
-   One or more valid X (formerly Twitter) accounts with **X Premium** (for Grok access).
-   Authentication tokens (Cookies, CSRF Token, Bearer Token) for **each** account you want to use for rotation.

## Installation

1.  Clone the repository:
    ```sh
    git clone https://github.com/vibheksoni/GrokAiChat.git
    cd grok_ai_chat
    ```
    *(Note: If your local directory name differs, use that name instead of `grok_ai_chat`)*

2.  Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
    *(This installs `fastapi`, `uvicorn`, `requests`, `python-dotenv`, `aiohttp`, `aiofiles`, etc.)*

## Configuration: Multiple Accounts for Rotation

To enable credential rotation and mitigate rate limits, you need to provide credentials for **multiple** X accounts.

1.  Create a `.env` file in the project root (`grok_ai_chat/`).

2.  Add credentials for each account using numbered suffixes (up to 5 are checked by default in `main.py`, but you can add more):

    ```dotenv
    # Account 1
    COOKIES_1="your_account_1_cookie_string"
    X_CSRF_TOKEN_1="your_account_1_csrf_token"
    BEARER_TOKEN_1="your_account_1_bearer_token"

    # Account 2 (Optional, but recommended for rotation)
    COOKIES_2="your_account_2_cookie_string"
    X_CSRF_TOKEN_2="your_account_2_csrf_token"
    BEARER_TOKEN_2="your_account_2_bearer_token"

    # Account 3 (Optional)
    COOKIES_3="your_account_3_cookie_string"
    X_CSRF_TOKEN_3="your_account_3_csrf_token"
    BEARER_TOKEN_3="your_account_3_bearer_token"

    # ... add more accounts as needed (COOKIES_4, etc.)

    # Optional: Server configuration
    # HOST=0.0.0.0
    # PORT=5000
    ```

### How to Get Your Credentials (For Each Account)

1.  **Log into X.com** with the desired account.
2.  Open your browser's **Developer Tools** (usually F12).
3.  Navigate to the **Network** tab.
4.  Go to the Grok chat interface on X.com (e.g., `https://x.com/i/grok`).
5.  Send a message in the Grok chat.
6.  Find requests made to `x.com` or `api.x.com` in the Network tab. Look for requests like `CreateGrokConversation` or `add_response.json`.
7.  **Inspect Headers:**
    *   **Cookies**: Find the `cookie` request header. Copy the **entire** long string value.
    *   **X-CSRF-Token**: Find the `x-csrf-token` request header. Copy its value (usually a 32-character alphanumeric string).
    *   **Bearer Token**: Find the `authorization` request header. Copy the token part after `Bearer ` (it usually starts with `AAAA...`).
8.  Paste these values into your `.env` file for the corresponding account number (`_1`, `_2`, etc.).

## Running the API Server

Once configured, run the FastAPI server:

```sh
uvicorn main:app --reload --host 0.0.0.0 --port 5000
```

-   `--reload`: Automatically restarts the server when code changes (useful for development).
-   `--host 0.0.0.0`: Makes the server accessible from other devices on your network.
-   `--port 5000`: Specifies the port (can be changed, ensure it matches `.env` if set).

The API will be available at `http://localhost:5000` (or the host/port you configured).

## Using the OpenAI-Compatible API

You can now use any OpenAI-compatible client library or tool (like `curl`, Python `openai` library, UI clients) to interact with your local Grok API server.

**API Endpoint:** `http://localhost:5000/v1/chat/completions`
**Model Name:** `grok-3` (This is the only model ID currently supported by the adapter)

### Example: `curl` Request (Text)

```sh
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "grok-3",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain the concept of asynchronous programming in Python."}
    ],
    "stream": false
  }'
```

### Example: `curl` Request (Streaming)

```sh
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "grok-3",
    "messages": [
      {"role": "user", "content": "Write a short poem about the moon."}
    ],
    "stream": true
  }'
```

### Example: `curl` Request (Image URL)

```sh
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "grok-3",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat_paw.jpg/800px-Cat_paw.jpg"}}
        ]
      }
    ],
    "stream": false
  }'
```

### Example: Python `openai` Client

```python
import openai

# Point the client to your local server
client = openai.OpenAI(
    base_url="http://localhost:5000/v1",
    api_key="not-needed" # API key is not required by this server
)

try:
    completion = client.chat.completions.create(
        model="grok-3",
        messages=[
            {"role": "system", "content": "You are Grok."},
            {"role": "user", "content": "Tell me a fun fact about space."}
        ],
        stream=False # Set to True for streaming
    )
    print(completion.choices[0].message.content)

except openai.APIConnectionError as e:
    print(f"Failed to connect to the server: {e}")
except openai.APIStatusError as e:
    print(f"API Error: Status {e.status_code}, Response: {e.response}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Example with image URL
try:
    completion_with_image = client.chat.completions.create(
      model="grok-3",
      messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": "Describe the main object in this image."},
            {"type": "image_url", "image_url": {"url": "https://www.example.com/image.jpg"}}, # Replace with actual URL
          ],
        }
      ]
    )
    print(completion_with_image.choices[0].message.content)
except Exception as e:
    print(f"An error occurred with image request: {e}")

```

## Using the `Grok` Library Directly (Advanced)

While the primary interface is the API server, you can still use the underlying asynchronous `Grok` library directly in your Python code.

```python
import asyncio
import os
from dotenv import load_dotenv
from grok import Grok, GrokMessages, AllCredentialsLimitedError

async def main():
    load_dotenv()

    # Load credentials into the required list format
    credential_sets = []
    for i in range(1, 6): # Check for up to 5 accounts
        cookie = os.getenv(f"COOKIES_{i}")
        csrf_token = os.getenv(f"X_CSRF_TOKEN_{i}")
        bearer_token = os.getenv(f"BEARER_TOKEN_{i}")
        if all([cookie, csrf_token, bearer_token]):
            credential_sets.append({
                "cookies": cookie,
                "csrf": csrf_token,
                "bearer": bearer_token
            })

    if not credential_sets:
        print("Error: No complete credential sets found in .env")
        return

    try:
        grok_client = Grok(credential_sets=credential_sets)

        # 1. Create a conversation
        print("Creating conversation...")
        conversation_id = await grok_client.create_new_conversation()
        print(f"Conversation ID: {conversation_id}")

        # 2. Prepare message payload
        payload = grok_client.create_message_payload(
            model_name="grok-3", # Use the appropriate model ID
            conversation_id=conversation_id
        )

        # 3. Add user message
        grok_client.add_user_message_to_payload(
            request_payload=payload,
            message="Hello Grok! Write a haiku about code."
            # file_attachments=... # Optional: Use upload_file first
        )

        # 4. Send (non-streaming)
        print("Sending message...")
        response_text = await grok_client.send(payload)

        # 5. Parse and print response
        messages = GrokMessages(response_text)
        print("\nFull Response:")
        print(messages.get_full_message())

        # --- Example: Streaming ---
        print("\n--- Streaming Example ---")
        payload_stream = grok_client.create_message_payload("grok-3", conversation_id)
        grok_client.add_user_message_to_payload(payload_stream, "Tell me another short poem.")

        async for line in grok_client.send_stream(payload_stream):
            try:
                parsed = GrokMessages(line) # Parse each line individually
                token = parsed.get_full_message()
                if token:
                    print(token, end='', flush=True)
            except Exception: # Handle potential JSON errors on partial lines if needed
                pass
        print("\n--- Stream End ---")


    except AllCredentialsLimitedError:
        print("Error: All provided credentials hit the rate limit.")
    except (ConnectionError, IOError) as e:
        print(f"Error communicating with Grok: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

## ⚠️ Important Legal Warnings & Rate Limiting

1.  **Terms of Service**: Using this tool **may violate** X's Terms of Service regarding automated access and API usage. **Use entirely at your own risk.** The author assumes no responsibility for account restrictions or bans.
2.  **Account Security**:
    *   Never share your `.env` file or credentials publicly.
    *   The credential rotation helps, but excessive requests can still lead to temporary or permanent limitations on your accounts. Use responsibly.
3.  **Compliance**:
    *   This tool is intended for personal, experimental, and educational purposes.
    *   Commercial use is strongly discouraged and likely violates X's terms.
    *   You are solely responsible for ensuring your use complies with all applicable laws and X's policies.
4.  **Rate Limiting Mitigation**:
    *   The credential rotation significantly helps bypass individual account rate limits.
    *   However, X might implement broader IP-based or behavior-based limits.
    *   Avoid extremely high request volumes. Add delays in client applications if needed.

## API Documentation (`grok.py`)

-   `Grok(credential_sets: List[Dict[str, str]])`: Initializes the client with multiple account credentials. Handles rotation.
    -   `create_new_conversation()`: Starts a new chat session (async).
    -   `upload_file(file_path: str)`: Uploads a local file (async).
    -   `create_message_payload(...)`: Creates the JSON structure for sending a message.
    -   `add_user_message_to_payload(...)`: Adds the user's text and attachments to the payload.
    *   `send(payload: dict)`: Sends the message and returns the full response text (async).
    *   `send_stream(payload: dict)`: Sends the message and yields response lines asynchronously (async generator).
-   `GrokMessages(raw_data: str)`: Parses the raw JSONL response string from Grok.
    -   `get_full_message()`: Returns the concatenated message content.
    -   `is_limiter_response()`: Checks if the response indicates a rate limit.
    -   Other methods to extract specific parts of the response (web results, suggestions, etc.).
-   `AllCredentialsLimitedError`: Exception raised when all accounts in the rotation hit a rate limit.

Full details are available in the code comments within `grok.py`.

## Contributing

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

Distributed under the MIT License. See `LICENSE` file for details.

## Author

**Vibhek Soni**
- GitHub: [@vibheksoni](https://github.com/vibheksoni)
- Project Link: [https://github.com/vibheksoni/GrokAiChat](https://github.com/vibheksoni/GrokAiChat)

**anojndr**
- GitHub: [@anojndr](https://github.com/anojndr)
- Project Link: [GrokAiChat](https://github.com/anojndr/GrokAiChat)