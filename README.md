# GrokAiChat

A Python project for interacting with Grok AI through X's user account API. This project provides both a direct library interface and an OpenAI-compatible API server, allowing you to use Grok AI with any OpenAI client library. Note: This uses the account user API, not the paid enterprise API. Grok AI is free for all X (Twitter) members.

## Features

- 🤖 Full Grok AI API integration
- 🔄 OpenAI-compatible API server
- 📁 File upload support
- 💬 Conversation management
- 🌊 Streaming response support
- 🔄 Multiple account credential rotation
- 🐳 Docker deployment support
- 🛠️ Easy-to-use interface

## Prerequisites

- Python 3.8 or higher
- A valid X (formerly Twitter) account
- Grok AI access (free for all X members)
- Your account's authentication tokens
- Docker (optional, for containerized deployment)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/vibheksoni/GrokAiChat.git
    cd GrokAiChat
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Configuration

1. Create a `.env` file in the project root:
    ```dotenv
    # Raw cookies string (Example: "cookie1=value1; cookie2=value2")
    COOKIES=""
    X_CSRF_TOKEN=""
    BEARER_TOKEN=""
    
    # Optional: Additional account credentials for rotation
    COOKIES_2=""
    X_CSRF_TOKEN_2=""
    BEARER_TOKEN_2=""
    
    # Optional: API server port (default: 5000)
    PORT=5000
    
    # Optional: API timeout settings (in seconds)
    API_CONNECT_TIMEOUT=10.0
    API_READ_TIMEOUT=30.0
    ```

2. To obtain your tokens:
    - Log into X.com
    - Open Developer Tools (F12)
    - Create a Grok chat
    - Find the tokens in the Network tab request headers

## Account Requirements & Credentials

⚠️ **IMPORTANT**: You need:
1. A standard X account (Grok AI is free for all X members)
2. The following credentials from your account:

### How to Get Your Credentials

1. **Cookies**:
   - Log into X.com
   - Open Developer Tools (F12) → Network tab
   - Interact with Grok
   - Find any request to x.com
   - Copy the entire `cookie` header value

2. **X-CSRF-Token**:
   - In the same Network tab
   - Look for `x-csrf-token` in request headers
   - It's usually a 32-character string

3. **Bearer Token**:
   - Find any request header containing `authorization`
   - Copy the token after "Bearer "
   - Usually starts with "AAAA..."

Store these in your `.env` file as shown in the Configuration section above.

## Multiple Account Support

You can configure multiple X accounts for credential rotation to handle rate limits. Just add numbered credentials to your `.env` file:

```dotenv
# First account
COOKIES=""
X_CSRF_TOKEN=""
BEARER_TOKEN=""

# Second account
COOKIES_2=""
X_CSRF_TOKEN_2=""
BEARER_TOKEN_2=""

# Third account
COOKIES_3=""
X_CSRF_TOKEN_3=""
BEARER_TOKEN_3=""
```

The API server will automatically rotate credentials when rate limits are encountered.

## Usage Options

### 1. Direct Library Usage

```python
from grok import Grok, GrokMessages
from dotenv import load_dotenv
import os

load_dotenv()
grok = Grok(
    os.getenv("BEARER_TOKEN"),
    os.getenv("X_CSRF_TOKEN"),
    os.getenv("COOKIES")
)

# Create a conversation
grok.create_conversation()

# Send a message
request = grok.create_message("grok-2")
grok.add_user_message(request, "Hello, Grok!")
response = grok.send(request)

# Parse and print response
messages = GrokMessages(response)
print(messages.get_full_message())
```

### 2. API Server Usage

#### Start the API Server

```bash
# Using Python directly
python app.py

# Using Docker
docker-compose up -d
```

#### Using with OpenAI Client

```python
from openai import OpenAI

# Initialize with your API server address
client = OpenAI(
    api_key="dummy-key",  # Not actually used, but required
    base_url="http://localhost:5000/v1"  # Your GrokAI API server
)

# Use just like standard OpenAI API
response = client.chat.completions.create(
    model="grok-3",  # or use OpenAI model names like "gpt-4" 
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about yourself"}
    ]
)

print(response.choices[0].message.content)
```

#### Using with Direct API Calls

```python
import requests

# Define the API endpoint
endpoint = "http://localhost:5000/v1/chat/completions"

# Create the request payload
payload = {
    "model": "grok-3",
    "messages": [
        {"role": "user", "content": "Write a short poem about AI."}
    ],
    "stream": False
}

# Send the request
response = requests.post(endpoint, json=payload)
print(response.json()["choices"][0]["message"]["content"])
```

#### Special Features

The API server supports special keywords in your prompts:

- Add "deepsearch" to trigger Grok's deepsearch
- Add "reasoning" to use Grok's step-by-step reasoning mode

Example:
```python
client.chat.completions.create(
    model="grok-3",
    messages=[
        {"role": "user", "content": "deepsearch What were the major news events of March 2025?"}
    ]
)
```

## Docker Deployment

1. Build and start the container:
    ```bash
    docker-compose up -d
    ```

2. Check logs:
    ```bash
    docker-compose logs -f
    ```

3. Stop the container:
    ```bash
    docker-compose down
    ```

## Advanced Usage Examples

Check the `examples/` directory for more advanced use cases:
- Basic chat interaction (`chat.py`)
- File attachments (`chatwithfiles.py`)
- OpenAI-compatible API usage (`example_usage.py`)

## API Documentation

### Main Classes and Components

- `Grok`: Core interface for API interactions
- `GrokMessages`: Response parser and message handler
- `app.py`: OpenAI-compatible API server

### OpenAI Compatibility

The API server supports the following OpenAI-compatible endpoints:

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Create chat completions
- `GET /` - Server status endpoint

Model mapping:
- `gpt-3.5-turbo` → `grok-1`
- `gpt-4` → `grok-2`
- `gpt-4-turbo` → `grok-3`
- `gpt-4-vision` → `grok-3`

## ⚠️ Important Legal Warnings

1. **Terms of Service**: This project **may violate** X's Terms of Service. Use at your own risk.
2. **Account Security**: 
   - Never share your credentials
   - Avoid excessive requests
   - Use reasonable rate limits
3. **Compliance**:
   - This tool is for educational purposes only
   - Commercial use may violate X's terms
   - You are responsible for how you use this code

## Rate Limiting

To avoid account flags:
- Limit requests to reasonable human speeds
- Add delays between messages
- Consider using multiple accounts with the rotation feature
- Don't automate mass messaging

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Legal Notice

⚠️ **Disclaimer**: This project is for educational purposes only. Users are responsible for ensuring their usage complies with X's terms of service.

## License

[MIT License](LICENSE) - See license file for details.

## Authors

**Vibhek Soni**
- Age: 19
- GitHub: [@vibheksoni](https://github.com/vibheksoni)
- Project Link: [GrokAiChat](https://github.com/vibheksoni/GrokAiChat)

**anojndr**
- GitHub: [@anojndr](https://github.com/anojndr)