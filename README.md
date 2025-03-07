# GrokAI OpenAI-Compatible API

This project extends the GrokAiChat library to provide an OpenAI-compatible API, allowing you to use Grok AI with applications that support the OpenAI API format.

## Features

- 🔄 Full OpenAI API compatibility
- 🤖 Uses Grok AI for responses
- 📦 Support for multiple Grok models
- 💬 True real-time streaming responses
- 🛠️ Compatible with OpenAI SDKs

## Prerequisites

- Python 3.8 or higher
- A valid X (formerly Twitter) account
- Grok AI access
- Your account's authentication tokens (same as original GrokAiChat)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/grokai-openai-api.git
   cd grokai-openai-api
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Configuration

Create a `.env` file in the project root:

```env
# Original GrokAI credentials
COOKIES="your_cookies_here"
X_CSRF_TOKEN="your_csrf_token_here"
BEARER_TOKEN="your_bearer_token_here"

# Server settings
PORT=5000
```

To obtain credentials, follow the instructions in the original GrokAiChat README.

## Usage

### Starting the Server

```sh
python app.py
```

The server will start at `http://localhost:5000` (or your configured PORT).

### API Endpoints

#### List Models
```
GET /v1/models
```

#### Chat Completions
```
POST /v1/chat/completions
```

Request body:
```json
{
  "model": "grok-3",
  "messages": [
    {"role": "user", "content": "Hello, Grok!"}
  ],
  "stream": false
}
```

### Using with OpenAI-compatible clients

#### Python OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="any-key",  # Not used but required
    base_url="http://localhost:5000/v1"  # Your local server URL
)

response = client.chat.completions.create(
    model="grok-3",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)
```

#### Streaming Example

```python
from openai import OpenAI

client = OpenAI(
    api_key="any-key",
    base_url="http://localhost:5000/v1"
)

stream = client.chat.completions.create(
    model="grok-3",
    messages=[
        {"role": "user", "content": "Write me a short story about AI"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Model Mapping

This API maps OpenAI model names to Grok models:

- `gpt-3.5-turbo` → `grok-1`
- `gpt-4` → `grok-2`
- `gpt-4-turbo` → `grok-3`
- `gpt-4-vision` → `grok-3`

You can also use Grok model names directly: `grok-1`, `grok-2`, `grok-3`

## Features

### Real-time Streaming

The API now supports true real-time streaming of responses from Grok. When using `stream=True`, responses are sent token-by-token as they are received from Grok's API, rather than waiting for the complete response first.

### Multi-turn Conversations

The API simulates multi-turn conversations by including context from previous messages in each new request to Grok.

## Limitations

- The API currently doesn't support all OpenAI parameters
- Token counting is approximate
- Multi-turn conversations are simulated by including context in the message

## License

MIT License, following the original GrokAiChat license.