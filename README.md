# GrokAiChat

A Python library for interacting with Grok AI through X's user account API. This project provides both a clean interface for direct Grok API interactions and an OpenAI-compatible API server. Note: This uses the account user API, not the paid enterprise API. Grok AI is free for all X (Twitter) members.

## Features

- 🤖 Full Grok API integration
- 📁 File upload support
- 💬 Conversation management
- 🛠️ Easy-to-use interface
- 🔄 OpenAI-compatible API (use with existing OpenAI clients)
- 🐳 Docker support for easy deployment
- 🔍 Support for Grok DeepSearch and Reasoning modes
- 🔄 Credential rotation for higher throughput
- 📊 Streaming responses

## Prerequisites

- Python 3.8 or higher
- A valid X (formerly Twitter) account
- Grok AI access (free for all X members)
- Your account's authentication tokens

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
    # Raw cookies string
    # Example: "cookie1=value1; cookie2=value2"
    COOKIES=""
    # CSRF token
    X_CSRF_TOKEN=""
    # Bearer token
    BEARER_TOKEN=""
    
    # Optional: Multiple credential sets (for API server)
    COOKIES_2=""
    X_CSRF_TOKEN_2=""
    BEARER_TOKEN_2=""
    
    # API settings
    PORT=5000
    API_CONNECT_TIMEOUT=10.0
    API_READ_TIMEOUT=30.0
    DOWNLOAD_CONNECT_TIMEOUT=5.0
    DOWNLOAD_READ_TIMEOUT=10.0
    MAX_RETRIES=3
    GROK_RETRY_COUNT=2
    GROK_RETRY_BACKOFF=1.5
    STREAM_BUFFER_SIZE=10
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

Store these in your `.env` file as shown above. For the API server, you can add multiple credential sets (COOKIES_2, X_CSRF_TOKEN_2, etc.) for automatic rotation when rate limits are hit.

## Usage Options

### Option 1: Direct Library Usage

```python
from grok import Grok, GrokMessages
from dotenv import load_dotenv
import os

load_dotenv()
grok = Grok(
    os.getenv("BEARER_TOKEN"),
    os.getenv("X_CSRF_TOKEN"),
    os.getenv("COOKIES"),
    timeout=(10.0, 30.0)  # (connect_timeout, read_timeout)
)

# Create a conversation
grok.create_conversation()

# Send a message
request = grok.create_message("grok-3")
grok.add_user_message(request, "Hello, Grok!")
response = grok.send(request)

# Parse and print response
messages = GrokMessages(response)
print(messages.get_full_message())
```

### Option 2: OpenAI-Compatible API Server

#### Starting the Server

**Using Python:**
```sh
uvicorn app:app --host 0.0.0.0 --port 5000 --workers 4
```

**Using Docker:**
```sh
docker-compose up -d
```

#### Using the API with OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy-key",  # Not actually used, but required
    base_url="http://localhost:5000/v1"  # Your GrokAI API server address
)

# Create a simple chat completion
response = client.chat.completions.create(
    model="grok-3",  # You can also use OpenAI model names like "gpt-4"
    messages=[
        {"role": "system", "content": "You are a helpful assistant powered by Grok AI."},
        {"role": "user", "content": "Tell me about yourself. What AI model are you?"}
    ]
)

# Print the response
print(f"Response: {response.choices[0].message.content}")
```

#### Direct API Calls

```python
import requests

endpoint = "http://localhost:5000/v1/chat/completions"

payload = {
    "model": "grok-3",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant powered by Grok AI."},
        {"role": "user", "content": "Explain what makes you unique as an AI."}
    ],
    "stream": False
}

response = requests.post(endpoint, json=payload)

if response.status_code == 200:
    data = response.json()
    print(f"Response: {data['choices'][0]['message']['content']}")
else:
    print(f"Error: {response.status_code}, {response.text}")
```

#### Special Features

**DeepSearch Mode:**
```python
# Add the "deepsearch" keyword to your prompt to enable Grok's DeepSearch
response = client.chat.completions.create(
    model="grok-3",
    messages=[
        {"role": "user", "content": "deepsearch What are the latest developments in quantum computing?"}
    ]
)
```

**Reasoning Mode:**
```python
# Add the "reasoning" keyword to your prompt to enable Grok's reasoning mode
response = client.chat.completions.create(
    model="grok-3",
    messages=[
        {"role": "user", "content": "reasoning Solve this complex problem step by step..."}
    ]
)
```

## Docker Deployment

1. Build and start the container:
   ```sh
   docker-compose up -d
   ```

2. View logs:
   ```sh
   docker-compose logs -f
   ```

3. Stop the service:
   ```sh
   docker-compose down
   ```

### Environment Variables for Docker

You can configure the Docker deployment by modifying the `.env` file or setting environment variables. Key settings include:

- `PORT`: Server port (default: 5000)
- `WORKERS`: Number of Uvicorn workers (default: 4)
- `API_CONNECT_TIMEOUT`, `API_READ_TIMEOUT`: Request timeouts
- `MEMORY_LIMIT`, `CPU_LIMIT`: Resource limits for the container

## Advanced Usage

Check the `examples/` directory for more advanced use cases:
- Basic chat interaction (`chat.py`)
- File attachments (`chatwithfiles.py`)
- OpenAI-compatible API usage (`example_usage.py`)

## API Documentation

### Main Classes

- `Grok`: Main interface for API interactions
- `GrokMessages`: Response parser and message handler

### API Endpoints

- `GET /v1/models`: List available models
- `POST /v1/chat/completions`: Create a chat completion (OpenAI-compatible)
- `GET /`: API status and information

Full documentation is available in the code comments.

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Legal Notice

⚠️ **Disclaimer**: This project is for educational purposes only. Users are responsible for ensuring their usage complies with X's terms of service.

## License

[MIT License](LICENSE) - See license file for details.

## Author

**Vibhek Soni**
- Age: 19
- GitHub: [@vibheksoni](https://github.com/vibheksoni)
- Project Link: [GrokAiChat](https://github.com/vibheksoni/GrokAiChat)# GrokAiChat

A Python library for interacting with Grok AI through X's user account API. This project provides both a clean interface for direct Grok API interactions and an OpenAI-compatible API server. Note: This uses the account user API, not the paid enterprise API. Grok AI is free for all X (Twitter) members.

## Features

- 🤖 Full Grok API integration
- 📁 File upload support
- 💬 Conversation management
- 🛠️ Easy-to-use interface
- 🔄 OpenAI-compatible API (use with existing OpenAI clients)
- 🐳 Docker support for easy deployment
- 🔍 Support for Grok DeepSearch and Reasoning modes
- 🔄 Credential rotation for higher throughput
- 📊 Streaming responses

## Prerequisites

- Python 3.8 or higher
- A valid X (formerly Twitter) account
- Grok AI access (free for all X members)
- Your account's authentication tokens

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
    # Raw cookies string
    # Example: "cookie1=value1; cookie2=value2"
    COOKIES=""
    # CSRF token
    X_CSRF_TOKEN=""
    # Bearer token
    BEARER_TOKEN=""
    
    # Optional: Multiple credential sets (for API server)
    COOKIES_2=""
    X_CSRF_TOKEN_2=""
    BEARER_TOKEN_2=""
    
    # API settings
    PORT=5000
    API_CONNECT_TIMEOUT=10.0
    API_READ_TIMEOUT=30.0
    DOWNLOAD_CONNECT_TIMEOUT=5.0
    DOWNLOAD_READ_TIMEOUT=10.0
    MAX_RETRIES=3
    GROK_RETRY_COUNT=2
    GROK_RETRY_BACKOFF=1.5
    STREAM_BUFFER_SIZE=10
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

Store these in your `.env` file as shown above. For the API server, you can add multiple credential sets (COOKIES_2, X_CSRF_TOKEN_2, etc.) for automatic rotation when rate limits are hit.

## Usage Options

### Option 1: Direct Library Usage

```python
from grok import Grok, GrokMessages
from dotenv import load_dotenv
import os

load_dotenv()
grok = Grok(
    os.getenv("BEARER_TOKEN"),
    os.getenv("X_CSRF_TOKEN"),
    os.getenv("COOKIES"),
    timeout=(10.0, 30.0)  # (connect_timeout, read_timeout)
)

# Create a conversation
grok.create_conversation()

# Send a message
request = grok.create_message("grok-3")
grok.add_user_message(request, "Hello, Grok!")
response = grok.send(request)

# Parse and print response
messages = GrokMessages(response)
print(messages.get_full_message())
```

### Option 2: OpenAI-Compatible API Server

#### Starting the Server

**Using Python:**
```sh
uvicorn app:app --host 0.0.0.0 --port 5000 --workers 4
```

**Using Docker:**
```sh
docker-compose up -d
```

#### Using the API with OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy-key",  # Not actually used, but required
    base_url="http://localhost:5000/v1"  # Your GrokAI API server address
)

# Create a simple chat completion
response = client.chat.completions.create(
    model="grok-3",  # You can also use OpenAI model names like "gpt-4"
    messages=[
        {"role": "system", "content": "You are a helpful assistant powered by Grok AI."},
        {"role": "user", "content": "Tell me about yourself. What AI model are you?"}
    ]
)

# Print the response
print(f"Response: {response.choices[0].message.content}")
```

#### Direct API Calls

```python
import requests

endpoint = "http://localhost:5000/v1/chat/completions"

payload = {
    "model": "grok-3",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant powered by Grok AI."},
        {"role": "user", "content": "Explain what makes you unique as an AI."}
    ],
    "stream": False
}

response = requests.post(endpoint, json=payload)

if response.status_code == 200:
    data = response.json()
    print(f"Response: {data['choices'][0]['message']['content']}")
else:
    print(f"Error: {response.status_code}, {response.text}")
```

#### Special Features

**DeepSearch Mode:**
```python
# Add the "deepsearch" keyword to your prompt to enable Grok's DeepSearch
response = client.chat.completions.create(
    model="grok-3",
    messages=[
        {"role": "user", "content": "deepsearch What are the latest developments in quantum computing?"}
    ]
)
```

**Reasoning Mode:**
```python
# Add the "reasoning" keyword to your prompt to enable Grok's reasoning mode
response = client.chat.completions.create(
    model="grok-3",
    messages=[
        {"role": "user", "content": "reasoning Solve this complex problem step by step..."}
    ]
)
```

## Docker Deployment

1. Build and start the container:
   ```sh
   docker-compose up -d
   ```

2. View logs:
   ```sh
   docker-compose logs -f
   ```

3. Stop the service:
   ```sh
   docker-compose down
   ```

### Environment Variables for Docker

You can configure the Docker deployment by modifying the `.env` file or setting environment variables. Key settings include:

- `PORT`: Server port (default: 5000)
- `WORKERS`: Number of Uvicorn workers (default: 4)
- `API_CONNECT_TIMEOUT`, `API_READ_TIMEOUT`: Request timeouts
- `MEMORY_LIMIT`, `CPU_LIMIT`: Resource limits for the container

## Advanced Usage

Check the `examples/` directory for more advanced use cases:
- Basic chat interaction (`chat.py`)
- File attachments (`chatwithfiles.py`)
- OpenAI-compatible API usage (`example_usage.py`)

## API Documentation

### Main Classes

- `Grok`: Main interface for API interactions
- `GrokMessages`: Response parser and message handler

### API Endpoints

- `GET /v1/models`: List available models
- `POST /v1/chat/completions`: Create a chat completion (OpenAI-compatible)
- `GET /`: API status and information

Full documentation is available in the code comments.

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Legal Notice

⚠️ **Disclaimer**: This project is for educational purposes only. Users are responsible for ensuring their usage complies with X's terms of service.

## License

[MIT License](LICENSE) - See license file for details.

## Author

**Vibhek Soni**
- Age: 19
- GitHub: [@vibheksoni](https://github.com/vibheksoni)
- Project Link: [GrokAiChat](https://github.com/vibheksoni/GrokAiChat)

**anojndr**
- GitHub: [@anojndr](https://github.com/anojndr)
- Project Link: [GrokAiChat](https://github.com/anojndr/GrokAiChat)