# GrokAiChat

A Python library for interacting with Grok AI through X's user account API. This project provides a clean interface for creating conversations, sending messages, and handling responses. Note: This uses the account user API, not the paid enterprise API. Grok AI is free for all X (Twitter) members.

## Features

- 🤖 Full Grok API integration
- 📁 File upload support
- 💬 Conversation management
- 🛠️ Easy-to-use interface

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
    # Create a chat and send a message look at the headers of the request and check for the CSRF token
    X_CSRF_TOKEN=""
    # Bearer token
    # Create a chat and send a message look at the headers of the request and check for the Bearer token
    BEARER_TOKEN=""
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

Store these in your `.env` file:
```dotenv
COOKIES="your_copied_cookie_string"
X_CSRF_TOKEN="your_csrf_token"
BEARER_TOKEN="your_bearer_token"
```

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
- Don't automate mass messaging

## Quick Start

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

## Advanced Usage

Check the `examples/` directory for more advanced use cases:
- Basic chat interaction (`chat.py`)
- File attachments (`chatwithfiles.py`)
- Custom model parameters
- Response parsing

## API Documentation

### Main Classes

- `Grok`: Main interface for API interactions
- `GrokMessages`: Response parser and message handler

Full documentation is available in the code comments.

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

