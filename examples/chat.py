import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grok import Grok, GrokMessages
from dotenv import load_dotenv

load_dotenv()

COOKIES = os.getenv("COOKIES")
X_CSRF_TOKEN = os.getenv("X_CSRF_TOKEN")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# Get timeout settings from .env or use defaults
CONNECT_TIMEOUT = float(os.getenv("GROK_CONNECT_TIMEOUT", "10.0"))
READ_TIMEOUT = float(os.getenv("GROK_READ_TIMEOUT", "30.0"))
RETRY_COUNT = int(os.getenv("GROK_RETRY_COUNT", "2"))
RETRY_BACKOFF = float(os.getenv("GROK_RETRY_BACKOFF", "1.5"))

# Initialize the Grok class with configurable settings
grok = Grok(
    BEARER_TOKEN, 
    X_CSRF_TOKEN, 
    COOKIES, 
    timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
    retry_count=RETRY_COUNT,
    retry_backoff=RETRY_BACKOFF
)

# Create a conversation
grok.create_conversation()
# Send a message
request_data = grok.create_message("grok-2")
# Add a user message
grok.add_user_message(request_data, "Nice write me c++ add function")
# Send the message
response = grok.send(request_data)
# Parse the response
response = GrokMessages(response)
# Print the response
print(response.get_full_message())