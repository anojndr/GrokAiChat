import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from grok import Grok, GrokMessages
from dotenv import load_dotenv

load_dotenv()

COOKIES = os.getenv("COOKIES")
X_CSRF_TOKEN = os.getenv("X_CSRF_TOKEN")
BEARER_TOKEN = os.getenv("BEARER_TOKEN")

# Create client with retry disabled
grok = Grok(
    BEARER_TOKEN, 
    X_CSRF_TOKEN, 
    COOKIES, 
    retry_count=0,
    retry_backoff=0
)

# Create a conversation
grok.create_conversation()
# Send a message
request_data = grok.create_message("grok-3")
# Add a user message
grok.add_user_message(request_data, "Nice write me c++ add function")
# Send the message
response = grok.send(request_data)
# Parse the response
response = GrokMessages(response)
# Print the response
print(response.get_full_message())