"""
Name: Vibhek Soni
Age: 19
Github: https://github.com/vibheksoni
"""
import requests
import uuid
import json
import mimetypes
import base64
import secrets
import logging
import os
from typing import List, Optional, Dict, Any, Generator, Tuple, Union
import time
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("grok-client")

# API endpoints
CREATE_CONVERSATION_URL = "https://x.com/i/api/graphql/{}/CreateGrokConversation"
ADD_RESPONSE_URL = "https://api.x.com/2/grok/add_response.json"
UPLOAD_FILE_URL = "https://x.com/i/api/2/grok/attachment.json"

# Default retry settings - set to 0 to disable internal retries
DEFAULT_RETRY_COUNT = 0
DEFAULT_RETRY_BACKOFF = 0

# Pool settings - configurable from .env
POOL_CONNECTIONS = int(os.getenv("GROK_POOL_CONNECTIONS", "10"))
POOL_MAXSIZE = int(os.getenv("GROK_POOL_MAXSIZE", "20"))

class GrokMessages:
    """
    Represents a collection of conversation results parsed from raw JSON lines.
    Optimized for performance and memory usage.
    """
    class Result:
        """
        Represents a single result entry with optional fields like message, query, feedbackLabels, etc.
        """
        __slots__ = (
            'sender', 'message', 'query', 'feedback_labels', 'follow_up_suggestions',
            'tools_used', 'cited_web_results', 'web_results', 'media_post_ids', 'post_ids'
        )
        
        def __init__(
            self,
            sender: Optional[str] = None,
            message: Optional[str] = None,
            query: Optional[str] = None,
            feedback_labels: Optional[List[dict]] = None,
            follow_up_suggestions: Optional[List[dict]] = None,
            tools_used: Optional[dict] = None,
            cited_web_results: Optional[List[dict]] = None,
            web_results: Optional[List[dict]] = None,
            media_post_ids: Optional[List[str]] = None,
            post_ids: Optional[List[str]] = None
        ) -> None:
            self.sender = sender
            self.message = message
            self.query = query
            self.feedback_labels = feedback_labels
            self.follow_up_suggestions = follow_up_suggestions
            self.tools_used = tools_used
            self.cited_web_results = cited_web_results
            self.web_results = web_results
            self.media_post_ids = media_post_ids
            self.post_ids = post_ids

        def __repr__(self) -> str:
            return f"<Result(sender={self.sender}, message={self.message[:30] + '...' if self.message and len(self.message) > 30 else self.message})>"

    def __init__(self, raw_data: str) -> None:
        """
        Parse the provided raw JSONL data into a collection of Result objects.
        """
        self.raw_data = raw_data
        self.results: List[GrokMessages.Result] = []
        self._parse_raw_data()

    def _parse_raw_data(self) -> None:
        """
        Split the raw data by lines and convert each line to a Result object stored in self.results.
        Optimized to handle invalid JSON and thinking traces more efficiently.
        """
        if not self.raw_data:
            return
            
        lines = self.raw_data.splitlines()
        for line in lines:
            if not line.strip():
                continue
                
            try:
                # Only parse JSON if the line starts with a curly brace
                if line.strip().startswith("{"):
                    parsed = json.loads(line)
                    result_data = parsed.get("result", {})
                    
                    # Skip entries with messageTag set to "thinking_trace"
                    if result_data.get("messageTag") == "thinking_trace":
                        continue
                        
                    # Additional check for thinking traces
                    if result_data.get("isThinking") is True:
                        continue
                    
                    result = self.Result(
                        sender=result_data.get("sender"),
                        message=result_data.get("message"),
                        query=result_data.get("query"),
                        feedback_labels=result_data.get("feedbackLabels"),
                        follow_up_suggestions=result_data.get("followUpSuggestions"),
                        tools_used=result_data.get("toolsUsed"),
                        cited_web_results=result_data.get("citedWebResults"),
                        web_results=result_data.get("webResults"),
                        media_post_ids=result_data.get("xMediaPostIds"),
                        post_ids=result_data.get("xPostIds"),
                    )
                    self.results.append(result)
            except json.JSONDecodeError:
                # Skip malformed JSON
                continue
            except Exception as e:
                logger.error(f"Error parsing result: {str(e)}")

    @lru_cache(maxsize=1)
    def get_message_tokens(self) -> List[str]:
        """
        Return a list of message tokens from the parsed results.
        Cached for performance.
        """
        return [result.message for result in self.results if result.message]

    @lru_cache(maxsize=1)
    def get_full_message(self) -> str:
        """
        Return the full message from the parsed results.
        Cached for performance.
        """
        return ''.join(self.get_message_tokens())

    def get_queries(self) -> List[str]:
        """Return all query strings from the parsed results."""
        return [result.query for result in self.results if result.query]

    def get_feedback_labels(self) -> List[dict]:
        """Return a list of feedback label objects."""
        return [result.feedback_labels for result in self.results if result.feedback_labels]

    def get_follow_up_suggestions(self) -> List[dict]:
        """Return a list of follow-up suggestion objects."""
        return [result.follow_up_suggestions for result in self.results if result.follow_up_suggestions]

    def get_tools_used(self) -> List[dict]:
        """Return a list of tools used (metadata) from the parsed results."""
        return [result.tools_used for result in self.results if result.tools_used]

    def get_cited_web_results(self) -> List[dict]:
        """Return a list of cited web results."""
        return [result.cited_web_results for result in self.results if result.cited_web_results]

    def get_web_results(self) -> List[dict]:
        """Return a list of web results."""
        return [result.web_results for result in self.results if result.web_results]

    def get_media_post_ids(self) -> List[List[str]]:
        """Return a list of lists containing media post IDs."""
        return [result.media_post_ids for result in self.results if result.media_post_ids]

    def get_post_ids(self) -> List[List[str]]:
        """Return a list of lists containing post IDs."""
        return [result.post_ids for result in self.results if result.post_ids]

class Grok:
    """
    Provides methods to manage Grok interactions such as creating conversations, 
    uploading files, and sending messages.
    Modified to disable internal retries in favor of credential rotation.
    """
    @staticmethod
    def generate_random_base64_string(length=94) -> str:
        """
        Generate a random Base64 string of specified length.
        
        Args:
            length: The desired length of the output string
            
        Returns:
            A random Base64 string of the specified length
        """
        # Calculate how many random bytes we need
        # Base64 encodes 3 bytes into 4 characters
        byte_length = (length * 3) // 4 + 1
        
        # Generate random bytes using cryptographically secure random generator
        random_bytes = secrets.token_bytes(byte_length)
        
        # Encode the bytes as Base64
        base64_string = base64.b64encode(random_bytes).decode('utf-8')
        
        # Trim to the desired length and return
        return base64_string[:length]
        
    def __init__(
        self,
        account_bearer_token: str,
        x_csrf_token: str,
        cookies: str,
        retry_count: int = DEFAULT_RETRY_COUNT,
        retry_backoff: float = DEFAULT_RETRY_BACKOFF
    ) -> None:
        """
        Initialize a requests session and store relevant headers.
        
        Args:
            account_bearer_token: Bearer token for authentication
            x_csrf_token: CSRF token for protection against CSRF attacks
            cookies: Cookies string for authentication
            retry_count: Number of internal retries for transient errors (set to 0)
            retry_backoff: Backoff multiplier for retries (not used when retry_count=0)
        """
        self.session = requests.Session()
        
        # Set up connection pooling configuration with max_retries=0
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=POOL_CONNECTIONS,
            pool_maxsize=POOL_MAXSIZE,
            max_retries=0,  # Disable retries at the adapter level
            pool_block=False
        )
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        
        self.client_uuid = uuid.uuid4().hex
        self.retry_count = 0  # Force to 0 to ensure no retries
        self.retry_backoff = 0  # Force to 0 to ensure no backoff
        
        # Generate transaction ID
        self.transaction_id = self.generate_random_base64_string(94)
        
        # Clean and prepare headers
        self._prepare_headers(account_bearer_token, x_csrf_token, cookies)
        
        self.conversation_info = {
            "data": {
                "create_grok_conversation": {
                    "conversation_id": ""
                }
            }
        }
        
        # Stats for monitoring
        self.request_count = 0
        self.last_request_time = 0
    
    def _prepare_headers(self, account_bearer_token: str, x_csrf_token: str, cookies: str) -> None:
        """Prepare and sanitize request headers"""
        headers = {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "authorization": f"Bearer {account_bearer_token}",
            "content-type": "application/json",
            "cookie": cookies,
            "origin": "https://x.com",
            "priority": "u=1, i",
            "referer": "https://x.com/i/grok",
            "sec-ch-ua": "\"Chromium\";v=\"134\", \"Not:A-Brand\";v=\"24\", \"Google Chrome\";v=\"134\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
            "x-client-transaction-id": self.transaction_id,
            "x-client-uuid": self.client_uuid,
            "x-csrf-token": x_csrf_token,
            "x-twitter-active-user": "yes",
            "x-twitter-auth-type": "OAuth2Session",
            "x-twitter-client-language": "en"
        }
        
        # Clean headers to ensure valid ASCII characters
        sanitized_headers = {}
        for key, value in headers.items():
            if isinstance(value, str):
                # Replace any problematic Unicode characters with ASCII equivalents
                sanitized_value = value.replace("\u2026", "...").encode('ascii', 'ignore').decode('ascii')
                sanitized_headers[key] = sanitized_value
            else:
                sanitized_headers[key] = value
        
        self.session.headers.update(sanitized_headers)
    
    def create_conversation(self) -> str:
        """
        Create a new Grok conversation and store the conversation info.
        Returns the conversation ID.
        """
        query_id = "vvC5uy7pWWHXS2aDi1FZeA"
        data = {"variables":{},"queryId":query_id}
        
        # Track request time for rate limiting and monitoring
        self.last_request_time = time.time()
        self.request_count += 1
        
        try:
            response = self.session.post(
                CREATE_CONVERSATION_URL.format(query_id), 
                json=data
            )
            
            # Raise for HTTP errors
            response.raise_for_status()
            
            self.conversation_info = response.json()
            conversation_id = self.conversation_info.get('data', {}).get('create_grok_conversation', {}).get('conversation_id', 'unknown')
            logger.info(f"Created conversation: {conversation_id}")
            return conversation_id
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error when creating conversation: {e.response.status_code} - {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error creating conversation: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating conversation: {str(e)}")
            raise
    
    def upload_file(self, file_path: str) -> Dict:
        """
        Upload a file and return the JSON response containing mediaId and URL.
        Improved error handling and resource management.
        """
        # Save original content-type to restore later
        original_content_type = self.session.headers.get("content-type")
        
        # Temporarily remove content-type header for multipart form upload
        if "content-type" in self.session.headers:
            del self.session.headers["content-type"]
        
        # Determine MIME type
        content_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
        
        try:
            # Normalize file path
            file_path = file_path.replace('\\', '/')
            filename = file_path.split('/')[-1]
            
            # Track request for monitoring
            self.last_request_time = time.time()
            self.request_count += 1
            
            with open(file_path, "rb") as f:
                files = {
                    "image": (
                        filename,
                        f,
                        content_type
                    )
                }
                
                response = self.session.post(
                    UPLOAD_FILE_URL, 
                    files=files
                )
                
                # Raise for HTTP errors
                response.raise_for_status()
                
                response_json = response.json()
                
                if not response_json or not isinstance(response_json, list) or len(response_json) == 0:
                    raise ValueError(f"Invalid response format when uploading file: {response_json}")
                
                media_id = response_json[0].get("mediaId")
                if not media_id:
                    raise ValueError(f"No mediaId in response: {response_json}")
                    
                # Add URL to response for convenience
                response_json[0]["url"] = f"https://api.x.com/2/grok/attachment.json?mediaId={media_id}"
                return response_json
                
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error when uploading file: {e.response.status_code} - {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error uploading file: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            raise
        finally:
            # Always restore the content-type header
            if original_content_type:
                self.session.headers["content-type"] = original_content_type
    
    def create_message(
        self, 
        model_name: str, 
        imageGenerateCount: int = 1,
        returnSearchResults: bool = True,
        returnCitations: bool = True,
        eagerTweets: bool = True,
        serverHistory: bool = True,
        isDeepsearch: bool = False,
        isReasoning: bool = False,
        is_deeper_search: bool = False
    ) -> Dict[str, Any]:
        """
        Create a template for the conversation payload using the specified Grok model.
        """
        # Validate conversation ID exists
        conversation_id = self.conversation_info.get("data", {}).get("create_grok_conversation", {}).get("conversation_id")
        if not conversation_id:
            raise ValueError("No active conversation. Call create_conversation() first.")
        
        # Create base payload with conversationId first
        payload = {
            "responses": [],
            "systemPromptName": "",
            "grokModelOptionId": model_name,
            "conversationId": conversation_id,
        }
        
        # Add deepsearchArgs if deeper search is enabled
        if is_deeper_search:
            payload["deepsearchArgs"] = {
                "mode": "deeper"
            }
        
        # Add remaining fields
        payload.update({
            "returnSearchResults": returnSearchResults,
            "returnCitations": returnCitations,
            "promptMetadata": {
                "promptSource": "NATURAL",
                "action": "INPUT"
            },
            "imageGenerationCount": imageGenerateCount,
            "requestFeatures": {
                "eagerTweets": eagerTweets,
                "serverHistory": serverHistory
            },
            "enableSideBySide": not (isDeepsearch or is_deeper_search),
            "toolOverrides": {},
            "isDeepsearch": isDeepsearch,
            "isReasoning": isReasoning
        })
        
        return payload

    def add_user_message(
        self,
        request_data: Dict[str, Any],
        message: str,
        sender: int = 1,
        file_attachments: Union[List, Dict] = []
    ) -> None:
        """
        Append a user message, optionally with file attachments, to the request payload.
        """
        # Make file_attachments an empty list if None
        if file_attachments is None:
            file_attachments = []
            
        # Convert dict to list if a single attachment was provided
        if isinstance(file_attachments, dict):
            file_attachments = [file_attachments]
            
        # Ensure request_data has responses key
        if "responses" not in request_data:
            request_data["responses"] = []
            
        request_data["responses"].append({
            "message": message,
            "sender": sender,
            "promptSource": "",
            "fileAttachments": file_attachments
        })
    
    def send(self, request_data: Dict[str, Any], retry_on_error: bool = False) -> str:
        """
        Send the conversation payload to the server and return the response text.
        Modified to remove retry logic.
        """
        try:
            # Track request for monitoring
            self.last_request_time = time.time()
            self.request_count += 1
            
            response = self.session.post(
                ADD_RESPONSE_URL, 
                json=request_data
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            return response.text
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error sending message: {e.response.status_code} - {e.response.text}")
            raise
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {str(e)}")
            raise
    
    def _is_rate_limit_response(self, result_data: Dict[str, Any]) -> bool:
        """Check if the result_data indicates a rate limit."""
        # Quick check with multiple conditions to determine rate limiting
        if result_data.get("responseType") == "limiter" or "upsell" in result_data:
            return True
        
        # Check message content for rate limit text
        message = result_data.get("message", "")
        if message and ("You've reached your limit" in message or "limit of" in message):
            return True
        
        return False
    
    def send_streaming(self, request_data: Dict[str, Any], filter_final_only: bool = False) -> Generator:
        """
        Send the conversation payload to the server and yield message tokens as they arrive.
        This allows for real-time streaming of Grok's responses.
        
        Args:
            request_data: The request payload to send
            filter_final_only: If True, only yield messages with messageTag="final"
        """
        try:
            # Track request for monitoring
            self.last_request_time = time.time()
            self.request_count += 1
            
            with self.session.post(
                ADD_RESPONSE_URL, 
                json=request_data, 
                stream=True
            ) as response:
                # Check for HTTP errors
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if not line:
                        continue
                        
                    line_text = line.decode('utf-8')
                    try:
                        # Only try to parse JSON if it looks like JSON
                        if line_text.strip().startswith("{"):
                            parsed = json.loads(line_text)
                            result_data = parsed.get("result", {})
                            
                            # Check if this is a rate limit response BEFORE filtering
                            if self._is_rate_limit_response(result_data):
                                # Yield a special token to indicate rate limiting
                                yield "__RATE_LIMITED__"
                                return
                            
                            message_token = result_data.get("message")
                            message_tag = result_data.get("messageTag")
                            
                            # Skip entries with messageTag set to "thinking_trace"
                            if message_tag == "thinking_trace":
                                continue
                                
                            # Check if this is potentially a thinking trace (additional check)
                            if result_data.get("isThinking") is True:
                                continue
                            
                            # Check if we should filter for final messages
                            if message_token:
                                # For deepsearch responses, only include chunks tagged as "final"
                                if filter_final_only:
                                    if message_tag == "final":
                                        yield message_token
                                else:
                                    yield message_token
                        else:
                            # Non-JSON line, yield it directly unless filtering
                            if not filter_final_only:
                                yield line_text
                                
                    except json.JSONDecodeError:
                        # Skip malformed JSON
                        continue
                    except Exception as e:
                        logger.error(f"Error processing streaming response: {str(e)}")
                        
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error during streaming: {e.response.status_code} - {e.response.text}")
            yield f"__ERROR__: HTTP {e.response.status_code}"
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error during streaming: {str(e)}")
            yield f"__ERROR__: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error during streaming: {str(e)}")
            yield f"__ERROR__: {str(e)}"