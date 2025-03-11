"""
Name: Vibhek Soni
Age: 19
Github: https://github.com/vibheksoni
"""
import requests, uuid, json, mimetypes
from typing import List, Optional

CREATE_CONVERSATION_URL = "https://x.com/i/api/graphql/{}/CreateGrokConversation"
ADD_RESPONSE_URL = "https://api.x.com/2/grok/add_response.json"
UPLOAD_FILE_URL = "https://x.com/i/api/2/grok/attachment.json"

# Default timeout settings (in seconds)
DEFAULT_CONNECT_TIMEOUT = 10.0  # Connection timeout
DEFAULT_READ_TIMEOUT = 30.0     # Read timeout
DEFAULT_TIMEOUT = (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)  # (connect, read) tuple

class GrokMessages:
    """
    Represents a collection of conversation results parsed from raw JSON lines.
    """
    class Result:
        """
        Represents a single result entry with optional fields like message, query, feedbackLabels, etc.
        """
        def __init__(
            self,
            sender: str,
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
            return f"<Result(sender={self.sender}, message={self.message})>"

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
        """
        lines = self.raw_data.splitlines()
        for line in lines:
            if line.strip():
                try:
                    parsed = json.loads(line)
                    result_data = parsed.get("result", {})
                    
                    # Skip entries with messageTag set to "thinking_trace"
                    if result_data.get("messageTag") == "thinking_trace":
                        continue
                        
                    # Additional check for thinking traces
                    if result_data.get("isThinking") == True:
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
                    print(f"Error parsing result: {str(e)}")

    def get_message_tokens(self) -> List[str]:
        """
        Return a list of message tokens from the parsed results.
        """
        return [result.message for result in self.results if result.message]

    def get_full_message(self) -> str:
        """
        Return the full message from the parsed results.
        """
        return ''.join(self.get_message_tokens())

    def get_queries(self) -> List[str]:
        """
        Return all query strings from the parsed results.
        """
        return [result.query for result in self.results if result.query]

    def get_feedback_labels(self) -> List[dict]:
        """
        Return a list of feedback label objects.
        """
        return [result.feedback_labels for result in self.results if result.feedback_labels]

    def get_follow_up_suggestions(self) -> List[dict]:
        """
        Return a list of follow-up suggestion objects.
        """
        return [result.follow_up_suggestions for result in self.results if result.follow_up_suggestions]

    def get_tools_used(self) -> List[dict]:
        """
        Return a list of tools used (metadata) from the parsed results.
        """
        return [result.tools_used for result in self.results if result.tools_used]

    def get_cited_web_results(self) -> List[dict]:
        """
        Return a list of cited web results.
        """
        return [result.cited_web_results for result in self.results if result.cited_web_results]

    def get_web_results(self) -> List[dict]:
        """
        Return a list of web results.
        """
        return [result.web_results for result in self.results if result.web_results]

    def get_media_post_ids(self) -> List[List[str]]:
        """
        Return a list of lists containing media post IDs.
        """
        return [result.media_post_ids for result in self.results if result.media_post_ids]

    def get_post_ids(self) -> List[List[str]]:
        """
        Return a list of lists containing post IDs.
        """
        return [result.post_ids for result in self.results if result.post_ids]

class Grok:
    """
    Provides methods to manage Grok interactions such as creating conversations, uploading files, and sending messages.
    """
    def __init__(
        self,
        account_bearer_token: str,
        x_csrf_token: str,
        cookies: str,
        timeout: tuple = DEFAULT_TIMEOUT
    ) -> None:
        """
        Initialize a requests session and store relevant headers.
        
        Args:
            account_bearer_token: Bearer token for authentication
            x_csrf_token: CSRF token for protection against CSRF attacks
            cookies: Cookies string for authentication
            timeout: Request timeout as tuple (connect_timeout, read_timeout) in seconds
        """
        self.session = requests.Session()
        self.client_uuid = uuid.uuid4().hex
        self.timeout = timeout
        
        headers = {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate",
            "accept-language": "en-US,en;q=0.9",
            "authorization": f"Bearer {account_bearer_token}",
            "content-type": "application/json",
            "cookie": cookies,
            "origin": "https://x.com",
            "priority": "u=1, i",
            "referer": "https://x.com/i/grok",
            "sec-ch-ua": "\"Google Chrome\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "x-client-uuid": self.client_uuid,
            "x-csrf-token": x_csrf_token,
            "x-twitter-active-user": "yes",
            "x-twitter-auth-type": "OAuth2Session",
            "x-twitter-client-language": "en"
        }
        self.session.headers = headers
        self.conversation_info = {
            "data": {
                "create_grok_conversation": {
                    "conversation_id": ""
                }
            }
        }
    
    def create_conversation(self) -> None:
        """
        Create a new Grok conversation and store the conversation info.
        """
        query_id = "6cmfJY3d7EPWuCSXWrkOFg"
        data = {"variables":{},"queryId":query_id}
        try:
            response = self.session.post(
                CREATE_CONVERSATION_URL.format(query_id), 
                json=data, 
                timeout=self.timeout
            )
            self.conversation_info = response.json()
            print(f"Created conversation: {self.conversation_info.get('data', {}).get('create_grok_conversation', {}).get('conversation_id', 'unknown')}")
        except requests.exceptions.Timeout:
            print("Timeout error when creating conversation. Check your network connection or X.com API availability.")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Error creating conversation: {str(e)}")
            raise
    
    def upload_file(self, file_path: str) -> dict:
        """
        Upload a file and return the JSON response containing mediaId and URL.
        """
        original_content_type = self.session.headers.get("content-type")
        if "content-type" in self.session.headers:
            del self.session.headers["content-type"]
        
        content_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
        
        with open(file_path, "rb") as f:
            file_path = file_path.replace('\\', '/')
            filename = file_path.split('/')[-1]
            try:
                files = {
                    "image": (
                        filename,    
                        f,           
                        content_type 
                    )
                }
            except Exception as e:
                raise Exception(f"Error preparing file upload: {str(e)}")
            
            try:
                response = self.session.post(
                    UPLOAD_FILE_URL, 
                    files=files, 
                    timeout=self.timeout
                )
                response_json = response.json()
                media_id = response_json[0]["mediaId"]
                response_json[0]["url"] = f"https://api.x.com/2/grok/attachment.json?mediaId={media_id}"
                self.session.headers["content-type"] = original_content_type
                return response_json
            except requests.exceptions.Timeout:
                print("Timeout error when uploading file. Check your network connection or X.com API availability.")
                raise
            except requests.exceptions.RequestException as e:
                print(f"Error uploading file: {str(e)}")
                raise
            finally:
                # Ensure we restore the content-type header
                if original_content_type:
                    self.session.headers["content-type"] = original_content_type
    
    def create_message(
        self, 
        model_name: str, 
        imageGenerateCount:int = 1,
        returnSearchResults: bool = True,
        returnCitations: bool = True,
        eagerTweets: bool = True,
        serverHistory: bool = True,
        isDeepsearch: bool = False,
        isReasoning: bool = False
    ) -> dict:
        """
        Create a template for the conversation payload using the specified Grok model.
        """
        return {
            "responses": [],
            "systemPromptName": "",
            "grokModelOptionId": model_name,
            "conversationId": self.conversation_info["data"]["create_grok_conversation"]["conversation_id"],
            "returnSearchResults": returnSearchResults,
            "returnCitations": returnCitations,
            "promptMetadata": {
                "promptSource": "NATURAL",
                "action": "INPUT"
            },
            "imageGenerationCount": imageGenerateCount, # Seems like you can get more than one image at a time
            "requestFeatures": {
                "eagerTweets": eagerTweets,
                "serverHistory": serverHistory
            },
            "enableSideBySide": True,
            "toolOverrides": {},
            "isDeepsearch": isDeepsearch,
            "isReasoning": isReasoning
        }

    def add_user_message(
        self,
        request_data: dict,
        message: str,
        sender: int = 1,
        file_attachments: List[str] = []
    ) -> None:
        """
        Append a user message, optionally with file attachments, to the request payload.
        """
        request_data["responses"].append({
            "message": message,
            "sender": sender,
            "promptSource": "",
            "fileAttachments": file_attachments
        })
    
    def send(self, request_data: dict) -> str:
        """
        Send the conversation payload to the server and return the response text.
        """
        try:
            response = self.session.post(
                ADD_RESPONSE_URL, 
                json=request_data, 
                timeout=self.timeout
            )
            return response.text
        except requests.exceptions.Timeout:
            print("Timeout error when sending message. Check your network connection or X.com API availability.")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Error sending message: {str(e)}")
            raise
    
    def _is_rate_limit_response(self, result_data):
        """Check if the result_data indicates a rate limit."""
        # Check for explicit rate limit indicators
        if result_data.get("responseType") == "limiter":
            return True
        
        # Check for upsell object which indicates premium requirement
        if "upsell" in result_data:
            return True
        
        # Check message content for rate limit text
        message = result_data.get("message", "")
        if message and ("You've reached your limit" in message or "limit of" in message):
            return True
        
        return False
    
    def send_streaming(self, request_data: dict, filter_final_only=False):
        """
        Send the conversation payload to the server and yield message tokens as they arrive.
        This allows for real-time streaming of Grok's responses.
        
        Args:
            request_data: The request payload to send
            filter_final_only: If True, only yield messages with messageTag="final"
        """
        try:
            with self.session.post(
                ADD_RESPONSE_URL, 
                json=request_data, 
                stream=True, 
                timeout=self.timeout
            ) as response:
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        try:
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
                            if result_data.get("isThinking") == True:
                                continue
                            
                            # Check if we should filter for final messages
                            if message_token:
                                # For deepsearch responses, only include chunks tagged as "final"
                                if filter_final_only:
                                    if message_tag == "final":
                                        yield message_token
                                else:
                                    yield message_token
                        except json.JSONDecodeError:
                            # Skip malformed JSON
                            continue
                        except Exception as e:
                            print(f"Error processing streaming response: {str(e)}")
        except requests.exceptions.Timeout:
            print("Timeout error during streaming. Check your network connection or X.com API availability.")
            yield "__TIMEOUT__"
        except requests.exceptions.RequestException as e:
            print(f"Error during streaming: {str(e)}")
            yield f"__ERROR__: {str(e)}"