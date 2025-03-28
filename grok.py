"""
Name: Vibhek Soni
Age: 19
Github: https://github.com/vibheksoni
"""
import requests, uuid, json, mimetypes, time, asyncio
import os
from typing import List, Optional, Union, Literal, Dict, Any, AsyncGenerator
import aiohttp
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CREATE_CONVERSATION_URL = "https://x.com/i/api/graphql/{}/CreateGrokConversation"
ADD_RESPONSE_URL = "https://grok.x.com/2/grok/add_response.json"
UPLOAD_FILE_URL = "https://x.com/i/api/2/grok/attachment.json"
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
DEFAULT_QUERY_ID = "vvC5uy7pWWHXS2aDi1FZeA"
MAX_RETRIES_PER_CREDENTIAL = 1

class AllCredentialsLimitedError(Exception):
    """Raised when all available credentials hit the rate limiter."""
    pass

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
            post_ids: Optional[List[str]] = None,
            response_type: Optional[str] = None
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
            self.response_type = response_type

        def __repr__(self) -> str:
            return f"<Result(sender={self.sender}, message={self.message}, type={self.response_type})>"

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
        lines = self.raw_data.strip().splitlines()
        for line in lines:
            if line.strip():
                try:
                    parsed = json.loads(line)
                    result_data = parsed.get("result", {})
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
                        response_type=result_data.get("responseType")
                    )
                    self.results.append(result)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line: {line}. Error: {e}")

    def get_message_tokens(self) -> List[str]:
        """
        Return a list of message tokens from the parsed results.
        """
        return [result.message for result in self.results if result.message is not None]

    def get_full_message(self) -> str:
        """
        Return the full message from the parsed results.
        """
        return ''.join(self.get_message_tokens())

    def is_limiter_response(self) -> bool:
        """Check if any result in the response indicates a rate limit."""
        return any(result.response_type == "limiter" for result in self.results)

    def get_queries(self) -> List[str]:
        return [result.query for result in self.results if result.query]

    def get_feedback_labels(self) -> List[dict]:
        return [result.feedback_labels for result in self.results if result.feedback_labels]

    def get_follow_up_suggestions(self) -> List[dict]:
        return [result.follow_up_suggestions for result in self.results if result.follow_up_suggestions]

    def get_tools_used(self) -> List[dict]:
        return [result.tools_used for result in self.results if result.tools_used]

    def get_cited_web_results(self) -> List[dict]:
        return [result.cited_web_results for result in self.results if result.cited_web_results]

    def get_web_results(self) -> List[dict]:
        return [result.web_results for result in self.results if result.web_results]

    def get_media_post_ids(self) -> List[List[str]]:
        return [result.media_post_ids for result in self.results if result.media_post_ids]

    def get_post_ids(self) -> List[List[str]]:
        return [result.post_ids for result in self.results if result.post_ids]


class Grok:
    """
    Provides methods to manage Grok interactions with credential rotation and retry logic.
    Cycles through provided credentials if a rate limit ('limiter') response is detected.
    """
    def __init__(
        self,
        credential_sets: List[Dict[str, str]]
    ) -> None:
        """
        Initialize with a list of credential sets.
        Each set should be a dict: {"bearer": "...", "csrf": "...", "cookies": "..."}
        """
        if not credential_sets:
             raise ValueError("At least one credential set must be provided.")

        self.credential_sets = credential_sets
        self.num_credentials = len(credential_sets)
        self.current_cred_index = -1

        self.session = requests.Session()
        self.client_uuid = uuid.uuid4().hex

        self.bearer_token: Optional[str] = None
        self.csrf_token: Optional[str] = None
        self.cookies: Optional[str] = None
        self.current_headers: Dict[str, str] = {}

        self._switch_credentials(0)

    def _switch_credentials(self, index: int) -> None:
        """Switches to the credential set at the given index and updates headers."""
        if not (0 <= index < self.num_credentials):
            raise IndexError("Credential index out of bounds.")

        self.current_cred_index = index
        current_creds = self.credential_sets[index]
        self.bearer_token = current_creds.get("bearer")
        self.csrf_token = current_creds.get("csrf")
        self.cookies = current_creds.get("cookies")

        if not all([self.bearer_token, self.csrf_token, self.cookies]):
             logger.warning(f"Credential set at index {index} is incomplete. Missing bearer, csrf, or cookies.")

        logger.info(f"Switched to credential set #{index + 1}/{self.num_credentials}")
        self._update_headers()

    def _get_next_credentials(self) -> bool:
        """Cycles to the next credential set. Returns True if wrapped around, False otherwise."""
        next_index = (self.current_cred_index + 1) % self.num_credentials
        wrapped = next_index == 0
        self._switch_credentials(next_index)
        return wrapped

    def _update_headers(self) -> None:
        """Sets or updates the session headers and the self.current_headers dict."""
        headers = {
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br",
            "accept-language": "en-US,en;q=0.9",
            "authorization": f"Bearer {self.bearer_token}",
            "content-type": "application/json",
            "cookie": self.cookies,
            "origin": "https://x.com",
            "priority": "u=1, i",
            "referer": "https://x.com/i/grok",
            "sec-ch-ua": "\"Google Chrome\";v=\"131\", \"Chromium\";v=\"131\", \"Not_A Brand\";v=\"24\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "user-agent": DEFAULT_USER_AGENT,
            "x-client-uuid": self.client_uuid,
            "x-csrf-token": self.csrf_token,
            "x-twitter-active-user": "yes",
            "x-twitter-auth-type": "OAuth2Session",
            "x-twitter-client-language": "en"
        }
        self.session.headers.clear()
        self.session.headers.update(headers)
        self.current_headers = headers

    async def _execute_with_retry(self, func, *args, **kwargs):
        """
        Executes an async function with credential rotation on limiter errors.
        `func` should be an async function that returns a response object or raises an exception.
        It expects the response object to have a `text` attribute (or be awaitable for text)
        and potentially a `status` attribute.
        """
        initial_cred_index = self.current_cred_index
        attempt_count = 0

        while attempt_count < self.num_credentials:
            current_index_for_attempt = self.current_cred_index
            logger.debug(f"Attempt {attempt_count + 1}/{self.num_credentials} using credential #{current_index_for_attempt + 1}")
            try:
                response = await func(*args, **kwargs)


                response_text = ""
                is_limiter = False

                if isinstance(response, requests.Response):
                    response.raise_for_status()
                    response_text = response.text
                    try:
                        parsed_data = json.loads(response_text)
                        if isinstance(parsed_data, list):
                             for item in parsed_data:
                                 if item.get("result", {}).get("responseType") == "limiter":
                                     is_limiter = True
                                     break
                        elif parsed_data.get("result", {}).get("responseType") == "limiter":
                            is_limiter = True
                        elif "data" in parsed_data and "create_grok_conversation" in parsed_data["data"] and \
                             parsed_data["data"]["create_grok_conversation"] is None and "errors" in parsed_data:
                             if any("rate limit" in err.get("message","").lower() for err in parsed_data.get("errors",[])):
                                 is_limiter = True
                                 logger.warning("Detected potential rate limit from GraphQL errors.")

                    except json.JSONDecodeError:
                        pass
                    except Exception as e:
                        logger.error(f"Error checking non-streaming response for limiter: {e}")


                if is_limiter:
                    logger.warning(f"Limiter detected for credential #{current_index_for_attempt + 1}. Switching credentials.")
                    wrapped = self._get_next_credentials()
                    attempt_count += 1
                    if wrapped and attempt_count >= self.num_credentials:
                        raise AllCredentialsLimitedError("All credentials hit the rate limiter.")
                    await asyncio.sleep(1)
                    continue

                return response

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for credential #{current_index_for_attempt + 1}: {e}")
                is_limiter_http = False
                if e.response is not None:
                    if e.response.status_code == 429:
                        is_limiter_http = True
                        logger.warning(f"HTTP 429 (Too Many Requests) received for credential #{current_index_for_attempt + 1}.")
                    else:
                        try:
                            error_json = e.response.json()
                            if error_json.get("result", {}).get("responseType") == "limiter":
                                is_limiter_http = True
                                logger.warning(f"Limiter JSON found in error response body for credential #{current_index_for_attempt + 1}.")
                        except (json.JSONDecodeError, AttributeError):
                            pass

                if is_limiter_http:
                    wrapped = self._get_next_credentials()
                    attempt_count += 1
                    if wrapped and attempt_count >= self.num_credentials:
                        raise AllCredentialsLimitedError("All credentials hit the rate limiter (HTTP 429 or error body).")
                    await asyncio.sleep(1)
                    continue

                raise ConnectionError(f"Request failed: {e}") from e

            except (json.JSONDecodeError, KeyError, AttributeError, IndexError) as e:
                 logger.error(f"Error processing response for credential #{current_index_for_attempt + 1}: {e}")
                 raise ConnectionError(f"Failed to process response: {e}") from e

            except Exception as e:
                logger.error(f"Unexpected error during request with credential #{current_index_for_attempt + 1}: {e.__class__.__name__}: {e}")
                raise

        raise AllCredentialsLimitedError("Failed to get a successful response after trying all credentials.")


    async def create_new_conversation(self) -> str:
        """
        Creates a *new* Grok conversation, handling retries on limiter errors. (Async)
        """
        logger.info("Attempting to create new Grok conversation...")
        query_id = DEFAULT_QUERY_ID
        data = {"variables": {}, "queryId": query_id}
        url = CREATE_CONVERSATION_URL.format(query_id)

        async def _make_request():
            return await asyncio.to_thread(
                self.session.post,
                url,
                json=data,
                timeout=20
            )

        response = await self._execute_with_retry(_make_request)

        try:
            response_json = response.json()
            conv_id = response_json.get("data", {}).get("create_grok_conversation", {}).get("conversation_id")
            if not conv_id:
                logger.error(f"Could not extract conversation_id from response: {response_json}")
                raise ConnectionError("Failed to create Grok conversation: Invalid response format after retry.")
            logger.info(f"New Grok conversation created: {conv_id} using credential #{self.current_cred_index + 1}")
            return conv_id
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
             logger.error(f"Error parsing successful conversation creation response: {e}")
             raise ConnectionError(f"Failed to parse Grok conversation response after retry: {e}") from e


    async def upload_file(self, file_path: str) -> List[dict]:
        """
        Upload a file asynchronously, handling retries on limiter errors. (Async)
        """
        original_content_type = self.session.headers.pop("content-type", None)

        content_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'
        filename = os.path.basename(file_path)
        logger.info(f"Attempting to upload file: {filename}")

        try:
            with open(file_path, "rb") as f:
                file_content = f.read()

            files = {
                "image": (
                    filename,
                    file_content,
                    content_type
                )
            }

            async def _make_request():
                 upload_headers = self.current_headers.copy()
                 upload_headers.pop("content-type", None)
                 return await asyncio.to_thread(
                     self.session.post,
                     UPLOAD_FILE_URL,
                     files=files,
                     headers=upload_headers,
                     timeout=30
                 )

            response = await self._execute_with_retry(_make_request)

            try:
                response_json = response.json()
                if isinstance(response_json, list) and response_json:
                     media_id = response_json[0].get("mediaId")
                     if media_id:
                         response_json[0]["url"] = f"https://api.x.com/2/grok/attachment.json?mediaId={media_id}"
                     logger.info(f"File {filename} uploaded successfully using credential #{self.current_cred_index + 1}")
                     return response_json
                else:
                     logger.warning(f"Unexpected upload response format: {response_json}")
                     return []
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                logger.error(f"Error parsing upload response for {filename}: {e}")
                raise IOError(f"Failed to parse upload response after retry: {e}") from e

        except Exception as e:
            logger.error(f"Error during file upload preparation or execution for {filename}: {e}")
            raise IOError(f"Error during file upload: {e}") from e
        finally:
            if original_content_type:
                self.session.headers["content-type"] = original_content_type
                self.current_headers["content-type"] = original_content_type


    def create_message_payload(
        self,
        model_name: str,
        conversation_id: str,
        imageGenerateCount: int = 0,
        returnSearchResults: bool = True,
        returnCitations: bool = True,
        eagerTweets: bool = True,
        serverHistory: bool = True,
        isDeepsearch: bool = False,
        isReasoning: bool = False
    ) -> dict:
        """
        Create a template for the conversation payload using the specified Grok model and conversation ID.
        (No API call, no retry logic needed here)
        """
        if not isinstance(model_name, str) or not model_name.startswith("grok-"):
             logger.warning(f"Potentially invalid model name '{model_name}'. Expected format 'grok-...'")

        return {
            "responses": [],
            "systemPromptName": "",
            "grokModelOptionId": model_name,
            "conversationId": conversation_id,
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
            "isDeepsearch": isDeepsearch,
            "isReasoning": isReasoning
        }

    def add_user_message_to_payload(
        self,
        request_payload: dict,
        message: str,
        sender: int = 1,
        file_attachments: Optional[List[dict]] = None
    ) -> None:
        """
        Adds a single user message (potentially concatenated history) to the request payload.
        (No API call, no retry logic needed here)
        """
        request_payload["responses"] = [{
            "message": message,
            "sender": sender,
            "fileAttachments": file_attachments if file_attachments else []
        }]

    async def send(self, request_payload: dict) -> str:
        """
        Send the conversation payload (NON-STREAMING), handling retries on limiter errors.
        Returns the full response text.
        """
        logger.info("Attempting to send non-streaming message...")

        async def _make_request():
            return await asyncio.to_thread(
                self.session.post,
                ADD_RESPONSE_URL,
                json=request_payload,
                timeout=180
            )

        response = await self._execute_with_retry(_make_request)

        logger.info(f"Non-streaming message sent successfully using credential #{self.current_cred_index + 1}")
        return response.text


    async def send_stream(self, request_payload: dict) -> AsyncGenerator[str, None]:
        """
        Send the conversation payload and stream the response line by line asynchronously using aiohttp.
        Handles credential rotation on limiter errors detected *at the start* of the stream.
        Yields raw JSON lines (as strings with newline) from the response.
        """
        initial_cred_index = self.current_cred_index
        attempt_count = 0
        timeout = aiohttp.ClientTimeout(total=180)

        while attempt_count < self.num_credentials:
            current_index_for_attempt = self.current_cred_index
            logger.info(f"Attempting stream request {attempt_count + 1}/{self.num_credentials} using credential #{current_index_for_attempt + 1}")

            try:
                async with aiohttp.ClientSession(headers=self.current_headers, timeout=timeout) as session:
                     async with session.post(ADD_RESPONSE_URL, json=request_payload) as response:
                        if not response.ok:
                            error_text = await response.text()
                            logger.warning(f"Stream request failed with HTTP {response.status} for credential #{current_index_for_attempt + 1}. Body: {error_text[:500]}")
                            is_limiter_http = False
                            if response.status == 429:
                                is_limiter_http = True
                            else:
                                try:
                                    error_json = json.loads(error_text)
                                    if error_json.get("result", {}).get("responseType") == "limiter":
                                        is_limiter_http = True
                                except json.JSONDecodeError:
                                    pass

                            if is_limiter_http:
                                wrapped = self._get_next_credentials()
                                attempt_count += 1
                                if wrapped and attempt_count >= self.num_credentials:
                                    raise AllCredentialsLimitedError("All credentials hit the rate limiter (HTTP error during stream attempt).")
                                await asyncio.sleep(1)
                                continue
                            else:
                                response.raise_for_status()


                        buffer = ""
                        first_line_checked = False
                        async for chunk in response.content.iter_any():
                            if not chunk:
                                continue
                            try:
                                buffer += chunk.decode('utf-8', errors='ignore')
                            except UnicodeDecodeError as ude:
                                logger.warning(f"Unicode decode error in chunk: {ude}. Skipping problematic part.")
                                continue

                            while '\n' in buffer:
                                line, buffer = buffer.split('\n', 1)
                                line = line.strip()
                                if not line:
                                    continue

                                if not first_line_checked:
                                    first_line_checked = True
                                    try:
                                        parsed_first = json.loads(line)
                                        if parsed_first.get("result", {}).get("responseType") == "limiter":
                                            logger.warning(f"Limiter detected in first stream chunk for credential #{current_index_for_attempt + 1}. Switching.")
                                            raise StopAsyncIteration("limiter")
                                    except json.JSONDecodeError:
                                        pass
                                    except Exception as e:
                                        logger.error(f"Error checking first stream line for limiter: {e}")

                                yield line + "\n"

                        if buffer.strip():
                             if not first_line_checked:
                                 try:
                                     parsed_final = json.loads(buffer.strip())
                                     if parsed_final.get("result", {}).get("responseType") == "limiter":
                                         logger.warning(f"Limiter detected in final stream buffer for credential #{current_index_for_attempt + 1}. Switching.")
                                         raise StopAsyncIteration("limiter")
                                 except json.JSONDecodeError: pass
                                 except Exception as e: logger.error(f"Error checking final buffer for limiter: {e}")

                             yield buffer.strip() + "\n"

                        logger.info(f"Stream completed successfully using credential #{self.current_cred_index + 1}")
                        return

            except StopAsyncIteration as sai:
                if str(sai) == "limiter":
                    wrapped = self._get_next_credentials()
                    attempt_count += 1
                    if wrapped and attempt_count >= self.num_credentials:
                        raise AllCredentialsLimitedError("All credentials hit the rate limiter during streaming.")
                    await asyncio.sleep(1)
                    continue
                else:
                    raise

            except aiohttp.ClientResponseError as e:
                logger.error(f"Stream request failed (aiohttp {e.status}) for credential #{current_index_for_attempt + 1}: {e.message}")
                raise ConnectionError(f"Stream failed (HTTP {e.status}): {e.message}") from e

            except aiohttp.ClientError as e:
                logger.error(f"Stream connection error for credential #{current_index_for_attempt + 1}: {e}")
                raise ConnectionError(f"Stream connection failed: {e}") from e

            except asyncio.TimeoutError:
                 logger.error(f"Stream request timed out for credential #{current_index_for_attempt + 1} after {timeout.total} seconds.")
                 raise ConnectionError("Request timed out while streaming Grok response.") from asyncio.TimeoutError

            except Exception as e:
                logger.error(f"Unexpected error during streaming with credential #{current_index_for_attempt + 1}: {e.__class__.__name__}: {e}")
                import traceback
                traceback.print_exc()
                raise ConnectionError(f"Unexpected streaming error: {e}") from e

        raise AllCredentialsLimitedError("Failed to get a successful stream after trying all credentials.")