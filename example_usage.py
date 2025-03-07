"""
Example usage of the GrokAI OpenAI-compatible API.
"""
import os
import requests
from dotenv import load_dotenv

# Load environment variables (optional, only needed if setting API_BASE_URL)
load_dotenv()

# API configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:5000/v1")

def example_openai_client():
    """Example using the OpenAI Python client."""
    try:
        from openai import OpenAI
        
        # Initialize the client with your API base URL
        client = OpenAI(
            api_key="dummy-key",  # Not actually used, but required
            base_url=API_BASE_URL  # Your GrokAI API server address
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
        
    except ImportError:
        print("OpenAI package not installed. Run: pip install openai")
    except Exception as e:
        print(f"Error: {e}")

def example_direct_api_call():
    """Example using direct API calls without any client."""
    try:
        # Define the API endpoint
        endpoint = f"{API_BASE_URL}/chat/completions"
        
        # Create the request payload
        payload = {
            "model": "grok-3",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant powered by Grok AI."},
                {"role": "user", "content": "Explain what makes you unique as an AI."}
            ],
            "stream": False
        }
        
        # Send the request
        response = requests.post(endpoint, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {data['choices'][0]['message']['content']}")
        else:
            print(f"Error: {response.status_code}, {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

def example_streaming_response():
    """Example using streaming responses."""
    try:
        # Define the API endpoint
        endpoint = f"{API_BASE_URL}/chat/completions"
        
        # Create the request payload
        payload = {
            "model": "grok-3",
            "messages": [
                {"role": "user", "content": "Write a short poem about artificial intelligence."}
            ],
            "stream": True
        }
        
        # Send the request with streaming enabled
        with requests.post(endpoint, json=payload, stream=True) as response:
            # Check if the request was successful
            if response.status_code == 200:
                print("Streaming response:")
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: ') and not line_text.endswith('[DONE]'):
                            try:
                                # Process and print each chunk
                                chunk = line_text.replace('data: ', '')
                                import json
                                data = json.loads(chunk)
                                content = data['choices'][0]['delta'].get('content', '')
                                if content:
                                    print(content, end='', flush=True)
                            except Exception as e:
                                print(f"\nError processing chunk: {e}")
                print("\nStreaming complete.")
            else:
                print(f"Error: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("===== Testing OpenAI Client =====")
    example_openai_client()
    
    print("\n===== Testing Direct API Call =====")
    example_direct_api_call()
    
    print("\n===== Testing Streaming Response =====")
    example_streaming_response()