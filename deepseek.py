# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
from config import DEEPSEEK_API_KEY, validate_credentials

try:
    # Validate credentials are present
    validate_credentials()
    
    # Use the Deepseek API key
    print(f"Using Deepseek API Key: {DEEPSEEK_API_KEY}")
    
    # Your Deepseek API logic here
    # For example, configure your Deepseek client
    # deepseek_client.configure(api_key=DEEPSEEK_API_KEY)
    
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False
    )

    print(response.choices[0].message.content)
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")