import warnings
import urllib3
warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)

import google.generativeai as genai
from config import GOOGLE_API_KEY, validate_credentials

# Configure the Gemini API with your key
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 65536
}

model = genai.GenerativeModel(
  model_name="gemini-pro",  # Changed to use the stable model name
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
  ]
)

response = chat_session.send_message("If you had to add 2+1, what steps of reasoning would you take? only output this.")

print(response.text)