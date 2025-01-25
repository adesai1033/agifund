from typing import Dict, List, Optional, Union
import google.generativeai as genai
from openai import OpenAI
import warnings
import urllib3
from config import (
    GOOGLE_API_KEY, 
    OPENAI_API_KEY, 
    DEEPSEEK_API_KEY,
    validate_credentials
)
from prompt_creator2 import PromptCreator
import argparse

warnings.filterwarnings("ignore", category=urllib3.exceptions.NotOpenSSLWarning)

class ModelInterface:
    def __init__(self):
        self.initialize_clients()
        self.prompt_creator = PromptCreator()
        
    def initialize_clients(self):
        """Initialize API clients for each model provider"""
        # Initialize Google/Gemini
        genai.configure(api_key=GOOGLE_API_KEY)
        self.gemini_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 65536
        }
        
        # Initialize OpenAI
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize Deepseek
        self.deepseek_client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com"
        )

    def call_gemini(
        self,
        prompt: str,
        model: str = "gemini-2.0-flash-thinking-exp-01-21",
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Call Gemini models
        
        Args:
            prompt: The user prompt
            model: Either "gemini-exp-1206" or "gemini-2.0-flash-thinking-exp-01-21"
            system_prompt: Optional system prompt
        """
        try:
            # Model-specific configurations
            if model == "gemini-exp-1206":
                config = {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": 65536
                }
            elif model == "gemini-2.0-flash-thinking-exp-01-21":
                config = {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": 65536
                }
            else:
                raise ValueError(f"Unsupported Gemini model: {model}. Use either 'gemini-exp-1206' or 'gemini-2.0-flash-thinking-exp-01-21'")
            
            # Initialize model with specific config
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config=config
            )
            
            if system_prompt:
                chat = model_instance.start_chat(history=[])
                response = chat.send_message(prompt)
            else:
                response = model_instance.generate_content(prompt)
                
            return response.text
            
        except Exception as e:
            print(f"Error calling Gemini model {model}: {e}")
            return None

    def call_openai(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "o1-mini"
    ) -> str:
        """Call OpenAI models"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=4096
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return None

    def call_deepseek(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "deepseek-reasoner"
    ) -> str:
        """Call Deepseek models"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.deepseek_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error calling Deepseek: {e}")
            return None

    def generate_response(
        self,
        provider: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None
    ) -> str:
        """
        Unified interface to generate responses from any model
        
        Args:
            provider: One of 'gemini', 'openai', or 'deepseek'
            prompt: The user prompt
            system_prompt: Optional system prompt
            model: Optional specific model name
        """
        provider_map = {
            'gemini': self.call_gemini,
            'openai': self.call_openai,
            'deepseek': self.call_deepseek
        }
        
        if provider not in provider_map:
            raise ValueError(f"Unknown provider: {provider}. Must be one of {list(provider_map.keys())}")
            
        return provider_map[provider](
            prompt=prompt,
            system_prompt=system_prompt,
            model=model if model else None
        )

    def generate_response_with_created_prompt(
        self,
        provider: str,
        ticker: str,
        indicator: str,
        include_econ_events: bool = False,
        model: Optional[str] = None
    ) -> str:
        """
        Generate response using a prompt created by PromptCreator
        """
        try:
            # Get the filled prompt from PromptCreator
            filled_prompt = self.prompt_creator.create_prompt(
                model=model or provider,
                ticker=ticker,
                indicator=indicator,
                include_econ_events=include_econ_events
            )
            
            # Generate response based on provider
            if provider.startswith('gemini'):
                return self.call_gemini(
                    prompt=filled_prompt,
                    model=model or "gemini-2.0-flash-thinking-exp-01-21"
                )
            else:
                user_prompt, system_prompt = filled_prompt
                return self.generate_response(
                    provider=provider,
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    model=model
                )
                
        except Exception as e:
            print(f"Error generating response: {e}")
            return None

def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    parser = argparse.ArgumentParser(description='Generate model responses using formatted prompts')
    parser.add_argument('--provider', type=str, required=True,
                       choices=['gemini', 'openai', 'deepseek'],
                       help='Model provider to use')
    parser.add_argument('--model', type=str,
                       choices=["claude-sonnet-3.5", "o1-mini", "gemini-exp-1206", "gemini-2.0-flash-thinking-exp-01-21"],
                       help='Specific model to use (optional)')
    parser.add_argument('--ticker', type=str, required=True,
                       help='Stock ticker symbol')
    parser.add_argument('--indicator', type=str, required=True,
                       help='Technical indicator to analyze')
    parser.add_argument('--start_date', type=str, required=True,
                       help='Start date in YYYY_MM format')
    parser.add_argument('--end_date', type=str, required=True,
                       help='End date in YYYY_MM format')
    parser.add_argument('--include_market_events', type=str2bool, default=False,
                       help='Whether to include market events data')

    args = parser.parse_args()
    
    # Initialize model interface
    interface = ModelInterface()
    
    try:
        # Create the prompt using PromptCreator
        creator = interface.prompt_creator
        
        # Override the input() calls in create_prompt by monkey patching
        import builtins
        original_input = builtins.input
        input_values = [args.start_date, args.end_date]
        input_counter = 0
        
        def mock_input(prompt):
            nonlocal input_counter
            value = input_values[input_counter]
            input_counter += 1
            return value
            
        builtins.input = mock_input
        
        try:
            # Get the filled prompt
            filled_prompt = creator.create_prompt(
                model=args.model or args.provider,
                ticker=args.ticker,
                indicator=args.indicator,
                include_econ_events=args.include_market_events
            )
        finally:
            # Restore original input function
            builtins.input = original_input
        
        # Generate response based on provider
        if args.provider == 'gemini':
            response = interface.call_gemini(
                prompt=filled_prompt,
                model=args.model or "gemini-2.0-flash-thinking-exp-01-21"
            )
        else:
            user_prompt, system_prompt = filled_prompt
            response = interface.generate_response(
                provider=args.provider,
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=args.model
            )
        
        print("\nModel Response:")
        print("-" * 80)
        print(response)
        print("-" * 80)
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    '''
    Example usage:
    python model_interface.py --provider gemini --model gemini-2.0-flash-thinking-exp-01-21 --ticker NVDA --indicator bop --start_date 2024_12 --end_date 2024_12 --include_market_events False
    python model_interface.py --provider openai --ticker NVDA --indicator bop --start_date 2024_12 --end_date 2024_12 --include_market_events False
    python model_interface.py --provider deepseek --ticker NVDA --indicator bop --start_date 2024_12 --end_date 2024_12 --include_market_events False
    '''
    import sys
    sys.exit(main()) 