"""
Script to list all available Gemini models
"""
import google.generativeai as genai
import os

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')

if not GEMINI_API_KEY:
    print("ERROR: GEMINI_API_KEY not set in environment variables")
    print("Please set it using: set GEMINI_API_KEY=your_api_key_here")
    exit(1)

# Configure the API
genai.configure(api_key=GEMINI_API_KEY)

print("Fetching available Gemini models...\n")
print("="*80)

try:
    # List all available models
    models = genai.list_models()
    
    print(f"Found {len(list(genai.list_models()))} models:\n")
    
    for model in models:
        print(f"Model Name: {model.name}")
        print(f"Display Name: {model.display_name}")
        print(f"Description: {model.description}")
        print(f"Supported Methods: {', '.join(model.supported_generation_methods)}")
        print(f"Input Token Limit: {model.input_token_limit}")
        print(f"Output Token Limit: {model.output_token_limit}")
        print("-"*80)
    
    print("\n" + "="*80)
    print("Models that support 'generateContent':")
    print("="*80)
    
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"✓ {model.name}")
            print(f"  Display Name: {model.display_name}")
            print(f"  Use as: genai.GenerativeModel('{model.name}')")
            print()

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    print(traceback.format_exc())
