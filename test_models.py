import google.generativeai as genai

# Warning: Hardcoding raw API keys in plain text is a security risk.
# For production or Streamlit deployments, store this safely inside .streamlit/secrets.toml
API_KEY = "AQ.Ab8RN6KqVAKij5Pvni0hFO1Um_qH4onJHINoLAG3HfprKuflIA"

genai.configure(api_key=API_KEY)

print("Available Models:\n")

try:
    for model in genai.list_models():
        print(f"Model Name: {model.name}")
        print(f"Methods: {model.supported_generation_methods}")
        print("-" * 60)
except Exception as e:
    print(f"Could not retrieve models: {e}")