from dotenv import load_dotenv
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os

load_dotenv(override=True)  # Load API keys from .env file

# Model Selection Configuration
MODEL_PROVIDER = "openai"  # Default to OpenAI

# Model-specific configurations using ChatOpenAI with base_url/api_key overrides
MODEL_CONFIGS = {
    "openai": {
        "model": "gpt-4o-mini",  # or "gpt-4o-mini"
        "base_url": None,  # default OpenAI endpoint
        "api_key_env": "OPENAI_API_KEY"
    },
    "gemini": {
        "model": "gemini-2.0-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GOOGLE_API_KEY"
    },
    "claude": {
        "model": "claude-3-5-sonnet-20241022",
        "base_url": "https://api.anthropic.com/v1/",  # OpenAI-compatible proxy required
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "ollama": {
        "model": "llama3-groq-tool-use",
        "base_url": "http://localhost:11434/v1",
        "api_key_env": "OLLAMA_API_KEY"  # usually not required
    },
    "mistral": {  # NEW!
        "model": "mistral-small-latest", 
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY"
    }
}

def get_llm(provider: str):
    """Initialize and return the appropriate LLM based on provider."""
    if provider not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model provider: {provider}. Choose from: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[provider]
    model_name = config["model"]
    base_url = config.get("base_url")
    api_key_env = config.get("api_key_env")

    api_key = None
    if api_key_env:
        import os
        api_key = os.getenv(api_key_env)
        if not api_key and base_url:  # only warn when non-default endpoint needs a key
            raise ValueError(f"Environment variable {api_key_env} is required for provider '{provider}'.")

    # Initialize ChatOpenAI with optional base_url and api_key overrides
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
    )

def chat_response(message, history, model_provider: str):
    """Main chat function - handles everything."""
    """Fixed: Return NEW history list, don't mutate input."""
    # Step 1: Start with copy of current history
    messages = [SystemMessage(content="You are a helpful assistant.")]
    
    # Copy history to avoid mutation
    chat_history = history.copy() if history else []
    
    # Step 2: Convert to LangChain format
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    
    # Step 3: Add current message
    messages.append(HumanMessage(content=message))
    
    # Step 4: Get response
    llm = get_llm(model_provider)
    response = llm.invoke(messages)
    
    # Step 5: Create NEW history list with response
    new_history = chat_history.copy()  # Fresh copy
    new_history.append({"role": "user", "content": message})           # SHOW QUESTION
    new_history.append({"role": "assistant", "content": response.content})  # SHOW RESPONSE
    
    return new_history, ""  # Return NEW list - history persists!

# Create simple interface
with gr.Blocks() as demo:
    model_options = list(MODEL_CONFIGS.keys())
    gr.Markdown("# Simple Chat Test")

    model_dropdown = gr.Dropdown(
            choices=model_options,
            value=MODEL_PROVIDER,  # default value
            label="Select Model Provider",
            info="Choose which AI model to use for the chat"
        )
    
    # Chatbot - automatically uses messages format in Gradio 6+
    chatbot = gr.Chatbot(
        height=500,
        show_label=True,           # Shows "Chat" label
        container=True,            # Better styling
        # Visual separators happen automatically between user/assistant pairs
    )
    
    # User input
    msg = gr.Textbox(placeholder="Type here...", label="Message")
    
    # Buttons
    with gr.Row():
        send = gr.Button("Send")
        clear = gr.Button("Clear")
    
    # Wire up events
    msg.submit(chat_response, [msg, chatbot,model_dropdown], [chatbot, msg])
    send.click(chat_response, [msg, chatbot,model_dropdown], [chatbot, msg])
    clear.click(lambda: ([], ""), None, [chatbot, msg])

if __name__ == "__main__":
    demo.launch()
