# Custom-Knowledge Chatbot using Gemini

This chatbot contains 2 parts, the index and the chat model. For the index, SentenceTransformers from HuggingFace have been used.

<a>Video tutorial</a>

### Indexing

For text files (.txt, .rtf), use indexing.py.

For PDFs, use pdf_loading.ipynb

Both of them use SentenceTransformers, which are free and do not require an API key.

### Chat bot

All code for the chatbot is in main.py. Gemini is used, which is free up to 60 queries per minute. However, an API key is needed and can be acquired from <a href="https://aistudio.google.com/app/prompts/new_chat?utm_source=agd&utm_medium=referral&utm_campaign=core-cta&utm_content=">Google AI Studio</a> 