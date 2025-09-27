from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="llama3")

embeddings.embed_query("Hello, world!")
