"""
RAG Engine module for Healthmate.

Integrates local vector retrieval via ChromaDB with the Gemini API to perform
context-augmented response generation for user health queries.
"""
from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY, GEMINI_MODEL, TOP_K, MAX_HISTORY_TURNS
from src.prompts import SYSTEM_PROMPT, build_user_turn
from src.vector_store import retrieve


class HealthRAGChatbot:
    """
    RAG Chatbot handler class to manage query retrieval and API generation lifecycle.
    """

    def __init__(self):
        """
        Initializes the Gemini Client instance using the configured environment API Key.
        
        Raises:
            ValueError: If the GEMINI_API_KEY environment variable is not present.
        """
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY belum di-set. Isi file .env berdasarkan .env.example."
            )
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    def _build_contents(self, query: str, chat_history: list, retrieved_docs: list):
        """
        Formats past conversation history and current retrieved context into Gemini payload objects.

        Args:
            query (str): The raw input question from the user.
            chat_history (list): List of dicts representing conversation turns [{"role": ..., "text": ...}].
            retrieved_docs (list): List of document matches fetched from vector retrieval.

        Returns:
            list: Formatted list of `types.Content` objects for Gemini API request.
        """

        # Limit conversation history buffer to MAX_HISTORY_TURNS to prevent context window overflow
        contents = []
        recent_history = chat_history[-MAX_HISTORY_TURNS:] if chat_history else []
        for turn in recent_history:
            contents.append(
                types.Content(
                    role=turn["role"], parts=[types.Part(text=turn["text"])]
                )
            )

        # Inject retrieved document context into the current user turn
        current_turn_text = build_user_turn(query, retrieved_docs)
        contents.append(
            types.Content(role="user", parts=[types.Part(text=current_turn_text)])
        )
        return contents

    def ask(self, query: str, chat_history: list = None):
        """
        Executes the end-to-end RAG pipeline: vector search followed by API generation.

        Args:
            query (str): The question asked by the user.
            chat_history (list, optional): Active chat history list. Defaults to None.

        Returns:
            dict: Contains 'answer' (str) and 'sources' (list of retrieved documents).
        """

        chat_history = chat_history or []

        # 1. Local similarity search via ChromaDB vector index
        retrieved_docs = retrieve(query, top_k=TOP_K)

        # 2. Build structured payload incorporating history and retrieved docs
        contents = self._build_contents(query, chat_history, retrieved_docs)

        # 3. Request completion from Gemini API using the assembled context
        response = self.client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
                max_output_tokens=800,
            ),
        )

        answer_text = response.text or "Maaf, aku belum bisa menjawab itu sekarang."

        return {
            "answer": answer_text,
            "sources": retrieved_docs,
        }
