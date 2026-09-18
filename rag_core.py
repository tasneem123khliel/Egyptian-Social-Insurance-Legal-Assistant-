"""Core Arabic RAG pipeline shared by the Streamlit UI (app.py) and the REST API (api.py).

Pipeline: PyPDFLoader -> normalize_arabic -> RecursiveCharacterTextSplitter
           -> multilingual FastEmbed embeddings -> Chroma -> Groq LLM (Arabic prompt)
"""

import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------- Config (overridable with environment variables / Streamlit secrets) ----------
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# ملاحظة: Groq أوقف llama-3.1-8b-instant و llama-3.3-70b-versatile نهائياً (16 أغسطس 2026).
# البديل الرسمي السريع: openai/gpt-oss-20b (أسرع موديل ~1000 tok/s ويجاوب في ثواني).
DEFAULT_LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL", "intfloat/multilingual-e5-large"
)
DEFAULT_TOP_K = int(os.environ.get("TOP_K", "5"))
DEFAULT_PERSIST_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")

ARABIC_PROMPT = """أنت مساعد قانوني متخصص في قانون التأمين الاجتماعي المصري.
أجب عن السؤال باستخدام المعلومات الموجودة في السياق فقط.

قواعد إلزامية:

1. لا تضف أي معلومة غير موجودة صراحة في السياق.
2. لا تغيّر المعنى القانوني للنص.
3. حافظ بدقة على ألفاظ:
   - يسري
   - لا يسري
   - يستثنى
   - باستثناء
   - يشترط
   - لا يشترط
   - يجوز
   - لا يجوز
4. لا تعتبر أي شخص أو فئة "مستثناة" إلا إذا ذكر السياق صراحة أنها مستثناة من تطبيق الحكم.
5. إذا ورد في النص:
   "تسري أحكام هذا القانون على..."
   فلا يجوز تحويلها إلى:
   "يُستثنى من القانون..."
6. عند وجود مادة قانونية مرتبطة بالسؤال، اذكر رقم المادة.
7. إذا كان السياق غير كافٍ للإجابة، قل:
   "لا أعرف بناءً على المستند المقدم."
8. لا تستخدم معرفتك الخارجية بالقانون.
9. عند تلخيص مادة قانونية، حافظ على اتجاه المعنى الأصلي للنص ولا تعكس العلاقة بين الفئات والاستثناءات.
10. ممنوع منعاً باتاً عرض خطوات تفكيرك أو تحليلك أو أي كلام زائد — اكتب الإجابة النهائية فقط مباشرة باللغة العربية الفصحى وبشكل مختصر.

السياق:
{context}

السؤال: {input}

الإجابة:"""


# ---------- Arabic text utilities ----------
def normalize_arabic(text: str) -> str:
    """Unify Arabic text (tashkeel, tatweel, alef/hamza, alef-maqsura) for better retrieval."""
    text = re.sub(r"[ً-ٰٟ]", "", text)
    text = text.replace("ـ", "")
    text = re.sub(r"[أإآا]", "ا", text)
    text = text.replace("ى", "ي")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_query(query: str) -> str:
    return normalize_arabic(query)


def clean_answer(text: str) -> str:
    """Strip any <think>...</think> reasoning remnants from the model output."""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Unclosed <think> (truncated output): everything after it is reasoning -> drop it
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    text = text.replace("</think>", "")
    return text.strip()


# ---------- Pipeline builders ----------
def load_and_split_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Load a PDF, normalize its Arabic text, and split it into chunks."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    for doc in docs:
        doc.page_content = normalize_arabic(doc.page_content)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "،", "؟", ".", ":", ";", " ", ""],
    )
    return docs, splitter.split_documents(docs)


def build_vectorstore(splits, embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                      persist_dir: str | None = DEFAULT_PERSIST_DIR):
    """Embed chunks and store them in Chroma (persisted to disk when persist_dir is set)."""
    embedding = FastEmbedEmbeddings(model_name=embedding_model)
    if persist_dir:
        return Chroma.from_documents(splits, embedding=embedding, persist_directory=persist_dir)
    return Chroma.from_documents(splits, embedding=embedding)


def load_or_build_vectorstore(splits=None, embedding_model: str = DEFAULT_EMBEDDING_MODEL,
                              persist_dir: str | None = DEFAULT_PERSIST_DIR):
    """Reuse a persisted Chroma DB when it already exists, otherwise build it.

    Use this when you downloaded/copied an existing `chroma_db` folder: rebuilding
    with `from_documents` on top of it would duplicate every chunk, while loading
    is instant. The folder must have been built with the SAME embedding model.
    """
    if persist_dir and os.path.isdir(persist_dir) and os.listdir(persist_dir):
        embedding = FastEmbedEmbeddings(model_name=embedding_model)
        return Chroma(persist_directory=persist_dir, embedding_function=embedding)
    if splits is None:
        raise ValueError("No persisted DB found and no splits given to build one.")
    return build_vectorstore(splits, embedding_model=embedding_model, persist_dir=persist_dir)


def build_chain(api_key: str, llm_model: str = DEFAULT_LLM_MODEL,
                temperature: float = 0.2, max_tokens: int = 512,
                timeout: int = 60):
    """Build the RAG chain: Arabic prompt -> Groq LLM -> string output.

    ملاحظة: الـ chain هنا لا يحتوي على الـ retriever بداخله، حتى نعمل
    استرجاع واحد فقط لكل سؤال (النسخة القديمة كانت تسترجع مرتين).
    """
    llm = ChatOpenAI(
        model=llm_model,
        openai_api_base=GROQ_BASE_URL,
        openai_api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=2,
        # تقليل التفكير الداخلي لموديلات gpt-oss => إجابة أسرع بكتير
        extra_body={"reasoning_effort": "low"},
    )
    prompt = PromptTemplate.from_template(ARABIC_PROMPT)
    return prompt | llm | StrOutputParser()


def _format_docs(docs, max_chars: int = 6000) -> str:
    text = "\n\n".join(doc.page_content for doc in docs)
    return text[:max_chars]


def ask_question(chain, retriever, question: str, top_k: int = DEFAULT_TOP_K):
    """Ask in Arabic. Returns (answer: str, sources: list of {page, text, source}).

    `page` is 1-based (human page number, not the 0-based PDF index).
    يعمل استرجاع واحد فقط (سريع) ثم ينادي الـ LLM مرة واحدة.
    """
    normalized = normalize_query(question)
    docs = retriever.invoke(normalized)  # استرجاع واحد فقط
    docs = docs[:top_k]
    context = _format_docs(docs)
    raw = chain.invoke({"context": context, "input": normalized})  # LLM مرة واحدة
    answer = clean_answer(raw)
    sources = [
        {
            "page": int(doc.metadata.get("page", 0)) + 1,
            "text": doc.page_content,
            "source": str(doc.metadata.get("source", "")),
        }
        for doc in docs[:top_k]
    ]
    return answer, sources
