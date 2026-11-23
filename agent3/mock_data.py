# Файл: mock_data.py
try:
    from .schemas import EvidenceItem, ModelCandidate
except ImportError:
    from schemas import EvidenceItem, ModelCandidate

def make_ev(value):
    """Помощник для быстрой генерации EvidenceItem"""
    return EvidenceItem(value=value, source="mock_catalog", status="catalog")

MOCK_CANDIDATES = [
    # LLM (5 штук) 
    ModelCandidate(
        model_id="qwen-2.5-7b",
        name="Qwen 2.5 7B",
        organization="Alibaba",
        role="llm",
        license="Apache 2.0",
        languages=["ru", "en", "multilingual"],
        context_length=make_ev(32000),
        parameters_count_b=make_ev(7.0), \
        is_local_compatible=True,
        source_url="https://huggingface.co/Qwen/Qwen2.5-7B"
    ),
    ModelCandidate(
        model_id="llama-3-8b-base",
        name="Llama 3 8B (Base)",
        organization="Meta",
        role="llm",
        license="Llama License",
        languages=["en"], 
        context_length=make_ev(8192),
        parameters_count_b=make_ev(8.0),
        is_local_compatible=True,
        source_url="https://huggingface.co/meta-llama/Meta-Llama-3-8B"
    ),
    ModelCandidate(
        model_id="saiga-llama-3-8b",
        name="Saiga Llama 3 8B",
        organization="IlyaGusev",
        role="llm",
        license="Llama License",
        languages=["ru", "en"],
        context_length=make_ev(8192),
        parameters_count_b=make_ev(8.0), 
        is_local_compatible=True,
        source_url="https://huggingface.co/IlyaGusev/saiga_llama3_8b"
    ),
    ModelCandidate(
        model_id="phi-3-mini",
        name="Phi-3 Mini",
        organization="Microsoft",
        role="llm",
        license="MIT",
        languages=["ru", "en", "multilingual"],
        context_length=make_ev(4096),
        parameters_count_b=make_ev(3.8), 
        is_local_compatible=True,
        source_url="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct"
    ),
    ModelCandidate(
        model_id="qwen-2.5-14b",
        name="Qwen 2.5 14B",
        organization="Alibaba",
        role="llm",
        license="Apache 2.0",
        languages=["ru", "en", "multilingual"],
        context_length=make_ev(32000),
        parameters_count_b=make_ev(14.0), 
        is_local_compatible=True,
        source_url="https://huggingface.co/Qwen/Qwen2.5-14B"
    ),

    # Embedding (4 штуки) 
    ModelCandidate(
        model_id="bge-m3",
        name="BGE M3",
        organization="BAAI",
        role="embedding",
        license="MIT",
        languages=["ru", "en", "multilingual"],
        context_length=make_ev(8192),
        parameters_count_b=make_ev(0.57), 
        is_local_compatible=True,
        source_url="https://huggingface.co/BAAI/bge-m3"
    ),
    ModelCandidate(
        model_id="nomic-embed-text-v1.5",
        name="Nomic Embed Text v1.5",
        organization="Nomic AI",
        role="embedding",
        license="Apache 2.0",
        languages=["ru", "en", "multilingual"],
        context_length=make_ev(8192),
        parameters_count_b=make_ev(0.13),
        is_local_compatible=True,
        source_url="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5"
    ),
    ModelCandidate(
        model_id="e5-mistral-7b",
        name="E5 Mistral 7B",
        organization="Intfloat",
        role="embedding",
        license="MIT",
        languages=["ru", "en", "multilingual"],
        context_length=make_ev(32000),
        parameters_count_b=make_ev(7.1), 
        is_local_compatible=True,
        source_url="https://huggingface.co/intfloat/e5-mistral-7b-instruct"
    ),
    ModelCandidate(
        model_id="text-embedding-3-small",
        name="OpenAI text-embedding-3-small",
        organization="OpenAI",
        role="embedding",
        license="Proprietary",
        languages=["ru", "en", "multilingual"],
        context_length=make_ev(8192),
        parameters_count_b=make_ev(0.1), 
        is_local_compatible=False, 
        source_url="https://platform.openai.com/docs/guides/embeddings"
    ),

    # Reranker (4 штуки) 
    ModelCandidate(
        model_id="bge-reranker-v2-m3",
        name="BGE Reranker V2 M3",
        organization="BAAI",
        role="reranker",
        license="MIT",
        languages=["ru", "en", "multilingual"],
        context_length=make_ev(8192),
        parameters_count_b=make_ev(0.57),
        is_local_compatible=True,
        source_url="https://huggingface.co/BAAI/bge-reranker-v2-m3"
    ),
    ModelCandidate(
        model_id="jina-reranker-v2-base-multilingual",
        name="Jina Reranker v2 Base",
        organization="Jina AI",
        role="reranker",
        license="Apache 2.0",
        languages=["ru", "en", "multilingual"],
        context_length=make_ev(1024),
        parameters_count_b=make_ev(0.27),
        is_local_compatible=True,
        source_url="https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual"
    ),
    ModelCandidate(
        model_id="bge-reranker-large",
        name="BGE Reranker Large",
        organization="BAAI",
        role="reranker",
        license="MIT",
        languages=["ru", "en", "multilingual"],
        context_length=make_ev(512),
        parameters_count_b=make_ev(1.0), 
        is_local_compatible=True,
        source_url="https://huggingface.co/BAAI/bge-reranker-large"
    ),
    ModelCandidate(
        model_id="cohere-rerank-v3",
        name="Cohere Rerank 3",
        organization="Cohere",
        role="reranker",
        license="Proprietary",
        languages=["ru", "en", "multilingual"],
        context_length=make_ev(4096),
        parameters_count_b=make_ev(1.0),
        is_local_compatible=False,
        source_url="https://cohere.com/rerank"
    )
]