import structlog
from typing import List
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.http import models
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from core.config import settings
from core.schemas import Article, ArticleChunk, generate_chunk_id
import asyncio

logger = structlog.get_logger()

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model=settings.EMBEDDING_MODEL,
    google_api_key=settings.GOOGLE_API_KEY
)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception), # In production, filter for 429/503
)
async def embed_chunks(texts: List[str]) -> List[List[float]]:
    return await embeddings.aembed_documents(texts)

async def init_qdrant(client: AsyncQdrantClient):
    """
    Idempotent setup for Qdrant collection and indexes.
    """
    collections = await client.get_collections()
    collection_names = [c.name for c in collections.collections]
    
    if settings.QDRANT_COLLECTION not in collection_names:
        logger.info("creating_qdrant_collection", collection=settings.QDRANT_COLLECTION)
        await client.create_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        )
        
        # Create Payload Indexes
        await client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION,
            field_name="parent_url",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        await client.create_payload_index(
            collection_name=settings.QDRANT_COLLECTION,
            field_name="timestamp",
            field_schema=models.PayloadSchemaType.INT, # Or DATETIME if supported/preferred
        )

async def upsert_article_to_vector_db(client: AsyncQdrantClient, article: Article):
    """
    Chunks, embeds, and upserts an article to Qdrant.
    """
    # 1. Semantic Chunking
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_splits = markdown_splitter.split_text(article.markdown_content)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(header_splits)
    
    chunks = []
    for i, split in enumerate(splits):
        chunk = ArticleChunk(
            chunk_id=generate_chunk_id(str(article.url), i),
            parent_url=article.url,
            text=split.page_content,
            metadata_headers=split.metadata,
            timestamp=article.timestamp
        )
        chunks.append(chunk)
    
    if not chunks:
        logger.warn("no_chunks_generated", url=str(article.url))
        return

    # 2. Batch Embedding (50 at a time)
    batch_size = 50
    all_embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_texts = [c.text for c in batch]
        batch_embeddings = await embed_chunks(batch_texts)
        all_embeddings.extend(batch_embeddings)
    
    # 3. Atomic Upsert: Delete existing points for this URL
    await client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="parent_url",
                        match=models.MatchValue(value=str(article.url))
                    )
                ]
            )
        )
    )
    
    # 4. Batch Upsert
    points = [
        models.PointStruct(
            id=chunk.chunk_id,
            vector=vector,
            payload={
                "parent_url": str(chunk.parent_url),
                "text": chunk.text,
                "headers": chunk.metadata_headers,
                "timestamp": int(chunk.timestamp.timestamp())
            }
        )
        for chunk, vector in zip(chunks, all_embeddings)
    ]
    
    await client.upsert(
        collection_name=settings.QDRANT_COLLECTION,
        points=points
    )
    logger.info("upsert_complete", url=str(article.url), chunks=len(chunks))
