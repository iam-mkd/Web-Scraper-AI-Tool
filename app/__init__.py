"""
Web Scraper AI Application Package
"""
from app.models import Product, ProductDetail, ResearchRequest, ResearchResponse
from app.agent.service import get_agent_service, AgentService
from app.rag.ingestion import IngestionService
from app.rag.qa import QAService
from app.rag.vector_store import get_vector_store_manager, VectorStoreManager

__all__ = [
    "Product",
    "ProductDetail",
    "ResearchRequest",
    "ResearchResponse",
    "get_agent_service",
    "AgentService",
    "IngestionService",
    "QAService",
    "get_vector_store_manager",
    "VectorStoreManager",
]

