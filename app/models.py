from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Product(BaseModel):
    name: str
    price: float
    capacity_gb: int
    speed_mhz: Optional[int] = None
    cl_latency: Optional[int] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    form_factor: Optional[str] = None  # "UDIMM" (Desktop) or "SODIMM" (Laptop)
    kit_size: Optional[str] = None     # e.g., "1x8GB", "2x16GB", "1x16GB"
    value_score: Optional[float] = None # 0.0 to 100.0
    voltage: Optional[str] = None
    warranty: Optional[str] = None
    tech_specs: Optional[Dict[str, str]] = None
    highlights: Optional[List[str]] = None
    url: str


class ProductDetail(BaseModel):
    name: str
    price: Optional[float] = None
    capacity_gb: Optional[int] = None
    speed_mhz: Optional[int] = None
    cl_latency: Optional[int] = None
    voltage: Optional[str] = None
    warranty: Optional[str] = None
    form_factor: Optional[str] = None
    kit_size: Optional[str] = None
    brand: Optional[str] = None
    model_number: Optional[str] = None
    tech_specs: Dict[str, str] = Field(default_factory=dict)
    highlights: List[str] = Field(default_factory=list)
    url: str


class ResearchRequest(BaseModel):
    task: str = Field(
        ...,
        min_length=2,
        description="Natural language shopping or research task (e.g. 'Find me the best DDR5 RAM deals on Amazon under ₹50,000')",
    )


class ResearchResponse(BaseModel):
    task: str
    tool_called: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    tools_called: List[str] = Field(default_factory=list)
    steps_taken: List[Dict[str, Any]] = Field(default_factory=list)
    stats: Optional[Dict[str, Any]] = None
    products: List[Product] = Field(default_factory=list)
    detailed_products: List[Dict[str, Any]] = Field(default_factory=list)
    comparison: Optional[Dict[str, Any]] = None
    final_answer: str
    model: str


class AgentTaskRequest(BaseModel):
    task: str = Field(
        ...,
        min_length=2,
        description="Natural language task or command for the agent (e.g. 'Find me DDR5 RAM deals on Amazon under 10000')",
    )


class AgentTaskResponse(BaseModel):
    task: str
    tool_called: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    tools_called: Optional[List[str]] = None
    steps_taken: List[Dict[str, Any]] = Field(default_factory=list)
    stats: Optional[Dict[str, Any]] = None
    products: List[Dict[str, Any]] = Field(default_factory=list)
    detailed_products: List[Dict[str, Any]] = Field(default_factory=list)
    comparison: Optional[Dict[str, Any]] = None
    explanation: str
    model: str

