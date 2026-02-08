"""Core data models with Pydantic validation."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field, HttpUrl, validator


class ShopifyStore(BaseModel):
    """Shopify store profile."""
    domain: str = Field(..., description="Store domain")
    is_shopify: bool = Field(default=False)
    product_count: int = Field(default=0, ge=0)
    products: List[dict] = Field(default_factory=list)
    launch_velocity_7d: int = Field(default=0, ge=0)
    launch_velocity_30d: int = Field(default=0, ge=0)
    avg_price: float = Field(default=0.0, ge=0)
    scraped_at: datetime = Field(default_factory=datetime.now)

    @validator('domain')
    def clean_domain(cls, v):
        return v.strip().lower()


class MetaAdData(BaseModel):
    """Meta Ad Library intelligence."""
    domain: str
    ad_count: int = Field(default=0, ge=0)
    active_ads: bool = Field(default=False)
    earliest_ad_date: Optional[datetime] = None
    latest_ad_date: Optional[datetime] = None
    ad_running_days: int = Field(default=0, ge=0)
    scraped_at: datetime = Field(default_factory=datetime.now)


class ContactInfo(BaseModel):
    """Enriched contact data."""
    name: Optional[str] = None
    email: Optional[str] = None
    title: Optional[str] = None
    twitter_handle: Optional[str] = None
    linkedin_url: Optional[HttpUrl] = None
    source: str = Field(default="unknown")


class Lead(BaseModel):
    """Complete lead profile."""
    domain: str
    store: Optional[ShopifyStore] = None
    meta_ads: Optional[MetaAdData] = None
    contact: Optional[ContactInfo] = None
    intent_score: float = Field(default=0.0, ge=0, le=100)
    last_updated: datetime = Field(default_factory=datetime.now)


class OutreachTask(BaseModel):
    """Outreach task for human review."""
    task_id: str
    lead_domain: str
    task_type: str = Field(..., regex="^(TWITTER|EMAIL|LINKEDIN)$")
    contact_name: Optional[str] = None
    handle: Optional[str] = None
    context: str
    priority: str = Field(default="medium", regex="^(high|medium|low)$")
    created_at: datetime = Field(default_factory=datetime.now)
    approved: bool = Field(default=False)
    executed: bool = Field(default=False)


class ProxyConfig(BaseModel):
    """Proxy configuration."""
    host: str
    port: int
    username: Optional[str] = None
    password: Optional[str] = None

    @property
    def url(self) -> str:
        if self.username and self.password:
            return f"http://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"http://{self.host}:{self.port}"
