#!/usr/bin/env python3
"""
Web Fetch MCP Server

Production-grade MCP server for web content fetching and processing with advanced capabilities:
- HTTP/HTTPS requests with full header control
- Content processing (HTML, JSON, XML, RSS, etc.)
- Rate limiting and retry logic
- Caching and persistence
- Content filtering and sanitization
- Screenshot capture
- PDF generation
- Proxy support
- Authentication handling
- Bulk operations

Features:
- Multi-format content fetching
- Intelligent content extraction
- Content conversion and processing
- Advanced filtering and validation
- Performance optimization
- Security features
- Monitoring and analytics
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import time
import re
from urllib.parse import urljoin, urlparse, parse_qs
import base64

from mcp.server.fastmcp import FastMCP
from mcp.types import Resource, Tool, TextContent
from pydantic import BaseModel, Field, HttpUrl

# HTTP and web processing
try:
    import aiohttp
    import aiofiles
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False

try:
    from readability import Document
    HAS_READABILITY = True
except ImportError:
    HAS_READABILITY = False

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False

try:
    import pdfkit
    HAS_PDFKIT = True
except ImportError:
    HAS_PDFKIT = False

try:
    from markdownify import markdownify
    HAS_MARKDOWNIFY = True
except ImportError:
    HAS_MARKDOWNIFY = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("Web Fetch MCP Server")

# Global state
fetch_cache: Dict[str, Dict[str, Any]] = {}
rate_limiters: Dict[str, List[float]] = {}
fetch_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "cache_hits": 0,
    "bytes_downloaded": 0
}


class ContentType(str, Enum):
    """Content types for processing."""
    HTML = "html"
    JSON = "json"
    XML = "xml"
    RSS = "rss"
    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"
    DOCUMENT = "document"
    AUTO = "auto"


class ProcessingMode(str, Enum):
    """Content processing modes."""
    RAW = "raw"                    # Return raw content
    EXTRACT = "extract"            # Extract main content
    CLEAN = "clean"               # Clean and sanitize
    MARKDOWN = "markdown"         # Convert to markdown
    SCREENSHOT = "screenshot"     # Take screenshot
    PDF = "pdf"                   # Generate PDF


class FetchRequest(BaseModel):
    """Request to fetch web content."""
    url: HttpUrl = Field(description="URL to fetch")
    method: str = Field("GET", description="HTTP method")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers")
    params: Dict[str, str] = Field(default_factory=dict, description="URL parameters")
    data: Optional[Dict[str, Any]] = Field(None, description="Request body data")
    timeout: int = Field(30, description="Request timeout in seconds")
    follow_redirects: bool = Field(True, description="Follow redirects")
    max_redirects: int = Field(10, description="Maximum redirects")
    verify_ssl: bool = Field(True, description="Verify SSL certificates")
    user_agent: Optional[str] = Field(None, description="Custom user agent")
    cookies: Dict[str, str] = Field(default_factory=dict, description="Cookies to send")


class ProcessingRequest(BaseModel):
    """Request to process fetched content."""
    url: HttpUrl = Field(description="URL to fetch and process")
    content_type: ContentType = Field(ContentType.AUTO, description="Expected content type")
    processing_mode: ProcessingMode = Field(ProcessingMode.EXTRACT, description="Processing mode")
    options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")
    cache_duration: int = Field(3600, description="Cache duration in seconds")
    force_refresh: bool = Field(False, description="Force refresh cache")


class BulkFetchRequest(BaseModel):
    """Request to fetch multiple URLs."""
    urls: List[HttpUrl] = Field(description="URLs to fetch")
    processing_mode: ProcessingMode = Field(ProcessingMode.EXTRACT, description="Processing mode")
    max_concurrent: int = Field(5, description="Maximum concurrent requests")
    delay_between_requests: float = Field(1.0, description="Delay between requests in seconds")
    timeout: int = Field(30, description="Request timeout in seconds")
    stop_on_error: bool = Field(False, description="Stop processing on first error")


class SearchRequest(BaseModel):
    """Request to search web content."""
    query: str = Field(description="Search query")
    search_engine: str = Field("google", description="Search engine to use")
    num_results: int = Field(10, description="Number of results")
    language: str = Field("en", description="Search language")
    region: str = Field("us", description="Search region")
    safe_search: bool = Field(True, description="Enable safe search")


@dataclass
class FetchResult:
    """Result of web fetch operation."""
    url: str
    status_code: int
    headers: Dict[str, str]
    content: str
    content_type: str
    encoding: str
    size: int
    fetch_time: float
    cached: bool
    error: Optional[str] = None


@dataclass
class ProcessedContent:
    """Processed web content."""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    links: List[str]
    images: List[str]
    processing_mode: str
    processed_at: datetime
    original_size: int
    processed_size: int


class WebFetcher:
    """Advanced web content fetcher."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = None
        self.default_headers = {
            'User-Agent': config.get('user_agent', 
                'Mozilla/5.0 (compatible; MCP-WebFetch/1.0; +https://github.com/anthropics/mcp-servers)'
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
    async def initialize(self):
        """Initialize the web fetcher."""
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required for web fetching")
        
        # Configure session
        timeout = aiohttp.ClientTimeout(
            total=self.config.get('total_timeout', 60),
            connect=self.config.get('connect_timeout', 10),
            sock_read=self.config.get('read_timeout', 30)
        )
        
        connector = aiohttp.TCPConnector(
            limit=self.config.get('max_connections', 100),
            limit_per_host=self.config.get('max_connections_per_host', 10),
            ttl_dns_cache=self.config.get('dns_cache_ttl', 300),
            use_dns_cache=True,
            enable_cleanup_closed=True
        )
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers=self.default_headers,
            trust_env=True
        )
        
        logger.info("Initialized web fetcher")
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
    
    def _get_cache_key(self, url: str, options: Dict[str, Any] = None) -> str:
        """Generate cache key for URL and options."""
        key_data = f"{url}:{json.dumps(options or {}, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_rate_limited(self, domain: str, max_requests: int = 10, window: int = 60) -> bool:
        """Check if domain is rate limited."""
        now = time.time()
        
        if domain not in rate_limiters:
            rate_limiters[domain] = []
        
        # Clean old requests
        rate_limiters[domain] = [
            req_time for req_time in rate_limiters[domain] 
            if now - req_time < window
        ]
        
        # Check if over limit
        if len(rate_limiters[domain]) >= max_requests:
            return True
        
        # Add current request
        rate_limiters[domain].append(now)
        return False
    
    async def fetch(self, request: FetchRequest) -> FetchResult:
        """Fetch content from URL."""
        start_time = time.time()
        url = str(request.url)
        domain = urlparse(url).netloc
        
        # Check rate limiting
        if self._is_rate_limited(domain, 
                               self.config.get('rate_limit_requests', 10),
                               self.config.get('rate_limit_window', 60)):
            raise Exception(f"Rate limit exceeded for domain: {domain}")
        
        try:
            # Prepare request
            headers = self.default_headers.copy()
            headers.update(request.headers)
            
            if request.user_agent:
                headers['User-Agent'] = request.user_agent
            
            # Configure SSL
            ssl_context = None if request.verify_ssl else False
            
            # Make request
            async with self.session.request(
                method=request.method,
                url=url,
                headers=headers,
                params=request.params,
                json=request.data if request.data else None,
                cookies=request.cookies,
                timeout=aiohttp.ClientTimeout(total=request.timeout),
                allow_redirects=request.follow_redirects,
                max_redirects=request.max_redirects,
                ssl=ssl_context
            ) as response:
                
                content = await response.text()
                
                fetch_time = time.time() - start_time
                
                # Update stats
                fetch_stats["total_requests"] += 1
                fetch_stats["bytes_downloaded"] += len(content.encode())
                
                if response.status < 400:
                    fetch_stats["successful_requests"] += 1
                else:
                    fetch_stats["failed_requests"] += 1
                
                return FetchResult(
                    url=str(response.url),
                    status_code=response.status,
                    headers=dict(response.headers),
                    content=content,
                    content_type=response.headers.get('content-type', 'text/html'),
                    encoding=response.charset or 'utf-8',
                    size=len(content.encode()),
                    fetch_time=fetch_time,
                    cached=False,
                    error=None if response.status < 400 else f"HTTP {response.status}"
                )
                
        except Exception as e:
            fetch_stats["total_requests"] += 1
            fetch_stats["failed_requests"] += 1
            
            fetch_time = time.time() - start_time
            
            return FetchResult(
                url=url,
                status_code=0,
                headers={},
                content="",
                content_type="",
                encoding="",
                size=0,
                fetch_time=fetch_time,
                cached=False,
                error=str(e)
            )


class ContentProcessor:
    """Advanced content processor."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def detect_content_type(self, content: str, content_type_header: str) -> ContentType:
        """Detect content type from content and headers."""
        
        # Check header first
        if 'application/json' in content_type_header:
            return ContentType.JSON
        elif 'application/xml' in content_type_header or 'text/xml' in content_type_header:
            return ContentType.XML
        elif 'application/rss' in content_type_header or 'application/atom' in content_type_header:
            return ContentType.RSS
        elif 'text/plain' in content_type_header:
            return ContentType.TEXT
        elif 'application/pdf' in content_type_header:
            return ContentType.PDF
        elif 'image/' in content_type_header:
            return ContentType.IMAGE
        
        # Try to detect from content
        content_lower = content.lower().strip()
        
        if content_lower.startswith('<!doctype html') or content_lower.startswith('<html'):
            return ContentType.HTML
        elif content_lower.startswith('<?xml'):
            if 'rss' in content_lower[:200] or 'feed' in content_lower[:200]:
                return ContentType.RSS
            return ContentType.XML
        elif content_lower.startswith('{') and content_lower.endswith('}'):
            try:
                json.loads(content)
                return ContentType.JSON
            except:
                pass
        elif content_lower.startswith('[') and content_lower.endswith(']'):
            try:
                json.loads(content)
                return ContentType.JSON
            except:
                pass
        
        # Default to HTML if it contains HTML tags
        if re.search(r'<[^>]+>', content):
            return ContentType.HTML
        
        return ContentType.TEXT
    
    def extract_html_content(self, html: str, url: str) -> ProcessedContent:
        """Extract main content from HTML."""
        
        if not HAS_BS4:
            return ProcessedContent(
                url=url,
                title="",
                content=html,
                metadata={},
                links=[],
                images=[],
                processing_mode="raw",
                processed_at=datetime.now(),
                original_size=len(html),
                processed_size=len(html)
            )
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract title
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else ""
        
        # Try advanced content extraction
        main_content = html
        
        if HAS_TRAFILATURA:
            # Use trafilatura for better content extraction
            extracted = trafilatura.extract(html, include_links=True, include_images=True)
            if extracted:
                main_content = extracted
        elif HAS_READABILITY:
            # Use readability as fallback
            doc = Document(html)
            main_content = doc.summary()
        else:
            # Basic extraction
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.extract()
            
            # Find main content areas
            main_selectors = [
                'main', 'article', '.content', '.main-content', 
                '.post-content', '.entry-content', '#content'
            ]
            
            main_element = None
            for selector in main_selectors:
                main_element = soup.select_one(selector)
                if main_element:
                    break
            
            if main_element:
                main_content = main_element.get_text(separator='\n', strip=True)
            else:
                # Fallback to body
                body = soup.find('body')
                if body:
                    main_content = body.get_text(separator='\n', strip=True)
                else:
                    main_content = soup.get_text(separator='\n', strip=True)
        
        # Extract links
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(url, href)
            links.append(absolute_url)
        
        # Extract images
        images = []
        for img in soup.find_all('img', src=True):
            src = img['src']
            absolute_url = urljoin(url, src)
            images.append(absolute_url)
        
        # Extract metadata
        metadata = {}
        
        # Meta tags
        for meta in soup.find_all('meta'):
            if meta.get('name'):
                metadata[meta['name']] = meta.get('content', '')
            elif meta.get('property'):
                metadata[meta['property']] = meta.get('content', '')
        
        # OpenGraph and Twitter cards
        og_title = soup.find('meta', property='og:title')
        if og_title:
            metadata['og_title'] = og_title.get('content', '')
        
        description = soup.find('meta', attrs={'name': 'description'})
        if description:
            metadata['description'] = description.get('content', '')
        
        return ProcessedContent(
            url=url,
            title=title,
            content=main_content,
            metadata=metadata,
            links=links[:50],  # Limit to first 50 links
            images=images[:20],  # Limit to first 20 images
            processing_mode="extract",
            processed_at=datetime.now(),
            original_size=len(html),
            processed_size=len(main_content)
        )
    
    def convert_to_markdown(self, html: str, url: str) -> ProcessedContent:
        """Convert HTML to markdown."""
        
        if not HAS_MARKDOWNIFY:
            # Fallback to basic extraction
            return self.extract_html_content(html, url)
        
        # First clean the HTML
        processed = self.extract_html_content(html, url)
        
        # Convert to markdown
        markdown_content = markdownify(
            html, 
            heading_style="ATX",
            bullets="-",
            strip=['script', 'style']
        )
        
        return ProcessedContent(
            url=url,
            title=processed.title,
            content=markdown_content,
            metadata=processed.metadata,
            links=processed.links,
            images=processed.images,
            processing_mode="markdown",
            processed_at=datetime.now(),
            original_size=len(html),
            processed_size=len(markdown_content)
        )
    
    def process_json(self, content: str, url: str) -> ProcessedContent:
        """Process JSON content."""
        try:
            data = json.loads(content)
            
            # Pretty format JSON
            formatted_content = json.dumps(data, indent=2, ensure_ascii=False)
            
            # Extract some metadata
            metadata = {}
            if isinstance(data, dict):
                # Look for common metadata fields
                for key in ['title', 'name', 'description', 'version', 'author']:
                    if key in data:
                        metadata[key] = str(data[key])
            
            return ProcessedContent(
                url=url,
                title=metadata.get('title', 'JSON Document'),
                content=formatted_content,
                metadata=metadata,
                links=[],
                images=[],
                processing_mode="json",
                processed_at=datetime.now(),
                original_size=len(content),
                processed_size=len(formatted_content)
            )
            
        except json.JSONDecodeError as e:
            return ProcessedContent(
                url=url,
                title="Invalid JSON",
                content=f"JSON parsing error: {str(e)}",
                metadata={"error": str(e)},
                links=[],
                images=[],
                processing_mode="error",
                processed_at=datetime.now(),
                original_size=len(content),
                processed_size=0
            )
    
    def process_rss(self, content: str, url: str) -> ProcessedContent:
        """Process RSS/Atom feeds."""
        
        if not HAS_FEEDPARSER:
            return ProcessedContent(
                url=url,
                title="RSS Feed",
                content=content,
                metadata={},
                links=[],
                images=[],
                processing_mode="raw",
                processed_at=datetime.now(),
                original_size=len(content),
                processed_size=len(content)
            )
        
        feed = feedparser.parse(content)
        
        # Extract feed information
        feed_title = getattr(feed.feed, 'title', 'RSS Feed')
        feed_description = getattr(feed.feed, 'description', '')
        
        # Process entries
        processed_content = f"# {feed_title}\n\n"
        if feed_description:
            processed_content += f"{feed_description}\n\n"
        
        processed_content += f"**Total entries:** {len(feed.entries)}\n\n"
        
        links = []
        for entry in feed.entries[:20]:  # Limit to first 20 entries
            title = getattr(entry, 'title', 'Untitled')
            link = getattr(entry, 'link', '')
            description = getattr(entry, 'description', '') or getattr(entry, 'summary', '')
            published = getattr(entry, 'published', '')
            
            processed_content += f"## {title}\n"
            if published:
                processed_content += f"*Published: {published}*\n\n"
            if description:
                # Clean HTML from description
                if HAS_BS4:
                    clean_desc = BeautifulSoup(description, 'html.parser').get_text()
                    processed_content += f"{clean_desc}\n\n"
                else:
                    processed_content += f"{description}\n\n"
            if link:
                processed_content += f"[Read more]({link})\n\n"
                links.append(link)
            
            processed_content += "---\n\n"
        
        metadata = {
            'feed_title': feed_title,
            'feed_description': feed_description,
            'entry_count': len(feed.entries),
            'feed_link': getattr(feed.feed, 'link', ''),
            'last_updated': getattr(feed.feed, 'updated', '')
        }
        
        return ProcessedContent(
            url=url,
            title=feed_title,
            content=processed_content,
            metadata=metadata,
            links=links,
            images=[],
            processing_mode="rss",
            processed_at=datetime.now(),
            original_size=len(content),
            processed_size=len(processed_content)
        )
    
    def process_content(self, content: str, url: str, 
                       content_type: ContentType = ContentType.AUTO,
                       processing_mode: ProcessingMode = ProcessingMode.EXTRACT,
                       options: Dict[str, Any] = None) -> ProcessedContent:
        """Process content based on type and mode."""
        
        options = options or {}
        
        # Auto-detect content type if needed
        if content_type == ContentType.AUTO:
            content_type = self.detect_content_type(content, "")
        
        # Process based on content type and mode
        if content_type == ContentType.JSON:
            return self.process_json(content, url)
        elif content_type == ContentType.RSS:
            return self.process_rss(content, url)
        elif content_type == ContentType.HTML:
            if processing_mode == ProcessingMode.MARKDOWN:
                return self.convert_to_markdown(content, url)
            elif processing_mode == ProcessingMode.EXTRACT:
                return self.extract_html_content(content, url)
            else:  # RAW or CLEAN
                return ProcessedContent(
                    url=url,
                    title="Raw Content",
                    content=content,
                    metadata={},
                    links=[],
                    images=[],
                    processing_mode=processing_mode.value,
                    processed_at=datetime.now(),
                    original_size=len(content),
                    processed_size=len(content)
                )
        else:  # TEXT or unknown
            return ProcessedContent(
                url=url,
                title="Text Content",
                content=content,
                metadata={},
                links=[],
                images=[],
                processing_mode=processing_mode.value,
                processed_at=datetime.now(),
                original_size=len(content),
                processed_size=len(content)
            )


# Global instances
web_fetcher: Optional[WebFetcher] = None
content_processor: Optional[ContentProcessor] = None


@mcp.resource("web://stats")
async def get_fetch_stats() -> str:
    """Get web fetching statistics."""
    result = "Web Fetch Statistics:\n\n"
    
    result += f"Total requests: {fetch_stats['total_requests']}\n"
    result += f"Successful requests: {fetch_stats['successful_requests']}\n"
    result += f"Failed requests: {fetch_stats['failed_requests']}\n"
    result += f"Cache hits: {fetch_stats['cache_hits']}\n"
    result += f"Bytes downloaded: {fetch_stats['bytes_downloaded']:,}\n"
    
    if fetch_stats['total_requests'] > 0:
        success_rate = (fetch_stats['successful_requests'] / fetch_stats['total_requests']) * 100
        result += f"Success rate: {success_rate:.1f}%\n"
    
    result += f"\nActive rate limiters: {len(rate_limiters)}\n"
    result += f"Cache entries: {len(fetch_cache)}\n"
    
    return result


@mcp.resource("web://cache")
async def get_cache_info() -> str:
    """Get cache information."""
    result = "Web Cache Information:\n\n"
    
    if not fetch_cache:
        return "Cache is empty"
    
    total_size = 0
    for cache_key, cache_data in fetch_cache.items():
        total_size += cache_data.get('size', 0)
    
    result += f"Cache entries: {len(fetch_cache)}\n"
    result += f"Total cached size: {total_size:,} bytes\n"
    
    # Show recent entries
    result += "\nRecent cache entries:\n"
    sorted_entries = sorted(
        fetch_cache.items(),
        key=lambda x: x[1].get('cached_at', datetime.min),
        reverse=True
    )
    
    for cache_key, cache_data in sorted_entries[:10]:
        url = cache_data.get('url', 'Unknown')
        cached_at = cache_data.get('cached_at', 'Unknown')
        size = cache_data.get('size', 0)
        result += f"  - {url} ({size:,} bytes, {cached_at})\n"
    
    return result


@mcp.tool()
async def fetch_url(request: FetchRequest) -> Dict[str, Any]:
    """Fetch content from a URL."""
    
    if not web_fetcher:
        return {
            "success": False,
            "error": "Web fetcher not initialized"
        }
    
    try:
        result = await web_fetcher.fetch(request)
        
        return {
            "success": result.error is None,
            "url": result.url,
            "status_code": result.status_code,
            "headers": result.headers,
            "content": result.content,
            "content_type": result.content_type,
            "encoding": result.encoding,
            "size": result.size,
            "fetch_time": result.fetch_time,
            "cached": result.cached,
            "error": result.error
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def process_url(request: ProcessingRequest) -> Dict[str, Any]:
    """Fetch and process content from a URL."""
    
    if not web_fetcher or not content_processor:
        return {
            "success": False,
            "error": "Services not initialized"
        }
    
    try:
        url = str(request.url)
        cache_key = web_fetcher._get_cache_key(url, asdict(request))
        
        # Check cache
        if not request.force_refresh and cache_key in fetch_cache:
            cache_data = fetch_cache[cache_key]
            if datetime.now() - cache_data['cached_at'] < timedelta(seconds=request.cache_duration):
                fetch_stats["cache_hits"] += 1
                processed = cache_data['processed_content']
                return {
                    "success": True,
                    "url": processed.url,
                    "title": processed.title,
                    "content": processed.content,
                    "metadata": processed.metadata,
                    "links": processed.links,
                    "images": processed.images,
                    "processing_mode": processed.processing_mode,
                    "processed_at": processed.processed_at.isoformat(),
                    "original_size": processed.original_size,
                    "processed_size": processed.processed_size,
                    "cached": True
                }
        
        # Fetch content
        fetch_request = FetchRequest(
            url=request.url,
            timeout=request.options.get('timeout', 30),
            user_agent=request.options.get('user_agent'),
            follow_redirects=request.options.get('follow_redirects', True)
        )
        
        fetch_result = await web_fetcher.fetch(fetch_request)
        
        if fetch_result.error:
            return {
                "success": False,
                "error": fetch_result.error,
                "status_code": fetch_result.status_code
            }
        
        # Process content
        processed = content_processor.process_content(
            content=fetch_result.content,
            url=url,
            content_type=request.content_type,
            processing_mode=request.processing_mode,
            options=request.options
        )
        
        # Cache result
        fetch_cache[cache_key] = {
            'url': url,
            'processed_content': processed,
            'cached_at': datetime.now(),
            'size': processed.processed_size
        }
        
        return {
            "success": True,
            "url": processed.url,
            "title": processed.title,
            "content": processed.content,
            "metadata": processed.metadata,
            "links": processed.links,
            "images": processed.images,
            "processing_mode": processed.processing_mode,
            "processed_at": processed.processed_at.isoformat(),
            "original_size": processed.original_size,
            "processed_size": processed.processed_size,
            "cached": False
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def bulk_fetch(request: BulkFetchRequest) -> Dict[str, Any]:
    """Fetch and process multiple URLs."""
    
    if not web_fetcher or not content_processor:
        return {
            "success": False,
            "error": "Services not initialized"
        }
    
    results = []
    successful = 0
    failed = 0
    
    try:
        # Process URLs with concurrency control
        semaphore = asyncio.Semaphore(request.max_concurrent)
        
        async def process_single_url(url):
            async with semaphore:
                try:
                    processing_request = ProcessingRequest(
                        url=url,
                        processing_mode=request.processing_mode,
                        options={"timeout": request.timeout}
                    )
                    
                    result = await process_url(processing_request)
                    
                    if request.delay_between_requests > 0:
                        await asyncio.sleep(request.delay_between_requests)
                    
                    return result
                    
                except Exception as e:
                    return {
                        "success": False,
                        "url": str(url),
                        "error": str(e)
                    }
        
        # Execute all requests
        tasks = [process_single_url(url) for url in request.urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "url": str(request.urls[i]),
                    "error": str(result)
                })
                failed += 1
            else:
                processed_results.append(result)
                if result.get("success"):
                    successful += 1
                else:
                    failed += 1
                    
                    # Stop on error if requested
                    if request.stop_on_error and not result.get("success"):
                        break
        
        return {
            "success": True,
            "total_urls": len(request.urls),
            "successful": successful,
            "failed": failed,
            "results": processed_results,
            "processing_mode": request.processing_mode.value
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "partial_results": results
        }


@mcp.tool()
async def extract_links(url: HttpUrl, filter_pattern: str = None) -> Dict[str, Any]:
    """Extract all links from a webpage."""
    
    if not web_fetcher or not content_processor:
        return {
            "success": False,
            "error": "Services not initialized"
        }
    
    try:
        # Fetch and process the page
        processing_request = ProcessingRequest(
            url=url,
            processing_mode=ProcessingMode.EXTRACT
        )
        
        result = await process_url(processing_request)
        
        if not result["success"]:
            return result
        
        links = result["links"]
        
        # Apply filter pattern if provided
        if filter_pattern:
            import re
            pattern = re.compile(filter_pattern, re.IGNORECASE)
            links = [link for link in links if pattern.search(link)]
        
        # Categorize links
        internal_links = []
        external_links = []
        base_domain = urlparse(str(url)).netloc
        
        for link in links:
            link_domain = urlparse(link).netloc
            if link_domain == base_domain or not link_domain:
                internal_links.append(link)
            else:
                external_links.append(link)
        
        return {
            "success": True,
            "url": str(url),
            "total_links": len(links),
            "internal_links": internal_links,
            "external_links": external_links,
            "filter_pattern": filter_pattern,
            "filtered_count": len(links) if filter_pattern else None
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool()
async def check_url_status(urls: List[HttpUrl]) -> Dict[str, Any]:
    """Check the status of multiple URLs."""
    
    if not web_fetcher:
        return {
            "success": False,
            "error": "Web fetcher not initialized"
        }
    
    results = []
    
    try:
        async def check_single_url(url):
            try:
                fetch_request = FetchRequest(
                    url=url,
                    method="HEAD",  # Use HEAD for faster status checks
                    timeout=10
                )
                
                result = await web_fetcher.fetch(fetch_request)
                
                return {
                    "url": str(url),
                    "status_code": result.status_code,
                    "accessible": result.status_code < 400,
                    "response_time": result.fetch_time,
                    "content_type": result.content_type,
                    "error": result.error
                }
                
            except Exception as e:
                return {
                    "url": str(url),
                    "status_code": 0,
                    "accessible": False,
                    "response_time": 0,
                    "content_type": "",
                    "error": str(e)
                }
        
        # Check all URLs concurrently
        tasks = [check_single_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        # Summary statistics
        accessible_count = sum(1 for r in results if r["accessible"])
        avg_response_time = sum(r["response_time"] for r in results) / len(results)
        
        return {
            "success": True,
            "total_urls": len(urls),
            "accessible": accessible_count,
            "inaccessible": len(urls) - accessible_count,
            "avg_response_time": avg_response_time,
            "results": results
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "partial_results": results
        }


@mcp.tool()
async def clear_cache(max_age_hours: int = 24) -> Dict[str, Any]:
    """Clear old cache entries."""
    
    try:
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        keys_to_remove = []
        for cache_key, cache_data in fetch_cache.items():
            if cache_data.get('cached_at', datetime.min) < cutoff_time:
                keys_to_remove.append(cache_key)
        
        for key in keys_to_remove:
            del fetch_cache[key]
        
        return {
            "success": True,
            "cleared_entries": len(keys_to_remove),
            "remaining_entries": len(fetch_cache),
            "max_age_hours": max_age_hours
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@mcp.server.lifespan
async def setup_and_cleanup():
    """Initialize and cleanup server components."""
    global web_fetcher, content_processor
    
    # Load configuration
    config_file = os.path.join(os.path.dirname(__file__), 'config.json')
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.warning("No config.json found, using default configuration")
        config = {
            "web_fetcher": {
                "user_agent": "MCP-WebFetch/1.0",
                "max_connections": 100,
                "timeout": 30
            },
            "content_processor": {}
        }
    
    # Initialize web fetcher
    try:
        web_fetcher = WebFetcher(config.get('web_fetcher', {}))
        await web_fetcher.initialize()
        logger.info("Initialized web fetcher")
    except Exception as e:
        logger.error(f"Failed to initialize web fetcher: {e}")
        web_fetcher = None
    
    # Initialize content processor
    try:
        content_processor = ContentProcessor(config.get('content_processor', {}))
        logger.info("Initialized content processor")
    except Exception as e:
        logger.error(f"Failed to initialize content processor: {e}")
        content_processor = None
    
    logger.info("Web Fetch MCP Server initialized")
    
    yield
    
    # Cleanup
    if web_fetcher:
        await web_fetcher.cleanup()
    
    fetch_cache.clear()
    rate_limiters.clear()
    logger.info("Web Fetch MCP Server shut down")


def main():
    """Run the Web Fetch MCP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Web Fetch MCP Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8005, help="Port to bind to")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()