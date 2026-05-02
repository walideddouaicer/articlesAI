import structlog
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from core.schemas import Article
from typing import Optional
import asyncio

logger = structlog.get_logger()

async def scrape_article(url: str) -> Optional[Article]:
    """
    Scrapes an article using Crawl4AI with clean extraction.
    """
    browser_config = BrowserConfig(headless=True)
    run_config = CrawlerRunConfig(
        excluded_tags=["nav", "footer", "header", "aside", "script", "style"],
        remove_overlay_elements=True,
        word_count_threshold=10,
        bypass_cache=True,
    )
    
    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            result = await asyncio.wait_for(
                crawler.arun(url=url, config=run_config),
                timeout=60.0
            )
            
            if result.success:
                logger.info("scrape_success", url=url)
                return Article(
                    url=url,
                    title=result.metadata.get("title", "No Title"),
                    markdown_content=result.markdown
                )
            else:
                logger.error("scrape_failed", url=url, error=result.error_message)
                return None
                
    except asyncio.TimeoutError:
        logger.error("scrape_timeout", url=url)
        return None
    except Exception as e:
        logger.error("scrape_exception", url=url, error=str(e))
        return None
