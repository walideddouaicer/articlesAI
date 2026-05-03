import re
import structlog
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.http_retry.async_handler import AsyncRetryHandler
from core.config import settings
from core.schemas import SlackBriefing, BriefingItem

logger = structlog.get_logger()

class SlackService:
    def __init__(self):
        self.client = AsyncWebClient(token=settings.SLACK_BOT_TOKEN)
        self.client.retry_handlers.append(AsyncRetryHandler(max_retry_count=3))

    def md_to_mrkdwn(self, text: str) -> str:
        """
        Converts standard Markdown to Slack mrkdwn.
        """
        # Convert links [text](url) -> <url|text>
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<\2|\1>', text)
        
        # Convert bold **text** or __text__ -> *text*
        text = re.sub(r'(\*\*|__)(.*?)\1', r'*\2*', text)
        
        # Strip headers ### -> Bold
        text = re.sub(r'#+\s*(.*)', r'*\1*', text)
        
        # Normalize bullets
        text = re.sub(r'^\s*[\-\*]\s+', '• ', text, flags=re.MULTILINE)
        
        return text

    def truncate_text(self, text: str, limit: int = 2900) -> str:
        """
        Safe truncation at word boundary to avoid breaking mrkdwn tags.
        """
        if len(text) <= limit:
            return text
        
        truncated = text[:limit].rsplit(' ', 1)[0]
        return truncated + "..."

    def build_block_kit(self, briefing: SlackBriefing) -> list[dict]:
        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": briefing.header_text}
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn", 
                    "text": self.truncate_text(self.md_to_mrkdwn(briefing.summary_text))
                }
            },
            {"type": "divider"}
        ]

        for item in briefing.items:
            item_text = (
                f"*<{item.source_url}|{item.title}>*\n"
                f"{item.summary}\n"
                f"*Why it matters:* {item.why_it_matters}"
            )
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": self.truncate_text(self.md_to_mrkdwn(item_text))
                }
            })
            
        blocks.append({"type": "divider"})
        blocks.append({
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": briefing.footer_text}]
        })

        return blocks[:50] # Slack limit

    async def send_briefing(self, briefing: SlackBriefing):
        try:
            blocks = self.build_block_kit(briefing)
            await self.client.chat_postMessage(
                channel=settings.SLACK_CHANNEL_ID,
                blocks=blocks,
                text=briefing.summary_text[:150] # Fallback notification text
            )
            logger.info("slack_message_sent")
        except Exception as e:
            logger.error("slack_message_failed", error=str(e))
            raise e

slack_service = SlackService()
