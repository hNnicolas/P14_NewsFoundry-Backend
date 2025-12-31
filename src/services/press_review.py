from typing import List
from src.models import PressReviewArticleModel

def build_articles_from_chat(messages: list) -> List[PressReviewArticleModel]:
    articles: List[PressReviewArticleModel] = []

    for msg in messages:
        if msg.get("role") != "assistant":
            continue

        content = (msg.get("content") or "").strip()
        if len(content) < 120:
            continue

        title = content.split("\n")[0][:80]

        articles.append(
            PressReviewArticleModel(
                title=title,
                summary=content,
                url=None
            )
        )

        if len(articles) >= 5:
            break

    return articles
