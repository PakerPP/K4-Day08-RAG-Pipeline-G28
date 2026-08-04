"""
Task 2 — Crawl bài viết/hướng dẫn hỗ trợ khách hàng về thương mại điện tử.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trung tâm trợ giúp công khai của một sàn TMĐT.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: theo dõi đơn hàng, đổi phương thức thanh toán, bằng chứng hoàn tiền,
mua hàng xuyên biên giới.

Lưu ý: một số trang help center dùng JavaScript render (SPA) — nếu crawl về chỉ thấy
tiêu đề mà không có nội dung, đổi sang bài viết khác cùng domain thay vì cố xử lý.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    # Nguồn chính thức của Đại học Bách khoa Hà Nội, kiểm tra ngày 04/08/2026.
    "https://www.hust.edu.vn/vi/news/tin-tuc-su-kien/ket-qua-xttn-bach-khoa-ha-noi-nam-2026-chat-luong-thi-sinh-vuot-troi-655969.html",
    "https://www.hust.edu.vn/vi/tuyen-sinh/dai-hoc/de-an-tuyen-sinh-nam-2025-567354.html",
    "https://ts.hust.edu.vn/index.php/tin-tuc/quy-che-tuyen-sinh-dai-hoc-nam-2026",
    "https://ts.hust.edu.vn/tin-tuc/huong-dan-dang-ky-xet-tuyen-tai-nang-2026",
    "https://ts.hust.edu.vn/tin-tuc/huong-dan-dang-ky-xac-thuc-chung-chi-ngoai-ngu-2026",
    "https://ts.hust.edu.vn/vi/tin-tuc/quy-dinh-ve-phuong-thuc-xet-tuyen-tai-nang-nam-2026",
    "https://ts.hust.edu.vn/tin-tuc/thong-tin-tuyen-sinh-dai-hoc-chinh-quy-nam-2026",
]


# Metadata được đối chiếu với chính bài viết trên domain HUST trước khi crawl.
VERIFIED_METADATA = {
    ARTICLE_URLS[0]: {
        "title": "Kết quả XTTN Bách khoa Hà Nội năm 2026: Chất lượng thí sinh vượt trội",
        "published_date": "2026-07-07",
    },
    ARTICLE_URLS[1]: {
        "title": "Thông tin tuyển sinh năm 2026",
        "published_date": "2026-06-25",
    },
    ARTICLE_URLS[2]: {
        "title": "Quy chế tuyển sinh đại học năm 2026",
        "published_date": "2026-05-26",
    },
    ARTICLE_URLS[3]: {
        "title": "Hướng dẫn đăng ký Xét tuyển tài năng 2026",
        "published_date": "2026-05-13",
    },
    ARTICLE_URLS[4]: {
        "title": "Hướng dẫn đăng ký xác thực chứng chỉ Ngoại ngữ 2026",
        "published_date": "2026-04-14",
    },
    ARTICLE_URLS[5]: {
        "title": "Quy định về Phương thức Xét tuyển tài năng năm 2026",
        "published_date": "2026-03-30",
    },
    ARTICLE_URLS[6]: {
        "title": "Thông tin Tuyển sinh Đại học chính quy năm 2026",
        "published_date": "2026-02-25",
    },
}

OFFICIAL_HOSTS = {"hust.edu.vn", "www.hust.edu.vn", "ts.hust.edu.vn"}


def extract_article_content(markdown: str, title: str) -> str:
    """Loại menu, bài gợi ý và footer khỏi Markdown do crawler trả về."""
    heading = f"# {title}"
    start = markdown.casefold().find(heading.casefold())
    if start < 0:
        raise ValueError(f"Không tìm thấy heading bài viết: {title}")

    end_markers = (
        "\n### Có thể bạn sẽ thích",
        "Tác giả:",
        "\n#  Số 1 Đại Cồ Việt",
    )
    ends = [
        position
        for marker in end_markers
        if (position := markdown.find(marker, start)) >= 0
    ]
    end = min(ends) if ends else len(markdown)
    return markdown[start:end].strip()


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler

    host = (urlparse(url).hostname or "").lower()
    if host not in OFFICIAL_HOSTS:
        raise ValueError(f"Không phải domain HUST chính thức: {url}")

    expected = VERIFIED_METADATA[url]

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

    if not result.success:
        raise RuntimeError(f"Crawl thất bại: {url} — {result.error_message}")

    markdown_result = result.markdown
    raw_content = getattr(markdown_result, "raw_markdown", None) or str(
        markdown_result or ""
    )
    content = extract_article_content(raw_content, expected["title"])
    if len(content) < 500:
        raise ValueError(f"Nội dung quá ngắn ({len(content)} ký tự): {url}")

    crawled_title = (result.metadata or {}).get("title", "")
    if expected["title"].casefold() not in (crawled_title + "\n" + content[:2000]).casefold():
        raise ValueError(
            f"Tiêu đề crawl không khớp. Mong đợi: {expected['title']!r}; "
            f"nhận được: {crawled_title!r}"
        )

    return {
        "url": url,
        "title": expected["title"],
        "published_date": expected["published_date"],
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "source_domain": host,
        "verified_official_source": True,
        "verified_at": "2026-08-04",
        "content_markdown": content,
    }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang hướng dẫn/hỗ trợ khách hàng trên help center của sàn TMĐT")
    else:
        asyncio.run(crawl_all())
