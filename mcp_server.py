import sys
import io
import os
from contextlib import asynccontextmanager

# --- CẤU HÌNH UTF-8 ---
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    os.environ["PYTHONIOENCODING"] = "utf-8"

import logging
import urllib.parse
import asyncio
from mcp.server.fastmcp import FastMCP
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MCP_Traffic_Server")

playwright_instance = None
browser_instance = None

@asynccontextmanager
async def server_lifespan(server: FastMCP):
    global playwright_instance, browser_instance
    logger.info("🚀 Khởi động Playwright (V6 - Nuclear Mode)...")
    try:
        playwright_instance = await async_playwright().start()
        # Headless=True. Nếu vẫn lỗi, hãy thử đổi thành False để xem nó làm gì
        browser_instance = await playwright_instance.chromium.launch(headless=False)
        logger.info("✅ Browser Ready!")
        yield 
    finally:
        if browser_instance: await browser_instance.close()
        if playwright_instance: await playwright_instance.stop()

app = FastMCP("traffic_law_mcp", lifespan=server_lifespan)

@app.tool()
async def luat_lookup(query: str) -> str:
    """
    Tra cứu luật: Tìm kiếm rộng và lấy dữ liệu thô nếu cần.
    """
    global browser_instance
    if not browser_instance: return "❌ Lỗi: Browser chưa chạy."

    # 1. BỎ site:thuvienphapluat.vn để tìm rộng hơn (LuatVietnam, BaoChinhPhu, v.v.)
    # Điều này giúp tăng khả năng tìm thấy snippet có chứa con số
    search_query = f"{query} mức phạt nghị định 100 123 mới nhất 2025"
    logger.info(f"🔍 Search: {search_query}")
    
    page = None
    try:
        page = await browser_instance.new_page()
        # User-Agent của máy tính thật
        await page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        })

        q = urllib.parse.quote(search_query)
        # Dùng DuckDuckGo bản HTML (nhẹ hơn, dễ lấy text hơn bản JS)
        await page.goto(f"https://html.duckduckgo.com/html/?q={q}", timeout=20000)
        
        # --- CHIẾN THUẬT VÉT CẠN DỮ LIỆU ---
        content = await page.content()
        soup = BeautifulSoup(content, "html.parser")
        
        # 1. Thử lấy các kết quả chuẩn (class .result__body hoặc .result__snippet)
        snippets = []
        valid_links = []
        
        # Quét tất cả thẻ a (link) và div text
        for result in soup.select(".result"):
            title = result.select_one(".result__title")
            snippet = result.select_one(".result__snippet")
            url = result.select_one(".result__url")
            
            text_content = ""
            if title: text_content += title.get_text(" ", strip=True) + ". "
            if snippet: text_content += snippet.get_text(" ", strip=True)
            
            if text_content:
                snippets.append(f"- {text_content}")
            
            if url:
                href = url.get_text().strip()
                if "thuvienphapluat.vn" in href: valid_links.append(href)

        # 2. Dữ liệu dự phòng (Backup): Nếu không bắt được class, lấy TOÀN BỘ TEXT
        backup_data = "\n".join(snippets[:5])
        if len(backup_data) < 50:
             # Lấy toàn bộ chữ trên trang, xóa khoảng trắng thừa
             raw_text = soup.get_text(" ", strip=True)
             # Cắt đoạn giữa (thường là kết quả tìm kiếm)
             mid = len(raw_text) // 2
             start = max(0, mid - 1000)
             backup_data = "DỮ LIỆU THÔ TỪ TÌM KIẾM:\n" + raw_text[start : start + 2000]

        # 3. Ưu tiên vào Link TVPL nếu có (nhưng không bắt buộc)
        detail_content = ""
        if valid_links:
            target_link = "https://" + valid_links[0] if not valid_links[0].startswith("http") else valid_links[0]
            logger.info(f"📖 Thử vào: {target_link}")
            try:
                await page.goto(target_link, timeout=15000)
                sub_soup = BeautifulSoup(await page.content(), "html.parser")
                # Lấy text của body, bỏ script
                for s in sub_soup(["script", "style"]): s.decompose()
                
                # Tìm vùng nội dung chính (thử nhiều class)
                main_div = (sub_soup.select_one(".content-0") or 
                            sub_soup.select_one(".news-content") or 
                            sub_soup.select_one("article") or 
                            sub_soup.body)
                
                detail_content = " ".join(main_div.get_text(" ", strip=True).split())[:3000]
            except:
                pass # Lỗi thì bỏ qua, dùng backup

        # TỔNG HỢP
        if detail_content and len(detail_content) > 200:
            return f"Nguồn: Văn bản pháp luật\nLink: {valid_links[0]}\n\nCHI TIẾT:\n{detail_content}"
        else:
            return f"⚠️ Tìm thấy thông tin tóm tắt (AI hãy tự tổng hợp mức phạt từ đây):\n\n{backup_data}"

    except Exception as e:
        logger.error(f"🔥 Lỗi: {e}")
        return f"Lỗi tra cứu: {e}"
    finally:
        if page: await page.close()

if __name__ == "__main__":
    app.run()