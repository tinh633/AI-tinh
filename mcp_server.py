import asyncio
from mcp.server.fastmcp import FastMCP
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import urllib.parse
import logging
import sys

# 1. Fix lỗi khi chạy trong môi trường không chuẩn (VSCode/Jupyter)
if not hasattr(sys.stdout, 'buffer'):
    try:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    except Exception:
        pass

# Cấu hình log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastMCP("traffic_law_mcp")

# ==============================================================================
# TOOL: Luat Lookup (Sử dụng Playwright)
# ==============================================================================
@app.tool()
async def luat_lookup(query: str) -> str:
    """
    Tra cứu luật từ vbpl.vn sử dụng trình duyệt ảo (Playwright).
    Giúp vượt qua các lỗi chặn bot hoặc lỗi tải trang trắng.
    """
    log.info(f"🔍 Đang tìm kiếm: {query} (bằng Playwright)...")
    
    # Mã hóa từ khóa tìm kiếm
    q = urllib.parse.quote(query)
    search_url = f"https://vbpl.vn/Pages/timkiemvbpl.aspx?Keyword={q}"

    async with async_playwright() as p:
        # Khởi tạo trình duyệt Chrome (chạy ẩn)
        try:
            browser = await p.chromium.launch(headless=False, slow_mo=1000)
        except Exception as e:
            return f"❌ Lỗi: Chưa cài trình duyệt. Hãy chạy lệnh: 'python -m playwright install chromium'. Chi tiết: {e}"

        # Tạo ngữ cảnh giả lập trình duyệt thật
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        try:
            # --- BƯỚC 1: TÌM KIẾM VĂN BẢN ---
            log.info(f"Truy cập: {search_url}")
            # Chờ trang tải xong (domcontentloaded)
            await page.goto(search_url, timeout=30000, wait_until="domcontentloaded")
            
            # Lấy HTML đã render
            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")
            
            # Logic lọc link văn bản
            valid_links = []
            seen = set()
            
            # Tìm các thẻ <a> chứa link 'vbpq-toanvan.aspx'
            for a in soup.find_all("a", href=True):
                href = a.get("href")
                if "vbpq-toanvan.aspx" in href:
                    full_url = "https://vbpl.vn" + href if not href.startswith("http") else href
                    if full_url not in seen:
                        valid_links.append(full_url)
                        seen.add(full_url)
            
            # Nếu chưa tìm thấy, thử tìm theo class title (dự phòng)
            if not valid_links:
                for a in soup.select("a.title"):
                    href = a.get("href")
                    if href:
                        full_url = "https://vbpl.vn" + href if not href.startswith("http") else href
                        if full_url not in seen:
                            valid_links.append(full_url)
                            seen.add(full_url)

            if not valid_links:
                return f"❌ Không tìm thấy văn bản nào cho '{query}' trên VBPL."

            # --- BƯỚC 2: ĐỌC CHI TIẾT VĂN BẢN ---
            # Lấy tối đa 2 kết quả đầu tiên
            target_links = valid_links[:2]
            final_result = f"Kết quả tìm kiếm VBPL (Playwright) cho '{query}':\n"

            for idx, link in enumerate(target_links, 1):
                log.info(f"Đang đọc chi tiết: {link}")
                try:
                    await page.goto(link, timeout=30000, wait_until="domcontentloaded")
                    
                    # Thử chờ nội dung chính xuất hiện (nếu web load chậm)
                    try:
                        await page.wait_for_selector("#toanvancontent", timeout=3000)
                    except:
                        pass # Nếu không có ID này thì cứ đọc tiếp

                    sub_html = await page.content()
                    sub_soup = BeautifulSoup(sub_html, "html.parser")
                    
                    # Các selector chứa nội dung luật
                    body = (sub_soup.select_one("#toanvancontent") or 
                            sub_soup.select_one("#divContentDoc") or 
                            sub_soup.select_one("div.content-detail"))
                    
                    if body:
                        # Xóa các phần rác (script, quảng cáo, thanh công cụ)
                        for tag in body(["script", "style", "div.minitoolbar", "div.ads"]):
                            tag.decompose()
                        
                        text = body.get_text(" ", strip=True)
                        text = " ".join(text.split()) # Làm sạch khoảng trắng thừa
                        final_result += f"\n=== Văn bản {idx}: {link} ===\n{text[:4000]}\n"
                    else:
                        final_result += f"\n=== Văn bản {idx}: {link} (Không đọc được nội dung) ===\n"

                except Exception as e:
                    final_result += f"\nLỗi khi đọc link {link}: {e}\n"
            
            return final_result

        except Exception as e:
            return f"❌ Lỗi Playwright trong quá trình xử lý: {e}"
        finally:
            # Luôn đóng trình duyệt để giải phóng RAM
            await browser.close()

if __name__ == "__main__":
    try:
        app.run()
    except Exception:
        pass