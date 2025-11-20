# law_search.py - 
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import urllib.parse
import logging
import time

# Cấu hình log
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("LawSearch")

def search_vbpl_sync(user_query):
    """
    Tìm kiếm trên Thư Viện Pháp Luật.
    Phiên bản V10: Thêm cơ chế "Vét cạn" nội dung nếu không tìm thấy thẻ div chính.
    """
    # 1. Làm sạch câu hỏi
    clean_query = user_query.replace("?", "").strip()
    
    # 2. TỪ KHÓA MỚI NHẤT 2025
    search_query = f"site:thuvienphapluat.vn {clean_query} mức phạt mới nhất 2025"
    
    encoded_query = urllib.parse.quote(search_query)
    ddg_url = f"https://duckduckgo.com/?q={encoded_query}&t=h_&ia=web"
    
    print(f"🚀 [TVPL Search] Đang tìm: '{search_query}'")

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=False, slow_mo=1000)
        except Exception as e:
            return f"Lỗi: Chưa cài trình duyệt. {e}"
            
        page = browser.new_page()
        
        try:
            # --- BƯỚC 1: VÀO DUCKDUCKGO ---
            print(f"🌍 Truy cập Search Engine...")
            page.goto(ddg_url, timeout=30000, wait_until="domcontentloaded")
            time.sleep(3) 

            content = page.content()
            soup = BeautifulSoup(content, "html.parser")
            
            found_link = None
            found_title = ""
            
            # --- BƯỚC 2: LỌC LINK ---
            print("🔎 Đang lọc link kết quả...")
            all_links = soup.find_all("a", href=True)
            
            for a in all_links:
                href = a.get("href")
                title = a.get_text().strip()
                
                if not href.startswith("http"): continue

                if "thuvienphapluat.vn" in href:
                    if any(x in href for x in ["google", "search", "dang-nhap"]): continue
                    
                    print(f"   Link ứng viên: {title[:50]}... -> {href}")
                    
                    found_link = href
                    found_title = title
                    break 

            if not found_link:
                return "Không tìm thấy bài viết phù hợp trên Thư Viện Pháp Luật."

            # --- BƯỚC 3: ĐỌC BÀI VIẾT (CẢI TIẾN) ---
            print(f"🎯 Đọc bài: {found_link}")
            page.goto(found_link, timeout=30000, wait_until="domcontentloaded")
            
            # Chờ nội dung (thử nhiều selector hơn)
            try:
                page.wait_for_selector("div.content-0, div.news-content, div.content, article", timeout=5000)
            except: pass

            sub_soup = BeautifulSoup(page.content(), "html.parser")
            
            # Xóa các phần rác trước khi lọc nội dung để tránh lấy nhầm
            for tag in sub_soup(["script", "style", "div.relate-news", "div.comment", "div.adv", "div.bottom-mobile", "footer", "header"]):
                tag.decompose()

            # Thử danh sách selector mở rộng (quét hết các kiểu layout của TVPL)
            selectors = [
                "div.content-0", 
                "div.news-content", 
                "div.content", 
                "div#news-content",
                "article",
                "div.post-content"
            ]
            
            body = None
            for sel in selectors:
                body = sub_soup.select_one(sel)
                if body: break
            
            text = ""
            if body:
                text = " ".join(body.get_text(" ", strip=True).split())
            else:
                # FALLBACK: Nếu không khớp selector nào, lấy toàn bộ text trong body
                print("⚠️ Không tìm thấy div chính, dùng chế độ đọc thô (Raw Text)...")
                body_tag = sub_soup.find("body")
                if body_tag:
                    text = " ".join(body_tag.get_text(" ", strip=True).split())
                else:
                    return f"Lỗi: Trang web trống rỗng ({found_link})"

            # Trả về kết quả (Cắt bớt nếu quá dài để tránh lỗi API Chat)
            return f"Nguồn: Thư Viện Pháp Luật\nLink: {found_link}\n\nNỘI DUNG CHI TIẾT:\n{text[:12000]}..."

        except Exception as e:
            # browser.close() # Đóng ở finally rồi
            return f"Lỗi quá trình tìm kiếm: {e}"
            
        finally:
             browser.close()

if __name__ == "__main__":
    print(search_vbpl_sync("vượt đèn đỏ phạt bao nhiêu"))