#!/usr/bin/env python3
"""
BibTeX 파일에서 논문 제목을 읽고 자동으로 DOI와 URL을 찾아오는 스크립트
GitHub Actions에서 실행됨
"""

import re
import requests
from pathlib import Path
from typing import Optional
import time

class PublicationLinkFetcher:
    def __init__(self):
        self.crossref_url = "https://api.crossref.org/works"
        self.arxiv_url = "http://export.arxiv.org/api/query"
        
    def fetch_doi(self, title: str, author: Optional[str] = None) -> Optional[str]:
        """CrossRef API를 사용해서 DOI 찾기"""
        try:
            params = {
                "query.bibliographic": title,
                "rows": 5
            }
            if author:
                params["query.author"] = author.split(" and ")[0].strip()
            
            response = requests.get(
                self.crossref_url,
                params=params,
                headers={"User-Agent": "Publication-Link-Fetcher"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("message", {}).get("items"):
                    item = data["message"]["items"][0]
                    crossref_title = item.get("title", [""])[0].lower()
                    if self._similarity(title.lower(), crossref_title) > 0.8:
                        doi = item.get("DOI")
                        if doi:
                            return doi
        except Exception as e:
            print(f"⚠️ CrossRef API 오류: {e}")
        
        return None

    def fetch_arxiv_url(self, title: str, author: Optional[str] = None) -> Optional[str]:
        """arXiv API를 사용해서 논문 링크 찾기"""
        try:
            query = f'title:"{title}"'
            if author:
                first_author = author.split(" and ")[0].strip()
                query += f' AND author:"{first_author}"'
            
            params = {
                "search_query": query,
                "start": 0,
                "max_results": 3,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            response = requests.get(
                self.arxiv_url,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(response.content)
                
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                entries = root.findall("atom:entry", ns)
                
                if entries:
                    entry = entries[0]
                    entry_title = entry.find("atom:title", ns)
                    if entry_title is not None:
                        arxiv_title = entry_title.text.strip()
                        if self._similarity(title.lower(), arxiv_title.lower()) > 0.75:
                            arxiv_id = entry.find("atom:id", ns)
                            if arxiv_id is not None:
                                arxiv_url = arxiv_id.text.replace("http://", "https://")
                                return arxiv_url
        except Exception as e:
            print(f"⚠️ arXiv API 오류: {e}")
        
        return None

    def _similarity(self, s1: str, s2: str) -> float:
        """두 문자열의 유사도 계산"""
        s1_clean = re.sub(r'[^\w]', '', s1)
        s2_clean = re.sub(r'[^\w]', '', s2)
        
        min_len = min(len(s1_clean), len(s2_clean))
        if min_len == 0:
            return 0.0
        
        matches = sum(1 for a, b in zip(s1_clean, s2_clean) if a == b)
        return matches / min_len

    def process_bibtex(self, bib_file_path: Path) -> bool:
        """BibTeX 파일을 읽고 링크를 추가. 변경사항이 있으면 True 반환"""
        with open(bib_file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        new_content = original_content
        changed = False
        
        # BibTeX 항목 파싱
        pattern = r'@(\w+)\s*\{\s*([^,\n]+),\s*([\s\S]*?)(?=@|\Z)'
        
        for match in re.finditer(pattern, original_content):
            entry_type, cite_key, fields_str = match.groups()
            
            # title과 author 추출
            title_match = re.search(r'title\s*=\s*\{([^}]+)\}', fields_str, re.IGNORECASE)
            author_match = re.search(r'author\s*=\s*\{([^}]+)\}', fields_str, re.IGNORECASE)
            doi_match = re.search(r'doi\s*=\s*\{([^}]+)\}', fields_str, re.IGNORECASE)
            url_match = re.search(r'url\s*=\s*\{([^}]+)\}', fields_str, re.IGNORECASE)
            
            if not title_match:
                continue
            
            title = title_match.group(1)
            author = author_match.group(1) if author_match else None
            
            # 이미 DOI나 URL이 있으면 스킵
            if doi_match or url_match:
                print(f"✓ {cite_key}: 이미 링크가 있습니다")
                continue
            
            print(f"🔍 {cite_key} 검색 중...")
            
            # 1. DOI 찾기
            doi = self.fetch_doi(title, author)
            if doi:
                print(f"   ✓ DOI 찾음: {doi}")
                new_fields = fields_str.rstrip()
                if not new_fields.endswith(','):
                    new_fields += ','
                new_fields += f'\n  doi = {{{doi}}}'
                
                old_entry = match.group(0)
                new_entry = old_entry.replace(fields_str, new_fields)
                new_content = new_content.replace(old_entry, new_entry)
                changed = True
                time.sleep(1)
                continue
            
            # 2. arXiv 찾기
            arxiv_url = self.fetch_arxiv_url(title, author)
            if arxiv_url:
                print(f"   ✓ arXiv 찾음: {arxiv_url}")
                new_fields = fields_str.rstrip()
                if not new_fields.endswith(','):
                    new_fields += ','
                new_fields += f'\n  url = {{{arxiv_url}}}'
                
                old_entry = match.group(0)
                new_entry = old_entry.replace(fields_str, new_fields)
                new_content = new_content.replace(old_entry, new_entry)
                changed = True
                time.sleep(1)
                continue
            
            print(f"   ✗ 링크를 찾을 수 없습니다")
            time.sleep(1)
        
        # 변경사항이 있으면 파일에 저장
        if changed:
            with open(bib_file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("\n✅ 완료! BibTeX 파일이 업데이트되었습니다")
        else:
            print("\nℹ️ 새로운 링크를 추가할 항목이 없습니다")
        
        return changed

def main():
    bib_file = Path("_data/publications.bib")
    
    if not bib_file.exists():
        print(f"❌ 파일을 찾을 수 없습니다: {bib_file}")
        return
    
    print("📚 BibTeX 논문 링크 자동 추가 (GitHub Actions)")
    print("=" * 50)
    print(f"📄 파일: {bib_file}\n")
    
    fetcher = PublicationLinkFetcher()
    fetcher.process_bibtex(bib_file)

if __name__ == "__main__":
    main()
