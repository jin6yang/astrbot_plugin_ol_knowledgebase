import aiohttp
import asyncio
import logging
from typing import List, Optional

logger = logging.getLogger("astrbot")

class BaseKnowledgeBaseAdapter:
    DEFAULT_ENDPOINT: Optional[str] = None

    def __init__(self, api_endpoint: str, api_key: str, dataset_id: str, top_k: int, score_threshold: float):
        self.api_endpoint = api_endpoint.rstrip('/')
        self.api_key = api_key
        self.dataset_id = dataset_id
        self.top_k = top_k
        self.score_threshold = score_threshold

    async def retrieve(self, query: str) -> List[str]:
        raise NotImplementedError

class DifyAdapter(BaseKnowledgeBaseAdapter):
    DEFAULT_ENDPOINT = "https://api.dify.ai/v1"

    async def retrieve(self, query: str) -> List[str]:
        if not self.api_key or not self.dataset_id:
            logger.debug("Dify API Key 或 Dataset ID 未配置，跳过知识库检索")
            return []

        url = f"{self.api_endpoint}/datasets/{self.dataset_id}/retrieve"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": query,
            "retrieval_model": {
                "search_method": "semantic_search",
                "reranking_enable": False,
                "top_k": self.top_k,
                "score_threshold": self.score_threshold,
                "score_threshold_enabled": True
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=15) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"HTTP {resp.status} - {error_text}")
                data = await resp.json()
                
        records = data.get("records", [])
        contexts = []
        for record in records:
            segment = record.get("segment", {})
            content = segment.get("content", "").strip()
            if content:
                contexts.append(content)
                
        return contexts

class RAGFlowAdapter(BaseKnowledgeBaseAdapter):
    DEFAULT_ENDPOINT = None

    async def retrieve(self, query: str) -> List[str]:
        if not self.api_key or not self.dataset_id:
            logger.debug("RAGFlow API Key 或 Dataset ID 未配置，跳过知识库检索")
            return []

        url = f"{self.api_endpoint}/api/v1/retrieval"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "question": query,
            "dataset_ids": [d.strip() for d in self.dataset_id.split(",") if d.strip()],
            "top_k": self.top_k,
            "similarity_threshold": self.score_threshold
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=15) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"HTTP {resp.status} - {error_text}")
                data = await resp.json()
                
        chunks = data.get("data", {}).get("chunks", [])
        contexts = []
        for chunk in chunks:
            content = chunk.get("content", "").strip()
            if content:
                contexts.append(content)
                
        return contexts

class FlowiseAdapter(BaseKnowledgeBaseAdapter):
    DEFAULT_ENDPOINT = "https://cloud.flowiseai.com/api/v1"

    async def retrieve(self, query: str) -> List[str]:
        if not self.dataset_id:
            logger.debug("Flowise Store ID (Dataset ID) 未配置，跳过知识库检索")
            return []

        url = f"{self.api_endpoint}/document-store/vectorstore/query"
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        payload = {
            "storeId": self.dataset_id,
            "query": query
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=15) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    raise Exception(f"HTTP {resp.status} - {error_text}")
                data = await resp.json()
                
        docs = data.get("docs", [])
        contexts = []
        for doc in docs[:self.top_k]:  # Flowise query often doesn't strictly adhere to top_k locally inside payload based on some versions
            content = doc.get("pageContent", "").strip()
            if content:
                contexts.append(content)
                
        return contexts

class _RateLimitError(Exception):
    pass

class NotionAdapter(BaseKnowledgeBaseAdapter):
    DEFAULT_ENDPOINT = "https://api.notion.com/v1"
    NOTION_VERSION = "2022-06-28"
    MAX_DEPTH = 2
    MAX_PAGE_REQUESTS = 15

    def __init__(self, api_endpoint: str, api_key: str, top_k: int,
                 max_chars_per_page: int = 2000, max_blocks_per_page: int = 50):
        self.api_endpoint = api_endpoint.rstrip('/')
        self.api_key = api_key
        self.top_k = top_k
        self.max_chars_per_page = max(1, int(max_chars_per_page))
        self.max_blocks_per_page = max(1, int(max_blocks_per_page))

    async def retrieve(self, query: str) -> List[str]:
        if not self.api_key:
            logger.debug("Notion Integration Token 未配置，跳过知识库检索")
            return []

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": self.NOTION_VERSION,
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession(headers=headers) as session:
            pages = await self._search_pages(session, query)
            if not pages:
                return []
            sem = asyncio.Semaphore(3)
            results = await asyncio.gather(
                *(self._fetch_page_text(session, sem, page) for page in pages),
                return_exceptions=True
            )

        contexts = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"[Notion] 拉取页面内容失败: {result}")
            elif result:
                contexts.append(result)
        return contexts

    async def _search_pages(self, session: aiohttp.ClientSession, query: str) -> List[dict]:
        url = f"{self.api_endpoint}/search"
        payload = {
            "query": query,
            "filter": {"property": "object", "value": "page"},
            "page_size": max(1, min(int(self.top_k), 100))
        }
        async with session.post(url, json=payload, timeout=15) as resp:
            if resp.status in (401, 403):
                error_text = await resp.text()
                raise Exception(f"Notion 鉴权失败 (HTTP {resp.status}): Integration Token 无效，或知识库页面尚未通过 Connections 分享给该 Integration - {error_text}")
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"HTTP {resp.status} - {error_text}")
            data = await resp.json()

        pages = []
        for item in data.get("results", []):
            if item.get("object") != "page":
                continue
            if item.get("archived") or item.get("in_trash"):
                continue
            pages.append(item)
        return pages

    async def _fetch_page_text(self, session: aiohttp.ClientSession, sem: asyncio.Semaphore, page: dict) -> str:
        async with sem:
            title = self._extract_page_title(page) or "未命名页面"
            blocks, truncated_reason = await self._collect_blocks(session, page["id"])

            if truncated_reason:
                logger.warning(f"[Notion] 页面《{title}》内容过多，已达{truncated_reason}，仅注入已获取的部分内容")

            lines = []
            for block in blocks:
                text = self._extract_plain_text(block)
                if text:
                    lines.append(text)
            body = "\n".join(lines)

            if len(body) > self.max_chars_per_page:
                logger.warning(f"[Notion] 页面《{title}》正文共 {len(body)} 字符，超过单页上限 {self.max_chars_per_page}，已自动截断")
                body = body[:self.max_chars_per_page]

            if not body.strip():
                return ""
            return f"# {title}\n{body}"

    async def _collect_blocks(self, session: aiohttp.ClientSession, page_id: str):
        blocks = []
        budget = self.MAX_PAGE_REQUESTS
        truncated_reason = None

        async def fetch_level(parent_id: str, depth: int):
            nonlocal budget, truncated_reason
            cursor = None
            while True:
                if budget <= 0:
                    truncated_reason = f"单页请求数预算({self.MAX_PAGE_REQUESTS})"
                    return
                if len(blocks) >= self.max_blocks_per_page:
                    truncated_reason = f"块数上限({self.max_blocks_per_page})"
                    return
                budget -= 1
                params = {"page_size": 100}
                if cursor:
                    params["start_cursor"] = cursor
                try:
                    data = await self._get_json(session, f"{self.api_endpoint}/blocks/{parent_id}/children", params)
                except _RateLimitError:
                    logger.warning("[Notion] 触发 API 限流 (HTTP 429)，仅注入已获取的部分内容")
                    truncated_reason = truncated_reason or "API 限流"
                    return
                level_blocks = data.get("results", [])
                blocks.extend(level_blocks)
                if len(blocks) >= self.max_blocks_per_page:
                    del blocks[self.max_blocks_per_page:]
                    truncated_reason = f"块数上限({self.max_blocks_per_page})"
                    return
                if not data.get("has_more") or not data.get("next_cursor"):
                    break
                cursor = data.get("next_cursor")

            if depth < self.MAX_DEPTH:
                for block in level_blocks:
                    if not block.get("has_children"):
                        continue
                    if budget <= 0:
                        truncated_reason = f"单页请求数预算({self.MAX_PAGE_REQUESTS})"
                        return
                    if len(blocks) >= self.max_blocks_per_page:
                        truncated_reason = f"块数上限({self.max_blocks_per_page})"
                        return
                    await fetch_level(block["id"], depth + 1)
            elif any(b.get("has_children") for b in level_blocks):
                logger.debug(f"[Notion] 存在超过 {self.MAX_DEPTH} 层的嵌套块，更深层内容已忽略")

        await fetch_level(page_id, 1)
        return blocks, truncated_reason

    async def _get_json(self, session: aiohttp.ClientSession, url: str, params: dict) -> dict:
        async with session.get(url, params=params, timeout=15) as resp:
            if resp.status == 429:
                raise _RateLimitError()
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"HTTP {resp.status} - {error_text}")
            return await resp.json()

    @staticmethod
    def _extract_plain_text(block: dict) -> str:
        block_type = block.get("type", "")
        content = block.get(block_type)
        if not isinstance(content, dict):
            return ""
        rich_text = content.get("rich_text")
        if not isinstance(rich_text, list):
            return ""
        return "".join(t.get("plain_text", "") for t in rich_text).strip()

    @staticmethod
    def _extract_page_title(page: dict) -> str:
        for prop in page.get("properties", {}).values():
            if isinstance(prop, dict) and prop.get("type") == "title":
                return "".join(t.get("plain_text", "") for t in prop.get("title", [])).strip()
        return ""

def get_adapter(backend_type: str, api_endpoint: str, api_key: str, dataset_id: str,
                top_k: int, score_threshold: float,
                use_custom_api_endpoint: bool = False, **kwargs) -> BaseKnowledgeBaseAdapter:
    backend_type = (backend_type or "dify").lower()
    api_endpoint = (api_endpoint or "").strip()

    adapter_cls = {
        "dify": DifyAdapter,
        "ragflow": RAGFlowAdapter,
        "flowise": FlowiseAdapter,
        "notion": NotionAdapter
    }.get(backend_type, DifyAdapter)

    if use_custom_api_endpoint:
        if not api_endpoint:
            raise ValueError("已开启「使用自定义 API 端点」但未填写 API Base URL，请在插件配置中填写，或关闭该开关使用内置官方地址")
        resolved_endpoint = api_endpoint
    else:
        resolved_endpoint = adapter_cls.DEFAULT_ENDPOINT
        if not resolved_endpoint:
            raise ValueError(f"{backend_type} 后端无官方云服务地址，请开启「使用自定义 API 端点」并填写你的服务访问地址")

    if backend_type == "notion":
        return NotionAdapter(
            resolved_endpoint, api_key, top_k,
            max_chars_per_page=kwargs.get("notion_max_chars_per_page", 2000),
            max_blocks_per_page=kwargs.get("notion_max_blocks_per_page", 50)
        )
    return adapter_cls(resolved_endpoint, api_key, dataset_id, top_k, score_threshold)
