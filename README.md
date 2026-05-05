# Ollama WebUI（Gradio）

中文 | [English](./README_en.md)

## 專案說明
本專案提供一個以 Gradio 建立的 Ollama WebUI，可連線本地或遠端 Ollama 伺服器，支援一般聊天、影像提問、搜尋與工具整合。

## 主要功能
- 多語系 UI（預設含中文、英文、泰文，可透過 `language_settings.json` 擴充）
- 多主機管理（從 `server_settings.json` 讀取，UI 可新增主機）
- 模型參數設定（`Temperature`、`Max Tokens`、`Top P`、`Typical P`、`Num CTX`）
- Web Search（Serper.dev / Tavily）
- 搜尋設定：`Search Results Count`（1~20）與 `Search Summary Length`（short/medium/long）
- 內建工具：
  - `datetime`
  - `calculator`
  - `fetch_url`
  - `web_search`
- 支援手動工具命令：
  - `/tools`
  - `/tool <name> <json-args>`
- 主畫面提供「連線測試」按鈕與連線狀態燈（未測試/連線正常/連線失敗）
- 「停止回答」可中止串流與多數工具流程
- 「清除回答」會同步清空畫面與對話歷史
- 內建資料管理：
  - 對話 Session
  - Presets
  - Personas
  - Prompt Library
- 可匯出目前對話為 Markdown

## 架構（Orchestrator）
目前專案已由單一 `chat_service.py` 漸進拆分為分層架構：
- `app/main.py`
  - 應用入口，設定 logging 並啟動 Gradio
- `app/ui/gradio_app.py`
  - UI 組裝、事件綁定、語系字串與版面
- `app/services/chat_service.py`
  - 保留 Gradio 互動流程、串流輸出與匯出功能（adapter）
- `app/services/server_service.py`
  - Ollama 主機、模型清單與連線測試
- `app/services/session_service.py`
  - Session 持久化（`data/sessions.json`）
- `app/services/preset_service.py`
  - Preset 持久化（`data/presets.json`）
- `app/services/persona_service.py`
  - Persona 持久化（`data/personas.json`）
- `app/services/prompt_service.py`
  - Prompt Library 持久化（`data/prompts.json`）
- `app/orchestrator/auto_tool_planner.py`
  - deterministic 工具路由與 fallback 規劃
- `app/orchestrator/intent_router.py`
  - 工具意圖判斷（time/calc/fetch/search）
- `app/orchestrator/orchestrator.py`
  - orchestration 主流程與 typed output
- `app/orchestrator/tool_runtime.py`
  - 工具執行入口（含 policy + cancellation）
- `app/orchestrator/model_runtime.py`
  - 模型呼叫與摘要 prompt builder
- `app/orchestrator/conversation_pipeline.py`
  - legacy 對話流程抽離（tool decision/fallback/tool call/第二次模型回答）
- `app/orchestrator/types.py`
  - typed action 與 orchestrator 型別定義

此分層讓工具流程、模型流程、UI 流程可獨立演進，降低回歸風險並提升可測試性。

## 設定檔
- `server_settings.json`
  - `hosts`: Ollama 主機清單
  - `llm_parameters`: 預設模型參數（會與 UI 同步）
- `app_settings.json`
  - `search.provider`
  - `search.num_results`
  - `search.summary_length`
  - `search.tavily_api_key` / `search.serper_api_key`
  - `search.tavily_api_url` / `search.serper_api_url`
- `language_settings.json`
  - UI 語系與預設語言
- `data/*.json`
  - Session / Preset / Persona / Prompt Library 的持久化資料

## 環境需求
- Python 3.10+（建議）
- 相依套件請以 `requirements.txt` 為準

## 安裝與執行
1. 建立虛擬環境（可選）
2. 安裝套件：
   `pip install -r requirements.txt`
3. 啟動（相容入口）：
   `python ollama-webui.py`
4. 或使用模組入口：
   `python -m app.main`

## 常見操作
- 切換/新增 Ollama 主機：右側設定面板 > `Server`
- 調整搜尋與摘要長度：右側設定面板 > `Search`
- 調整 LLM 參數並寫回檔案：右側設定面板 > `Advanced` > `Save LLM Settings`
- 測試目前大模型連線：主畫面第二列 > `連線測試`（結果會顯示在狀態燈與狀態列）
- 管理 Session / Preset / Persona / Prompt：右側設定面板或左側 Session 區塊
- 匯出目前對話：主畫面工具列 > `Export Chat`

## 參考文件
- 英文說明：[README_en.md](./README_en.md)
- 跨電腦接手流程：[HANDOFF.md](./HANDOFF.md)

## 授權
本專案採用 [MIT License](LICENSE)。
