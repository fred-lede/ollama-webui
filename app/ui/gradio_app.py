from __future__ import annotations
from urllib.parse import urlparse

import gradio as gr

from app.core.app_settings import load_app_settings, save_app_settings
from app.core.config import load_language_settings, load_llm_parameters, load_settings, save_default_language, save_llm_parameters
from app.services.chat_service import (
    apply_preset_to_current_session,
    ask_question_stream,
    create_new_chat_session_with_state,
    delete_selected_persona,
    delete_selected_preset,
    delete_selected_prompt,
    delete_chat_session_with_state,
    export_current_chat_markdown,
    get_current_session_preferences,
    list_chat_session_choices,
    list_persona_choices,
    list_preset_choices,
    list_prompt_choices,
    load_selected_prompt,
    load_current_chat_history,
    is_session_pinned,
    load_selected_persona,
    rename_chat_session,
    save_persona,
    save_preset_from_values,
    save_prompt_entry,
    set_session_label_language,
    clear_current_chat_with_state,
    set_session_pinned,
    stop_response,
    switch_chat_session_with_state,
    insert_selected_prompt_into_workspace,
    update_current_session_preferences,
)
from app.services.server_service import fetch_models, handle_add_server, handle_server_change, test_llm_connection

css = """
:root {
    --session-panel-bg: transparent;
    --session-card-bg: #f5f5f4;
    --session-card-hover: #e7e5e4;
    --session-card-active: #e7e5e4;
    --session-border: #d6d3d1;
    --session-border-strong: #57534e;
    --session-text: #1c1917;
    --session-accent: #f59e0b;
}
@media (prefers-color-scheme: dark) {
    :root {
        --session-panel-bg: transparent;
        --session-card-bg: #2b2624;
        --session-card-hover: #332e2b;
        --session-card-active: #332e2b;
        --session-border: #4a4441;
        --session-border-strong: #d6d3d1;
        --session-text: #f5f5f4;
        --session-accent: #f59e0b;
    }
}
.my-button {
    min-width: 110px !important;
    width: auto !important;
}
.toolbar-button {
    min-width: 120px !important;
    width: auto !important;
    min-height: 40px !important;
    height: 40px !important;
    padding-left: 14px !important;
    padding-right: 14px !important;
    padding-top: 8px !important;
    padding-bottom: 8px !important;
}
.status-light {
    min-width: 130px;
}
.toolbar-row {
    gap: 8px;
    flex-wrap: wrap;
}
.drawer-note {
    opacity: 0.85;
    font-size: 0.9rem;
}
.settings-section-label {
    margin: 12px 0 6px !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase;
    opacity: 0.72;
}
.session-stack {
    max-height: 396px;
    overflow-y: auto;
    padding-right: 4px;
}
.session-item {
    justify-content: flex-start !important;
    width: 100% !important;
    min-height: 36px !important;
    height: 36px !important;
    margin-bottom: 4px !important;
    padding-top: 6px !important;
    padding-bottom: 6px !important;
    border-radius: 10px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    flex-shrink: 0 !important;
    position: relative !important;
    outline: none !important;
    box-shadow: none !important;
}
.session-item:last-child {
    margin-bottom: 0 !important;
}
.session-item:focus,
.session-item button:focus {
    outline: none !important;
    box-shadow: none !important;
}
.session-item:focus-visible,
.session-item button:focus-visible {
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.35) !important;
    border-color: var(--session-accent) !important;
}
.session-item.primary {
    background: var(--session-card-active) !important;
    color: var(--session-text) !important;
    border-color: var(--session-border-strong) !important;
    box-shadow: inset 0 0 0 1px var(--session-border-strong) !important;
}
.session-item.primary:focus-visible,
.session-item.primary button:focus-visible {
    box-shadow:
        inset 0 0 0 1px var(--session-border-strong) !important,
        0 0 0 2px rgba(245, 158, 11, 0.35) !important;
}
.session-item.primary::before {
    content: "";
    position: absolute;
    left: 0;
    top: 6px;
    bottom: 6px;
    width: 4px;
    border-radius: 999px;
    background: var(--session-accent);
}
@media (max-width: 1100px) {
    .app-shell {
        flex-wrap: wrap !important;
    }
    .sessions-panel,
    .chat-panel,
    .settings-panel {
        min-width: 100% !important;
        width: 100% !important;
    }
    .session-stack {
        max-height: 220px;
    }
}
@media (max-width: 720px) {
    .toolbar-button {
        min-width: calc(50% - 6px) !important;
        width: calc(50% - 6px) !important;
    }
    .status-light {
        min-width: 100%;
    }
    .session-item {
        min-height: 40px !important;
        height: 40px !important;
    }
}
.quick-control-row {
    gap: 8px;
    margin-bottom: 8px;
}
"""

UI_DEFAULT_TRANSLATIONS = {
    "English": {
        "title": "# Ollama WebUI",
        "select_server": "Select AI Server",
        "server_address": "Server Address",
        "select_model": "Select AI Model",
        "status_display": "Status Display",
        "status_display_msg": "Displaying the model status or an error message.",
        "enter_question": "Enter your question or upload an image with your question...",
        "ai_server_name": "New AI Server Name",
        "ai_server_name_msg": "Ex: GMK-K9, editable in server_settings.json.",
        "new_server_address": "New Server Address",
        "new_server_address_msg": "Ex. http://127.0.0.1",
        "new_server_port": "Port",
        "default_start_setting": "Make the new host the default.",
        "add_ai_server": "Add AI Server",
        "stop_button": "Stop Answer",
        "clean_answer_button": "Clear Answer",
        "sessions_heading": "### Sessions",
        "session_search": "Search Sessions",
        "session_search_placeholder": "Filter by title or time",
        "pin_session": "Pin Session",
        "unpin_session": "Unpin Session",
        "session_title": "Session Title",
        "rename_current_chat": "Rename current chat",
        "new_chat_button": "New Chat",
        "rename_button": "Rename",
        "delete_button": "Delete",
        "web_search_on": "Web Search: ON",
        "web_search_off": "Web Search: OFF",
        "settings_open": "Settings",
        "settings_close": "Close Settings",
        "settings_heading": "### Settings",
        "settings_note": "<div class='drawer-note'>All settings live here and apply immediately after saving.</div>",
        "settings_common": "Common",
        "settings_workspace": "Workspace",
        "settings_system": "System",
        "search_accordion": "Search",
        "search_provider": "Search Provider",
        "search_results_count": "Search Results Count",
        "search_results_info": "How many web search results to use for answer grounding",
        "search_summary_length": "Search Summary Length",
        "search_summary_info": "Controls how detailed search summaries should be",
        "search_summary_short": "Short",
        "search_summary_medium": "Medium",
        "search_summary_long": "Long",
        "tavily_api_key": "Tavily API Key",
        "tavily_api_key_placeholder": "Paste Tavily API key",
        "serper_api_key": "Serper.dev API Key",
        "serper_api_key_placeholder": "Paste Serper.dev API key",
        "save_search_settings": "Save Search Settings",
        "server_accordion": "Server",
        "presets_accordion": "Presets",
        "preset_label": "Preset",
        "preset_name": "Preset Name",
        "preset_placeholder": "Create or update a custom preset",
        "save_preset": "Save Preset",
        "delete_preset": "Delete Preset",
        "personas_accordion": "Personas",
        "persona_label": "Persona",
        "persona_name": "Persona Name",
        "persona_placeholder": "Example: Code Reviewer / Travel Planner / PM Assistant",
        "persona_description": "Description",
        "persona_description_placeholder": "Example: concise reviewer focused on bugs, risks, and regressions",
        "system_prompt": "System Prompt",
        "system_prompt_placeholder": "Example: You are a senior code reviewer. Prioritize bugs, risks, and missing tests.",
        "default_model": "Default Model",
        "default_preset": "Default Preset",
        "save_persona": "Save Persona",
        "delete_persona": "Delete Persona",
        "prompt_library_accordion": "Prompt Library",
        "prompt_label": "Prompt",
        "prompt_name": "Prompt Name",
        "prompt_placeholder": "Create or update a prompt",
        "category": "Category",
        "category_placeholder": "general / writing / coding ...",
        "prompt_content": "Prompt Content",
        "prompt_content_placeholder": "Write the reusable prompt here",
        "favorite": "Favorite",
        "save_prompt": "Save Prompt",
        "delete_prompt": "Delete Prompt",
        "insert_prompt": "Insert to Input",
        "advanced_accordion": "Advanced",
        "language_label": "Language",
        "temperature": "Temperature",
        "temperature_info": "Choose between 0 and 1",
        "max_tokens": "Max Tokens",
        "top_p": "Top P",
        "typical_p": "Typical P",
        "num_ctx": "Num CTX",
        "save_llm_settings": "Save LLM Settings",
        "tool_commands": (
            "**Tool Commands**  \n"
            "`/tools`  \n"
            "`/tool calculator {\"expression\":\"2+3*4\"}`  \n"
            "`/tool datetime {}`  \n"
            "`/tool web_search {\"query\":\"latest ollama news\",\"num_results\":3}`  \n"
            "`/tool fetch_url {\"url\":\"https://example.com\"}`"
        ),
        "test_connection": "Test Connection",
        "export_chat": "Export Chat",
        "connection_untested": "Not tested",
        "web_search_setting_saved": "Web search setting saved.",
        "web_search_setting_failed": "Failed to save web search setting.",
        "settings_saved": "Search settings saved.",
        "settings_save_failed": "Failed to save search settings.",
        "llm_settings_saved": "LLM settings saved to server_settings.json.",
        "llm_settings_save_failed": "Failed to save LLM settings.",
    },
    "Chinese": {
        "title": "# Ollama WebUI",
        "select_server": "選擇 AI 伺服器",
        "server_address": "伺服器位址",
        "select_model": "選擇 AI 模型",
        "status_display": "狀態顯示",
        "status_display_msg": "顯示模型狀態或錯誤訊息。",
        "enter_question": "輸入問題，或上傳圖片後一起提問……",
        "ai_server_name": "新的 AI 伺服器名稱",
        "ai_server_name_msg": "例如：GMK-K9，可在 server_settings.json 中手動編輯。",
        "new_server_address": "新的伺服器位址",
        "new_server_address_msg": "例如：http://127.0.0.1",
        "new_server_port": "連接埠",
        "default_start_setting": "將新主機設為預設值",
        "add_ai_server": "新增 AI 伺服器",
        "stop_button": "停止回答",
        "clean_answer_button": "清除回答",
        "sessions_heading": "### 對話列表",
        "session_search": "搜尋對話",
        "session_search_placeholder": "依標題或時間篩選",
        "pin_session": "置頂對話",
        "unpin_session": "取消置頂",
        "session_title": "對話標題",
        "rename_current_chat": "重新命名目前對話",
        "new_chat_button": "新增對話",
        "rename_button": "重新命名",
        "delete_button": "刪除",
        "web_search_on": "網頁搜尋：開",
        "web_search_off": "網頁搜尋：關",
        "settings_open": "設定",
        "settings_close": "關閉設定",
        "settings_heading": "### 設定面板",
        "settings_note": "<div class='drawer-note'>所有設定集中在這裡，儲存後即生效。</div>",
        "settings_common": "常用",
        "settings_workspace": "工作區",
        "settings_system": "系統",
        "search_accordion": "搜尋",
        "search_provider": "搜尋服務",
        "search_results_count": "搜尋結果數量",
        "search_results_info": "回答時用多少筆搜尋結果作為參考",
        "search_summary_length": "搜尋摘要長度",
        "search_summary_info": "控制搜尋摘要的詳細程度",
        "search_summary_short": "短",
        "search_summary_medium": "中",
        "search_summary_long": "長",
        "tavily_api_key": "Tavily API 金鑰",
        "tavily_api_key_placeholder": "貼上 Tavily API 金鑰",
        "serper_api_key": "Serper.dev API 金鑰",
        "serper_api_key_placeholder": "貼上 Serper.dev API 金鑰",
        "save_search_settings": "儲存搜尋設定",
        "server_accordion": "伺服器",
        "presets_accordion": "預設組合",
        "preset_label": "預設組合",
        "preset_name": "預設名稱",
        "preset_placeholder": "建立或更新自訂預設",
        "save_preset": "儲存預設",
        "delete_preset": "刪除預設",
        "personas_accordion": "角色",
        "persona_label": "角色",
        "persona_name": "角色名稱",
        "persona_placeholder": "例如：程式碼審查助手 / 旅遊規劃助手 / PM 助手",
        "persona_description": "描述",
        "persona_description_placeholder": "例如：回答精簡，優先指出 bug、風險與回歸問題",
        "system_prompt": "系統提示詞",
        "system_prompt_placeholder": "例如：你是一位資深程式碼審查員，優先指出 bug、風險與缺少的測試。",
        "default_model": "預設模型",
        "default_preset": "預設組合",
        "save_persona": "儲存角色",
        "delete_persona": "刪除角色",
        "prompt_library_accordion": "提示詞庫",
        "prompt_label": "提示詞",
        "prompt_name": "提示詞名稱",
        "prompt_placeholder": "建立或更新提示詞",
        "category": "分類",
        "category_placeholder": "general / writing / coding ...",
        "prompt_content": "提示詞內容",
        "prompt_content_placeholder": "在這裡輸入可重複使用的提示詞",
        "favorite": "收藏",
        "save_prompt": "儲存提示詞",
        "delete_prompt": "刪除提示詞",
        "insert_prompt": "插入輸入框",
        "advanced_accordion": "進階",
        "language_label": "語言",
        "temperature": "溫度",
        "temperature_info": "請選擇 0 到 1 之間的數值",
        "max_tokens": "最大 Token 數",
        "top_p": "Top P",
        "typical_p": "Typical P",
        "num_ctx": "上下文長度",
        "save_llm_settings": "儲存 LLM 設定",
        "tool_commands": (
            "**工具指令**  \n"
            "`/tools`  \n"
            "`/tool calculator {\"expression\":\"2+3*4\"}`  \n"
            "`/tool datetime {}`  \n"
            "`/tool web_search {\"query\":\"latest ollama news\",\"num_results\":3}`  \n"
            "`/tool fetch_url {\"url\":\"https://example.com\"}`"
        ),
        "test_connection": "連線測試",
        "export_chat": "匯出對話",
        "connection_untested": "未測試",
        "web_search_setting_saved": "已儲存網頁搜尋設定。",
        "web_search_setting_failed": "儲存網頁搜尋設定失敗。",
        "settings_saved": "已儲存搜尋設定。",
        "settings_save_failed": "儲存搜尋設定失敗。",
        "llm_settings_saved": "已將 LLM 設定儲存到 server_settings.json。",
        "llm_settings_save_failed": "儲存 LLM 設定失敗。",
    },
    "Thailand": {
        "title": "# Ollama WebUI",
        "select_server": "เลือกเซิร์ฟเวอร์ AI",
        "server_address": "ที่อยู่เซิร์ฟเวอร์",
        "select_model": "เลือกรุ่น AI",
        "status_display": "การแสดงสถานะ",
        "status_display_msg": "แสดงสถานะของโมเดลหรือข้อความผิดพลาด",
        "enter_question": "พิมพ์คำถามหรืออัปโหลดรูปภาพพร้อมคำถามของคุณ...",
        "ai_server_name": "ชื่อเซิร์ฟเวอร์ AI ใหม่",
        "ai_server_name_msg": "เช่น GMK-K9 สามารถแก้ไขใน server_settings.json ได้",
        "new_server_address": "ที่อยู่เซิร์ฟเวอร์ใหม่",
        "new_server_address_msg": "เช่น http://127.0.0.1",
        "new_server_port": "พอร์ต",
        "default_start_setting": "ตั้งโฮสต์ใหม่นี้เป็นค่าเริ่มต้น",
        "add_ai_server": "เพิ่มเซิร์ฟเวอร์ AI",
        "stop_button": "หยุดคำตอบ",
        "clean_answer_button": "ล้างคำตอบ",
        "sessions_heading": "### รายการแชต",
        "session_search": "ค้นหาแชต",
        "session_search_placeholder": "กรองตามชื่อหรือเวลา",
        "pin_session": "ปักหมุดแชต",
        "unpin_session": "เลิกปักหมุด",
        "session_title": "ชื่อแชต",
        "rename_current_chat": "เปลี่ยนชื่อแชตปัจจุบัน",
        "new_chat_button": "แชตใหม่",
        "rename_button": "เปลี่ยนชื่อ",
        "delete_button": "ลบ",
        "web_search_on": "ค้นหาเว็บ: เปิด",
        "web_search_off": "ค้นหาเว็บ: ปิด",
        "settings_open": "การตั้งค่า",
        "settings_close": "ปิดการตั้งค่า",
        "settings_heading": "### แผงการตั้งค่า",
        "settings_note": "<div class='drawer-note'>การตั้งค่าทั้งหมดอยู่ที่นี่และจะมีผลทันทีหลังบันทึก</div>",
        "settings_common": "ทั่วไป",
        "settings_workspace": "พื้นที่ทำงาน",
        "settings_system": "ระบบ",
        "search_accordion": "ค้นหา",
        "search_provider": "ผู้ให้บริการค้นหา",
        "search_results_count": "จำนวนผลการค้นหา",
        "search_results_info": "จำนวนผลค้นหาที่ใช้เป็นข้อมูลอ้างอิงในการตอบ",
        "search_summary_length": "ความยาวสรุปการค้นหา",
        "search_summary_info": "กำหนดระดับความละเอียดของสรุปผลการค้นหา",
        "search_summary_short": "สั้น",
        "search_summary_medium": "กลาง",
        "search_summary_long": "ยาว",
        "tavily_api_key": "คีย์ API ของ Tavily",
        "tavily_api_key_placeholder": "วางคีย์ API ของ Tavily",
        "serper_api_key": "คีย์ API ของ Serper.dev",
        "serper_api_key_placeholder": "วางคีย์ API ของ Serper.dev",
        "save_search_settings": "บันทึกการตั้งค่าการค้นหา",
        "server_accordion": "เซิร์ฟเวอร์",
        "presets_accordion": "พรีเซ็ต",
        "preset_label": "พรีเซ็ต",
        "preset_name": "ชื่อพรีเซ็ต",
        "preset_placeholder": "สร้างหรืออัปเดตพรีเซ็ตแบบกำหนดเอง",
        "save_preset": "บันทึกพรีเซ็ต",
        "delete_preset": "ลบพรีเซ็ต",
        "personas_accordion": "บทบาท",
        "persona_label": "บทบาท",
        "persona_name": "ชื่อบทบาท",
        "persona_placeholder": "ตัวอย่าง: ผู้ช่วยรีวิวโค้ด / ผู้ช่วยวางแผนทริป / ผู้ช่วย PM",
        "persona_description": "คำอธิบาย",
        "persona_description_placeholder": "ตัวอย่าง: ตอบสั้นและเน้น bug ความเสี่ยง และ regression",
        "system_prompt": "System Prompt",
        "system_prompt_placeholder": "ตัวอย่าง: คุณคือผู้รีวิวโค้ดระดับอาวุโส ให้เน้น bug ความเสี่ยง และ test ที่ขาด",
        "default_model": "โมเดลเริ่มต้น",
        "default_preset": "พรีเซ็ตเริ่มต้น",
        "save_persona": "บันทึกบทบาท",
        "delete_persona": "ลบบทบาท",
        "prompt_library_accordion": "คลังพรอมป์",
        "prompt_label": "พรอมป์",
        "prompt_name": "ชื่อพรอมป์",
        "prompt_placeholder": "สร้างหรืออัปเดตพรอมป์",
        "category": "หมวดหมู่",
        "category_placeholder": "general / writing / coding ...",
        "prompt_content": "เนื้อหาพรอมป์",
        "prompt_content_placeholder": "เขียนพรอมป์ที่ใช้ซ้ำได้ที่นี่",
        "favorite": "รายการโปรด",
        "save_prompt": "บันทึกพรอมป์",
        "delete_prompt": "ลบพรอมป์",
        "insert_prompt": "แทรกลงช่องป้อนข้อความ",
        "advanced_accordion": "ขั้นสูง",
        "language_label": "ภาษา",
        "temperature": "Temperature",
        "temperature_info": "เลือกค่าระหว่าง 0 ถึง 1",
        "max_tokens": "Max Tokens",
        "top_p": "Top P",
        "typical_p": "Typical P",
        "num_ctx": "Num CTX",
        "save_llm_settings": "บันทึกการตั้งค่า LLM",
        "tool_commands": (
            "**คำสั่งเครื่องมือ**  \n"
            "`/tools`  \n"
            "`/tool calculator {\"expression\":\"2+3*4\"}`  \n"
            "`/tool datetime {}`  \n"
            "`/tool web_search {\"query\":\"latest ollama news\",\"num_results\":3}`  \n"
            "`/tool fetch_url {\"url\":\"https://example.com\"}`"
        ),
        "test_connection": "ทดสอบการเชื่อมต่อ",
        "export_chat": "ส่งออกแชต",
        "connection_untested": "ยังไม่ได้ทดสอบ",
        "web_search_setting_saved": "บันทึกการตั้งค่าค้นหาเว็บแล้ว",
        "web_search_setting_failed": "บันทึกการตั้งค่าค้นหาเว็บไม่สำเร็จ",
        "settings_saved": "บันทึกการตั้งค่าการค้นหาแล้ว",
        "settings_save_failed": "บันทึกการตั้งค่าการค้นหาไม่สำเร็จ",
        "llm_settings_saved": "บันทึกการตั้งค่า LLM ไปยัง server_settings.json แล้ว",
        "llm_settings_save_failed": "บันทึกการตั้งค่า LLM ไม่สำเร็จ",
    },
}


def _resolved_translations(language_settings: dict, selected_language: str) -> dict:
    merged = dict(UI_DEFAULT_TRANSLATIONS["English"])
    english_translations = language_settings.get("languages", {}).get("English", {})
    selected_translations = language_settings.get("languages", {}).get(selected_language, {})
    if isinstance(english_translations, dict):
        merged.update(english_translations)
    if isinstance(selected_translations, dict):
        merged.update(selected_translations)
    merged.update(UI_DEFAULT_TRANSLATIONS.get(selected_language, {}))
    return merged


def _summary_length_choices(translations: dict) -> list[tuple[str, str]]:
    return [
        (translations.get("search_summary_short", "Short"), "short"),
        (translations.get("search_summary_medium", "Medium"), "medium"),
        (translations.get("search_summary_long", "Long"), "long"),
    ]


def _default_llm_connection_light(translations: dict) -> str:
    label = translations.get("connection_untested", "Not tested")
    return (
        "<div style='display:inline-flex;align-items:center;gap:8px;"
        "padding:0;background:transparent;'>"
        "<span style='width:10px;height:10px;border-radius:50%;background:#f59e0b;display:inline-block;'></span>"
        f"<span style='font-size:13px;'>{label}</span>"
        "</div>"
    )


DEFAULT_LLM_CONNECTION_LIGHT = _default_llm_connection_light(UI_DEFAULT_TRANSLATIONS["English"])

def export_chat_ui() -> str:
    export_path, status = export_current_chat_markdown()
    if export_path:
        return f"{status}: {export_path}"
    return status

def _clamp_float(value: float | int | str, minimum: float, maximum: float, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _clamp_int(value: float | int | str, minimum: int, maximum: int, default: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _normalize_provider(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"serper", "serper.dev", "serper_dev"}:
        return "serper.dev"
    if v == "tavily":
        return "tavily"
    return "serper.dev"


def _normalize_summary_length(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"short", "medium", "long"}:
        return v
    return "medium"


def _normalize_bool(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "on"}:
            return True
        if v in {"false", "0", "no", "off"}:
            return False
    return default

def build_demo() -> gr.Blocks:
    language_settings = load_language_settings()
    current_language = language_settings.get("default_language", "English")
    translations = _resolved_translations(language_settings, current_language)
    set_session_label_language(current_language)

    settings = load_app_settings()
    ui_settings = settings.get("ui", {}) if isinstance(settings.get("ui", {}), dict) else {}
    session_preferences = get_current_session_preferences()
    search_cfg = settings.get("search", {}) if isinstance(settings.get("search", {}), dict) else {}
    default_search_provider = _normalize_provider(str(search_cfg.get("provider", "serper.dev")))
    default_tavily_api_key = str(search_cfg.get("tavily_api_key", ""))
    default_serper_api_key = str(search_cfg.get("serper_api_key", ""))
    default_search_num_results = _clamp_int(search_cfg.get("num_results", 5), 1, 20, 5)
    default_search_summary_length = _normalize_summary_length(str(search_cfg.get("summary_length", "medium")))
    default_web_search_enabled = _normalize_bool(search_cfg.get("enabled", False), default=False)
    llm_defaults = load_llm_parameters()
    default_llm_temperature = _clamp_float(llm_defaults.get("llm_temperature", 0.8), 0.0, 1.0, 0.8)
    default_llm_max_tokens = _clamp_int(llm_defaults.get("llm_max_tokens", 2048), 1, 131072, 2048)
    default_llm_top_p = _clamp_float(llm_defaults.get("llm_top_p", 0.9), 0.0, 1.0, 0.9)
    default_llm_typical_p = _clamp_float(llm_defaults.get("llm_typical_p", 0.7), 0.0, 1.0, 0.7)
    default_llm_num_ctx = _clamp_int(llm_defaults.get("llm_num_ctx", 2048), 1, 131072, 2048)

    hosts, default_host = load_settings()
    server_choices = [(host["server_name"], f"{host['address']}:{host['port']}") for host in hosts]
    current_host = (
        str(ui_settings.get("last_selected_server") or "").strip()
        or session_preferences["server"]
        or (f"{default_host['address']}:{default_host['port']}" if default_host else None)
    )
    if current_host and "://" in current_host:
        parsed_host = urlparse(current_host)
        models = (
            fetch_models(f"{parsed_host.scheme}://{parsed_host.hostname}", parsed_host.port or 11434)
            if parsed_host.scheme and parsed_host.hostname
            else []
        )
    else:
        models = []
    initial_history = load_current_chat_history()
    session_choices, current_session_id = list_chat_session_choices()
    preset_choices, current_preset_id = list_preset_choices()
    persona_choices, current_persona_id = list_persona_choices()
    prompt_choices, current_prompt_id = list_prompt_choices()
    current_model = (
        str(ui_settings.get("last_selected_model") or "").strip()
        or session_preferences["model"]
        or (models[0] if models else None)
    )
    if current_model and current_model not in models:
        models = [*models, current_model]
    current_preset_id = (
        str(ui_settings.get("last_selected_preset_id") or "").strip()
        or session_preferences["preset_id"]
        or current_preset_id
    )
    current_persona_id = (
        str(ui_settings.get("last_selected_persona_id") or "").strip()
        or session_preferences["persona_id"]
        or current_persona_id
    )
    current_preset_name = next((label for label, value in preset_choices if value == current_preset_id), "")
    current_persona_name = next((label for label, value in persona_choices if value == current_persona_id), "")

    def update_language(selected_language):
        nonlocal current_language, translations
        current_language = selected_language
        translations = _resolved_translations(language_settings, selected_language)
        set_session_label_language(selected_language)
        save_default_language(selected_language)
        return translations

    def tr(key: str, fallback: str) -> str:
        return str(translations.get(key, fallback))

    def web_search_button_label(enabled: bool) -> str:
        return tr("web_search_on", "Web Search: ON") if enabled else tr("web_search_off", "Web Search: OFF")

    def settings_button_label(is_open: bool) -> str:
        return tr("settings_close", "Close Settings") if is_open else tr("settings_open", "Settings")

    def persist_ui_preferences(
        *,
        server: str | None = None,
        model: str | None = None,
        preset_id: str | None = None,
        persona_id: str | None = None,
    ) -> None:
        existing = load_app_settings()
        ui_config = existing.get("ui", {}) if isinstance(existing.get("ui", {}), dict) else {}
        if server is not None:
            ui_config["last_selected_server"] = server
        if model is not None:
            ui_config["last_selected_model"] = model
        if preset_id is not None:
            ui_config["last_selected_preset_id"] = preset_id
        if persona_id is not None:
            ui_config["last_selected_persona_id"] = persona_id
        existing["ui"] = ui_config
        save_app_settings(existing)

    def pin_button_label(is_pinned: bool) -> str:
        return tr("unpin_session", "Unpin Session") if is_pinned else tr("pin_session", "Pin Session")

    SESSION_BUTTON_SLOTS = 12

    def filtered_session_choices(search_query: str | None):
        choices, current_id = list_chat_session_choices()
        normalized = (search_query or "").strip().lower()
        if not normalized:
            return choices, current_id
        filtered = [(label, session_id) for label, session_id in choices if normalized in label.lower()]
        return filtered, current_id

    def build_session_button_updates(active_session_id: str | None, search_query: str | None = None):
        choices, current_id = filtered_session_choices(search_query)
        selected_id = active_session_id or current_id
        button_updates = []
        button_ids = []
        for index in range(SESSION_BUTTON_SLOTS):
            if index < len(choices):
                label, session_id = choices[index]
                button_updates.append(
                    gr.update(
                        value=label,
                        visible=True,
                        variant="primary" if session_id == selected_id else "secondary",
                    )
                )
                button_ids.append(session_id)
            else:
                button_updates.append(gr.update(value="", visible=False))
                button_ids.append(None)
        return button_updates, button_ids, selected_id

    def handle_switch_session_button(session_id: str | None, search_query: str):
        button_updates, button_ids, selected_id = build_session_button_updates(session_id, search_query)
        result = switch_chat_session_with_state(selected_id)
        return (*button_updates, *button_ids, selected_id, gr.update(value=pin_button_label(is_session_pinned(selected_id))), *result)

    def handle_create_session_button(search_query: str):
        result = create_new_chat_session_with_state()
        button_updates, button_ids, selected_id = build_session_button_updates(None, search_query)
        return (*button_updates, *button_ids, selected_id, gr.update(value=pin_button_label(is_session_pinned(selected_id))), *result[1:])

    def handle_rename_session_button(session_id: str | None, title: str, search_query: str):
        _dropdown_update, status = rename_chat_session(session_id, title)
        button_updates, button_ids, selected_id = build_session_button_updates(session_id, search_query)
        return (*button_updates, *button_ids, selected_id, status)

    def handle_delete_session_button(session_id: str | None, search_query: str):
        result = delete_chat_session_with_state(session_id)
        button_updates, button_ids, selected_id = build_session_button_updates(None, search_query)
        return (*button_updates, *button_ids, selected_id, gr.update(value=pin_button_label(is_session_pinned(selected_id))), *result[1:])

    def handle_clear_current_chat_button(search_query: str):
        result = clear_current_chat_with_state()
        button_updates, button_ids, selected_id = build_session_button_updates(None, search_query)
        return (*button_updates, *button_ids, selected_id, gr.update(value=pin_button_label(is_session_pinned(selected_id))), *result)

    def handle_session_search_change(search_query: str, session_id: str | None):
        button_updates, button_ids, selected_id = build_session_button_updates(session_id, search_query)
        return (*button_updates, *button_ids, selected_id, gr.update(value=pin_button_label(is_session_pinned(selected_id))), search_query)

    def handle_toggle_pin_session(session_id: str | None, search_query: str):
        currently_pinned = is_session_pinned(session_id)
        status = set_session_pinned(session_id, not currently_pinned)
        next_pinned = is_session_pinned(session_id)
        button_updates, button_ids, selected_id = build_session_button_updates(session_id, search_query)
        return (*button_updates, *button_ids, selected_id, gr.update(value=pin_button_label(next_pinned)), status)

    def handle_server_change_persisted(selected_server: str | None):
        status, model_update, server_value = handle_server_change(selected_server)
        normalized_server = str(server_value or selected_server or "").strip() or None
        next_model = model_update.get("value") if isinstance(model_update, dict) else None
        persist_ui_preferences(
            server=normalized_server,
            model=str(next_model or "").strip() or None,
        )
        update_current_session_preferences(
            server=normalized_server,
            model=str(next_model or "").strip() or None,
        )
        return status, model_update, server_value

    def handle_model_change_persisted(selected_model: str | None, selected_server: str | None):
        normalized_model = str(selected_model or "").strip() or None
        normalized_server = str(selected_server or "").strip() or None
        persist_ui_preferences(server=normalized_server, model=normalized_model)
        update_current_session_preferences(server=normalized_server, model=normalized_model)
        return f"Model selected: {normalized_model}" if normalized_model else "Model cleared."

    def apply_preset_with_persistence(preset_id: str | None):
        result = apply_preset_to_current_session(preset_id)
        persist_ui_preferences(preset_id=preset_id)
        update_current_session_preferences(preset_id=preset_id)
        return result

    def load_persona_with_persistence(persona_id: str | None):
        result = load_selected_persona(persona_id)
        persist_ui_preferences(persona_id=persona_id)
        update_current_session_preferences(persona_id=persona_id)
        return result

    def save_preset_with_persistence(
        preset_id: str | None,
        name: str,
        temperature: float,
        max_tokens: float,
        top_p: float,
        typical_p: float,
        num_ctx: float,
    ):
        result = save_preset_from_values(preset_id, name, temperature, max_tokens, top_p, typical_p, num_ctx)
        selected_value = result[0].get("value") if isinstance(result[0], dict) else None
        persist_ui_preferences(preset_id=str(selected_value or "").strip() or None)
        update_current_session_preferences(preset_id=str(selected_value or "").strip() or None)
        return result

    def save_persona_with_persistence(
        persona_id: str | None,
        name: str,
        description: str,
        system_prompt: str,
        default_model: str | None,
        default_preset: str | None,
    ):
        result = save_persona(persona_id, name, description, system_prompt, default_model, default_preset)
        selected_value = result[0].get("value") if isinstance(result[0], dict) else None
        persist_ui_preferences(persona_id=str(selected_value or "").strip() or None)
        update_current_session_preferences(persona_id=str(selected_value or "").strip() or None)
        return result

    def delete_preset_with_persistence(preset_id: str | None):
        result = delete_selected_preset(preset_id)
        selected_value = result[0].get("value") if isinstance(result[0], dict) else None
        persist_ui_preferences(preset_id=str(selected_value or "").strip() or None)
        update_current_session_preferences(preset_id=str(selected_value or "").strip() or None)
        return result

    def delete_persona_with_persistence(persona_id: str | None):
        result = delete_selected_persona(persona_id)
        selected_value = result[0].get("value") if isinstance(result[0], dict) else None
        persist_ui_preferences(persona_id=str(selected_value or "").strip() or None)
        update_current_session_preferences(persona_id=str(selected_value or "").strip() or None)
        return result

    def toggle_web_search(enabled: bool):
        new_enabled = not bool(enabled)
        label = web_search_button_label(new_enabled)
        variant = "primary" if new_enabled else "secondary"
        existing = load_app_settings()
        existing_search = existing.get("search", {}) if isinstance(existing.get("search", {}), dict) else {}
        existing["search"] = {
            **existing_search,
            "enabled": new_enabled,
        }
        ok = save_app_settings(existing)
        status = tr("web_search_setting_saved", "Web search setting saved.") if ok else tr("web_search_setting_failed", "Failed to save web search setting.")
        return (
            gr.update(value=new_enabled),
            gr.update(value=label, variant=variant),
            status,
        )

    def toggle_settings_drawer(is_open: bool):
        new_open = not bool(is_open)
        label = settings_button_label(new_open)
        variant = "primary" if new_open else "secondary"
        return (
            gr.update(value=new_open),
            gr.update(visible=new_open),
            gr.update(value=label, variant=variant),
        )

    def save_search_settings_ui(
        provider: str,
        tavily_key: str,
        serper_key: str,
        num_results: float,
        summary_length: str,
    ):
        existing = load_app_settings()
        existing_search = existing.get("search", {}) if isinstance(existing.get("search", {}), dict) else {}

        num = _clamp_int(num_results, 1, 20, 5)
        normalized_summary_length = _normalize_summary_length(summary_length)
        normalized_provider = _normalize_provider(provider)

        existing["search"] = {
            "provider": normalized_provider,
            "num_results": num,
            "summary_length": normalized_summary_length,
            "tavily_api_key": tavily_key.strip(),
            "serper_api_key": serper_key.strip(),
            "tavily_api_url": str(existing_search.get("tavily_api_url", "https://api.tavily.com/search")),
            "serper_api_url": str(existing_search.get("serper_api_url", "https://google.serper.dev/search")),
        }

        ok = save_app_settings(existing)
        if ok:
            return (
                tr("settings_saved", "Search settings saved."),
                gr.update(value=normalized_provider),
                gr.update(value=num),
                gr.update(value=normalized_summary_length),
                normalized_provider,
                num,
                normalized_summary_length,
            )
        return (
            tr("settings_save_failed", "Failed to save search settings."),
            gr.update(),
            gr.update(),
            gr.update(),
            normalized_provider,
            num,
            normalized_summary_length,
        )

    def autosave_search_settings_ui(
        provider: str,
        tavily_key: str,
        serper_key: str,
        num_results: float,
        summary_length: str,
    ):
        status, provider_update, results_update, summary_update, provider_state, results_state, summary_state = save_search_settings_ui(
            provider,
            tavily_key,
            serper_key,
            num_results,
            summary_length,
        )
        return (
            status,
            provider_update,
            results_update,
            summary_update,
            provider_state,
            results_state,
            summary_state,
        )

    def save_llm_settings_ui(
        temperature: float,
        max_tokens: float,
        top_p: float,
        typical_p: float,
        num_ctx: float,
    ):
        payload = {
            "llm_temperature": _clamp_float(temperature, 0.0, 1.0, default_llm_temperature),
            "llm_max_tokens": _clamp_int(max_tokens, 1, 131072, default_llm_max_tokens),
            "llm_top_p": _clamp_float(top_p, 0.0, 1.0, default_llm_top_p),
            "llm_typical_p": _clamp_float(typical_p, 0.0, 1.0, default_llm_typical_p),
            "llm_num_ctx": _clamp_int(num_ctx, 1, 131072, default_llm_num_ctx),
        }

        ok = save_llm_parameters(payload)
        if ok:
            return (
                tr("llm_settings_saved", "LLM settings saved to server_settings.json."),
                gr.update(value=payload["llm_temperature"]),
                gr.update(value=payload["llm_max_tokens"]),
                gr.update(value=payload["llm_top_p"]),
                gr.update(value=payload["llm_typical_p"]),
                gr.update(value=payload["llm_num_ctx"]),
                payload["llm_temperature"],
                payload["llm_max_tokens"],
                payload["llm_top_p"],
                payload["llm_typical_p"],
                payload["llm_num_ctx"],
            )
        return (
            tr("llm_settings_save_failed", "Failed to save LLM settings."),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    with gr.Blocks() as demo:
        title_markdown = gr.Markdown(tr("title", "# Ollama WebUI"))

        settings_open_state = gr.Checkbox(value=False, visible=False)
        search_provider_state = gr.State(default_search_provider)
        search_num_results_state = gr.State(default_search_num_results)
        search_summary_length_state = gr.State(default_search_summary_length)
        session_id_state = gr.State(current_session_id)
        session_search_state = gr.State(value="")
        with gr.Row(equal_height=False, elem_classes=["app-shell"]):
            with gr.Column(scale=2, elem_classes=["sessions-panel"]):
                sessions_heading = gr.Markdown(tr("sessions_heading", "### Sessions"))
                session_button_updates, session_button_ids, _selected_session_id = build_session_button_updates(current_session_id)
                session_button_states = [gr.State(value=session_id) for session_id in session_button_ids]
                session_search_input = gr.Textbox(
                    label=tr("session_search", "Search Sessions"),
                    placeholder=tr("session_search_placeholder", "Filter by title or time"),
                )
                with gr.Column(elem_id="session-list", elem_classes=["session-stack"]):
                    session_buttons = []
                    for index in range(SESSION_BUTTON_SLOTS):
                        button = gr.Button(
                            value=session_button_updates[index]["value"] if isinstance(session_button_updates[index], dict) else "",
                            visible=session_button_updates[index]["visible"] if isinstance(session_button_updates[index], dict) else False,
                            variant="primary" if session_button_ids[index] == current_session_id else "secondary",
                            elem_classes=["session-item"],
                        )
                        session_buttons.append(button)
                rename_session_input = gr.Textbox(
                    label=tr("session_title", "Session Title"),
                    placeholder=tr("rename_current_chat", "Rename current chat"),
                )
                with gr.Row():
                    pin_session_button = gr.Button(
                        value=pin_button_label(is_session_pinned(current_session_id)),
                        variant="secondary",
                    )
                    new_session_button = gr.Button(value=tr("new_chat_button", "New Chat"), variant="primary")
                    rename_session_button = gr.Button(value=tr("rename_button", "Rename"), variant="secondary")
                    delete_session_button = gr.Button(value=tr("delete_button", "Delete"), variant="secondary")

            with gr.Column(scale=9, elem_classes=["chat-panel"]):
                with gr.Row():
                    server_dropdown = gr.Dropdown(
                        choices=server_choices,
                        allow_custom_value=True,
                        label=translations.get("select_server", "Select Server"),
                        value=current_host,
                        scale=3,
                    )
                    model_dropdown = gr.Dropdown(
                        choices=models,
                        allow_custom_value=True,
                        label=translations.get("select_model", "Select Model"),
                        value=current_model,
                        scale=3,
                    )
                    web_search_enabled = gr.Checkbox(value=default_web_search_enabled, visible=False)
                    web_search_toggle = gr.Button(
                        value=web_search_button_label(default_web_search_enabled),
                        variant="primary" if default_web_search_enabled else "secondary",
                        scale=2,
                    )
                    settings_toggle_button = gr.Button(value=settings_button_label(False), variant="secondary", scale=2)

                with gr.Row(elem_classes=["quick-control-row"]):
                    preset_dropdown = gr.Dropdown(
                        choices=preset_choices,
                        value=current_preset_id,
                        label=tr("preset_label", "Preset"),
                        interactive=True,
                        scale=1,
                    )
                    persona_dropdown = gr.Dropdown(
                        choices=persona_choices,
                        value=current_persona_id,
                        label=tr("persona_label", "Persona"),
                        interactive=True,
                        scale=1,
                    )

                chatbot = gr.Chatbot(
                    buttons=["copy", "copy_all"],
                    height=560,
                    autoscroll=True,
                    value=initial_history,
                )

                question_input = gr.MultimodalTextbox(
                    interactive=True,
                    autoscroll=True,
                    file_count="single",
                    placeholder=translations.get("enter_question", "Enter your question or upload image file with question..."),
                    show_label=False,
                    file_types=[".jpg", ".jpeg", ".png", ".bmp"],
                )

                with gr.Row(equal_height=True, elem_classes=["toolbar-row"]):
                    test_llm_connection_button = gr.Button(
                        value=tr("test_connection", "Test Connection"),
                        elem_classes=["my-button", "toolbar-button"],
                        scale=0,
                    )
                    llm_connection_light = gr.HTML(
                        value=_default_llm_connection_light(translations),
                        elem_classes=["status-light"],
                        scale=0,
                    )
                    stop_button = gr.Button(
                        value=translations.get("stop_button", "Stop Answer"),
                        elem_classes=["my-button", "toolbar-button"],
                        variant="primary",
                        scale=0,
                    )
                    clean_answer_button = gr.Button(
                        value=translations.get("clean_answer_button", "Clear Answer"),
                        elem_classes=["my-button", "toolbar-button"],
                        variant="secondary",
                        scale=0,
                    )
                    export_chat_button = gr.Button(
                        value=tr("export_chat", "Export Chat"),
                        elem_classes=["my-button", "toolbar-button"],
                        variant="secondary",
                        scale=0,
                    )

                status_output = gr.Textbox(
                    label=translations.get("status_display", "Status Display"),
                    show_label=True,
                    placeholder=translations.get(
                        "status_display_msg",
                        "Displaying the model's status or displaying an error message.",
                    ),
                    lines=2,
                    max_lines=3,
                )
                llm_temperature = gr.Slider(
                    0,
                    1,
                    step=0.1,
                    value=default_llm_temperature,
                    label=tr("temperature", "Temperature"),
                    info=tr("temperature_info", "Choose between 0 and 1"),
                    interactive=True,
                    visible=False,
                )
                llm_max_tokens = gr.Number(label=tr("max_tokens", "Max Tokens"), interactive=True, value=default_llm_max_tokens, visible=False)
                llm_top_p = gr.Slider(0, 1, step=0.05, value=default_llm_top_p, label=tr("top_p", "Top P"), interactive=True, visible=False)
                llm_typical_p = gr.Slider(
                    0,
                    1,
                    step=0.05,
                    value=default_llm_typical_p,
                    label=tr("typical_p", "Typical P"),
                    interactive=True,
                    visible=False,
                )
                llm_num_ctx = gr.Number(label=tr("num_ctx", "Num CTX"), interactive=True, value=default_llm_num_ctx, visible=False)

                chat_interface = gr.ChatInterface(
                    fn=ask_question_stream,
                    multimodal=True,
                    additional_inputs=[
                        model_dropdown,
                        server_dropdown,
                        llm_temperature,
                        llm_max_tokens,
                        llm_top_p,
                        llm_typical_p,
                        llm_num_ctx,
                        search_provider_state,
                        web_search_enabled,
                        search_num_results_state,
                        search_summary_length_state,
                    ],
                    fill_width=True,
                    fill_height=True,
                    chatbot=chatbot,
                    autofocus=False,
                    show_progress="full",
                    textbox=question_input,
                    additional_outputs=[status_output],
                )

            with gr.Column(scale=4, visible=False, elem_classes=["settings-panel"]) as settings_drawer:
                settings_heading = gr.Markdown(tr("settings_heading", "### Settings"))
                settings_note = gr.Markdown(tr("settings_note", "<div class='drawer-note'>All settings live here and apply immediately after saving.</div>"))

                common_section_label = gr.Markdown(
                    f"<div class='settings-section-label'>{tr('settings_common', 'Common')}</div>"
                )
                with gr.Accordion(tr("search_accordion", "Search"), open=True) as search_accordion:
                    search_provider_dropdown = gr.Dropdown(
                        choices=[("Serper.dev", "serper.dev"), ("Tavily", "tavily")],
                        label=tr("search_provider", "Search Provider"),
                        value=default_search_provider,
                        interactive=True,
                    )
                    search_num_results_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=max(1, min(default_search_num_results, 20)),
                        label=tr("search_results_count", "Search Results Count"),
                        info=tr("search_results_info", "How many web search results to use for answer grounding"),
                        interactive=True,
                    )
                    search_summary_length_dropdown = gr.Dropdown(
                        choices=_summary_length_choices(translations),
                        label=tr("search_summary_length", "Search Summary Length"),
                        value=default_search_summary_length,
                        info=tr("search_summary_info", "Controls how detailed search summaries should be"),
                        interactive=True,
                    )
                    tavily_api_key_input = gr.Textbox(
                        label=tr("tavily_api_key", "Tavily API Key"),
                        value=default_tavily_api_key,
                        type="password",
                        placeholder=tr("tavily_api_key_placeholder", "Paste Tavily API key"),
                    )
                    serper_api_key_input = gr.Textbox(
                        label=tr("serper_api_key", "Serper.dev API Key"),
                        value=default_serper_api_key,
                        type="password",
                        placeholder=tr("serper_api_key_placeholder", "Paste Serper.dev API key"),
                    )
                    save_search_setting_button = gr.Button(value=tr("save_search_settings", "Save Search Settings"), variant="secondary")

                with gr.Accordion(tr("presets_accordion", "Presets"), open=False) as presets_accordion:
                    preset_name_input = gr.Textbox(
                        label=tr("preset_name", "Preset Name"),
                        value=current_preset_name,
                        placeholder=tr("preset_placeholder", "Create or update a custom preset"),
                    )
                    with gr.Row():
                        save_preset_button = gr.Button(value=tr("save_preset", "Save Preset"), variant="secondary")
                        delete_preset_button = gr.Button(value=tr("delete_preset", "Delete Preset"), variant="secondary")

                with gr.Accordion(tr("personas_accordion", "Personas"), open=False) as personas_accordion:
                    persona_name_input = gr.Textbox(
                        label=tr("persona_name", "Persona Name"),
                        value=current_persona_name,
                        placeholder=tr("persona_placeholder", "Create or update a persona"),
                    )
                    persona_description_input = gr.Textbox(
                        label=tr("persona_description", "Description"),
                        placeholder=tr("persona_description_placeholder", "Short note about this persona"),
                    )
                    persona_system_prompt_input = gr.Textbox(
                        label=tr("system_prompt", "System Prompt"),
                        lines=6,
                        placeholder=tr("system_prompt_placeholder", "You are a pragmatic assistant..."),
                    )
                    persona_default_model_input = gr.Dropdown(
                        choices=models,
                        allow_custom_value=True,
                        label=tr("default_model", "Default Model"),
                        value=None,
                        interactive=True,
                    )
                    persona_default_preset_dropdown = gr.Dropdown(
                        choices=preset_choices,
                        value=None,
                        label=tr("default_preset", "Default Preset"),
                        interactive=True,
                    )
                    with gr.Row():
                        save_persona_button = gr.Button(value=tr("save_persona", "Save Persona"), variant="secondary")
                        delete_persona_button = gr.Button(value=tr("delete_persona", "Delete Persona"), variant="secondary")

                workspace_section_label = gr.Markdown(
                    f"<div class='settings-section-label'>{tr('settings_workspace', 'Workspace')}</div>"
                )
                with gr.Accordion(tr("prompt_library_accordion", "Prompt Library"), open=False) as prompt_library_accordion:
                    prompt_dropdown = gr.Dropdown(
                        choices=prompt_choices,
                        value=current_prompt_id,
                        label=tr("prompt_label", "Prompt"),
                        interactive=True,
                    )
                    prompt_name_input = gr.Textbox(
                        label=tr("prompt_name", "Prompt Name"),
                        placeholder=tr("prompt_placeholder", "Create or update a prompt"),
                    )
                    prompt_category_input = gr.Textbox(
                        label=tr("category", "Category"),
                        value="general",
                        placeholder=tr("category_placeholder", "general / writing / coding ..."),
                    )
                    prompt_content_input = gr.Textbox(
                        label=tr("prompt_content", "Prompt Content"),
                        lines=6,
                        placeholder=tr("prompt_content_placeholder", "Write the reusable prompt here"),
                    )
                    prompt_favorite_input = gr.Checkbox(
                        label=tr("favorite", "Favorite"),
                        value=False,
                    )
                    with gr.Row():
                        save_prompt_button = gr.Button(value=tr("save_prompt", "Save Prompt"), variant="secondary")
                        delete_prompt_button = gr.Button(value=tr("delete_prompt", "Delete Prompt"), variant="secondary")
                    insert_prompt_button = gr.Button(value=tr("insert_prompt", "Insert to Input"), variant="secondary")

                system_section_label = gr.Markdown(
                    f"<div class='settings-section-label'>{tr('settings_system', 'System')}</div>"
                )
                with gr.Accordion(tr("server_accordion", "Server"), open=False) as server_accordion:
                    new_server_name_input = gr.Textbox(
                        label=translations.get("ai_server_name", "Ai Server Name"),
                        placeholder=translations.get(
                            "ai_server_name_msg",
                            "Ex: GMK-K9, editable via manual editing, file name for settings: server_settings.json.",
                        ),
                    )
                    new_address_input = gr.Textbox(
                        label=translations.get("new_server_address", "New Server Address"),
                        placeholder=translations.get("new_server_address_msg", "Ex. http://127.0.0.1"),
                    )
                    new_port_input = gr.Number(label=translations.get("new_server_port", "Port"), value=11434)
                    new_default_server = gr.Checkbox(
                        label=translations.get("default_start_setting", "Make the new host by default."),
                        value=False,
                    )
                    add_server_button = gr.Button(value=translations.get("add_ai_server", "Add Ai Server"))

                with gr.Accordion(tr("advanced_accordion", "Advanced"), open=False) as advanced_accordion:
                    language_dropdown = gr.Dropdown(
                        choices=list(language_settings["languages"].keys()),
                        label=tr("language_label", "Language"),
                        value=current_language,
                        interactive=True,
                    )
                    llm_temperature_view = gr.Slider(
                        0,
                        1,
                        step=0.1,
                        value=default_llm_temperature,
                        label=tr("temperature", "Temperature"),
                        info=tr("temperature_info", "Choose between 0 and 1"),
                        interactive=True,
                    )
                    llm_max_tokens_view = gr.Number(label=tr("max_tokens", "Max Tokens"), interactive=True, value=default_llm_max_tokens)
                    llm_top_p_view = gr.Slider(0, 1, step=0.05, value=default_llm_top_p, label=tr("top_p", "Top P"), interactive=True)
                    llm_typical_p_view = gr.Slider(
                        0,
                        1,
                        step=0.05,
                        value=default_llm_typical_p,
                        label=tr("typical_p", "Typical P"),
                        interactive=True,
                    )
                    llm_num_ctx_view = gr.Number(label=tr("num_ctx", "Num CTX"), interactive=True, value=default_llm_num_ctx)
                    save_llm_setting_button = gr.Button(value=tr("save_llm_settings", "Save LLM Settings"), variant="secondary")
                    tool_commands_markdown = gr.Markdown(tr("tool_commands", UI_DEFAULT_TRANSLATIONS["English"]["tool_commands"]))

                server_name = gr.Textbox(
                    label=translations.get("server_address", "Server Address"),
                    value=current_host,
                    interactive=False,
                    visible=False,
                )

        # Sync drawer controls to hidden runtime inputs.
        search_provider_dropdown.change(lambda x: x, inputs=[search_provider_dropdown], outputs=[search_provider_state], queue=False)
        search_num_results_slider.change(lambda x: int(x), inputs=[search_num_results_slider], outputs=[search_num_results_state], queue=False)
        search_summary_length_dropdown.change(
            lambda x: _normalize_summary_length(str(x)),
            inputs=[search_summary_length_dropdown],
            outputs=[search_summary_length_state],
            queue=False,
        )
        search_provider_dropdown.change(
            fn=autosave_search_settings_ui,
            inputs=[
                search_provider_dropdown,
                tavily_api_key_input,
                serper_api_key_input,
                search_num_results_slider,
                search_summary_length_dropdown,
            ],
            outputs=[
                status_output,
                search_provider_dropdown,
                search_num_results_slider,
                search_summary_length_dropdown,
                search_provider_state,
                search_num_results_state,
                search_summary_length_state,
            ],
            queue=False,
        )
        search_num_results_slider.change(
            fn=autosave_search_settings_ui,
            inputs=[
                search_provider_dropdown,
                tavily_api_key_input,
                serper_api_key_input,
                search_num_results_slider,
                search_summary_length_dropdown,
            ],
            outputs=[
                status_output,
                search_provider_dropdown,
                search_num_results_slider,
                search_summary_length_dropdown,
                search_provider_state,
                search_num_results_state,
                search_summary_length_state,
            ],
            queue=False,
        )
        search_summary_length_dropdown.change(
            fn=autosave_search_settings_ui,
            inputs=[
                search_provider_dropdown,
                tavily_api_key_input,
                serper_api_key_input,
                search_num_results_slider,
                search_summary_length_dropdown,
            ],
            outputs=[
                status_output,
                search_provider_dropdown,
                search_num_results_slider,
                search_summary_length_dropdown,
                search_provider_state,
                search_num_results_state,
                search_summary_length_state,
            ],
            queue=False,
        )
        tavily_api_key_input.blur(
            fn=autosave_search_settings_ui,
            inputs=[
                search_provider_dropdown,
                tavily_api_key_input,
                serper_api_key_input,
                search_num_results_slider,
                search_summary_length_dropdown,
            ],
            outputs=[
                status_output,
                search_provider_dropdown,
                search_num_results_slider,
                search_summary_length_dropdown,
                search_provider_state,
                search_num_results_state,
                search_summary_length_state,
            ],
            queue=False,
        )
        serper_api_key_input.blur(
            fn=autosave_search_settings_ui,
            inputs=[
                search_provider_dropdown,
                tavily_api_key_input,
                serper_api_key_input,
                search_num_results_slider,
                search_summary_length_dropdown,
            ],
            outputs=[
                status_output,
                search_provider_dropdown,
                search_num_results_slider,
                search_summary_length_dropdown,
                search_provider_state,
                search_num_results_state,
                search_summary_length_state,
            ],
            queue=False,
        )
        llm_temperature_view.change(lambda x: x, inputs=[llm_temperature_view], outputs=[llm_temperature], queue=False)
        llm_max_tokens_view.change(lambda x: x, inputs=[llm_max_tokens_view], outputs=[llm_max_tokens], queue=False)
        llm_top_p_view.change(lambda x: x, inputs=[llm_top_p_view], outputs=[llm_top_p], queue=False)
        llm_typical_p_view.change(lambda x: x, inputs=[llm_typical_p_view], outputs=[llm_typical_p], queue=False)
        llm_num_ctx_view.change(lambda x: x, inputs=[llm_num_ctx_view], outputs=[llm_num_ctx], queue=False)

        preset_dropdown.change(
            fn=apply_preset_with_persistence,
            inputs=[preset_dropdown],
            outputs=[
                preset_dropdown,
                preset_name_input,
                llm_temperature_view,
                llm_max_tokens_view,
                llm_top_p_view,
                llm_typical_p_view,
                llm_num_ctx_view,
                llm_temperature,
                llm_max_tokens,
                llm_top_p,
                llm_typical_p,
                llm_num_ctx,
                status_output,
            ],
            queue=False,
        )

        save_preset_button.click(
            fn=save_preset_with_persistence,
            inputs=[
                preset_dropdown,
                preset_name_input,
                llm_temperature_view,
                llm_max_tokens_view,
                llm_top_p_view,
                llm_typical_p_view,
                llm_num_ctx_view,
            ],
            outputs=[preset_dropdown, persona_default_preset_dropdown, status_output],
            queue=False,
        )

        delete_preset_button.click(
            fn=delete_preset_with_persistence,
            inputs=[preset_dropdown],
            outputs=[preset_dropdown, persona_default_preset_dropdown, status_output],
            queue=False,
        )

        persona_dropdown.change(
            fn=load_persona_with_persistence,
            inputs=[persona_dropdown],
            outputs=[
                persona_dropdown,
                persona_name_input,
                persona_description_input,
                persona_system_prompt_input,
                persona_default_model_input,
                persona_default_preset_dropdown,
                preset_dropdown,
                model_dropdown,
                llm_temperature_view,
                llm_max_tokens_view,
                llm_top_p_view,
                llm_typical_p_view,
                llm_num_ctx_view,
                llm_temperature,
                llm_max_tokens,
                llm_top_p,
                llm_typical_p,
                llm_num_ctx,
                status_output,
            ],
            queue=False,
        )

        save_persona_button.click(
            fn=save_persona_with_persistence,
            inputs=[
                persona_dropdown,
                persona_name_input,
                persona_description_input,
                persona_system_prompt_input,
                persona_default_model_input,
                persona_default_preset_dropdown,
            ],
            outputs=[persona_dropdown, status_output],
            queue=False,
        )

        delete_persona_button.click(
            fn=delete_persona_with_persistence,
            inputs=[persona_dropdown],
            outputs=[
                persona_dropdown,
                persona_name_input,
                persona_description_input,
                persona_system_prompt_input,
                persona_default_model_input,
                persona_default_preset_dropdown,
                status_output,
            ],
            queue=False,
        )

        prompt_dropdown.change(
            fn=load_selected_prompt,
            inputs=[prompt_dropdown],
            outputs=[
                prompt_dropdown,
                prompt_name_input,
                prompt_category_input,
                prompt_content_input,
                prompt_favorite_input,
                status_output,
            ],
            queue=False,
        )

        save_prompt_button.click(
            fn=save_prompt_entry,
            inputs=[
                prompt_dropdown,
                prompt_name_input,
                prompt_category_input,
                prompt_content_input,
                prompt_favorite_input,
            ],
            outputs=[prompt_dropdown, status_output],
            queue=False,
        )

        delete_prompt_button.click(
            fn=delete_selected_prompt,
            inputs=[prompt_dropdown],
            outputs=[
                prompt_dropdown,
                prompt_name_input,
                prompt_category_input,
                prompt_content_input,
                prompt_favorite_input,
                status_output,
            ],
            queue=False,
        )

        insert_prompt_button.click(
            fn=insert_selected_prompt_into_workspace,
            inputs=[prompt_dropdown, question_input],
            outputs=[question_input, status_output],
            queue=False,
        )

        web_search_toggle.click(
            fn=toggle_web_search,
            inputs=[web_search_enabled],
            outputs=[web_search_enabled, web_search_toggle, status_output],
            queue=False,
        )

        settings_toggle_button.click(
            fn=toggle_settings_drawer,
            inputs=[settings_open_state],
            outputs=[settings_open_state, settings_drawer, settings_toggle_button],
            queue=False,
        )

        save_search_setting_button.click(
            fn=save_search_settings_ui,
            inputs=[
                search_provider_dropdown,
                tavily_api_key_input,
                serper_api_key_input,
                search_num_results_slider,
                search_summary_length_dropdown,
            ],
            outputs=[
                status_output,
                search_provider_dropdown,
                search_num_results_slider,
                search_summary_length_dropdown,
                search_provider_state,
                search_num_results_state,
                search_summary_length_state,
            ],
            queue=False,
        )

        save_llm_setting_button.click(
            fn=save_llm_settings_ui,
            inputs=[llm_temperature_view, llm_max_tokens_view, llm_top_p_view, llm_typical_p_view, llm_num_ctx_view],
            outputs=[
                status_output,
                llm_temperature_view,
                llm_max_tokens_view,
                llm_top_p_view,
                llm_typical_p_view,
                llm_num_ctx_view,
                llm_temperature,
                llm_max_tokens,
                llm_top_p,
                llm_typical_p,
                llm_num_ctx,
            ],
            queue=False,
        )

        test_llm_connection_button.click(
            fn=test_llm_connection,
            inputs=[server_dropdown, model_dropdown],
            outputs=[status_output, llm_connection_light],
            queue=False,
        )

        add_server_button.click(
            handle_add_server,
            inputs=[new_server_name_input, new_address_input, new_port_input, new_default_server],
            outputs=[status_output, server_dropdown, model_dropdown, new_server_name_input, new_address_input],
        )

        server_dropdown.change(
            handle_server_change_persisted,
            inputs=[server_dropdown],
            outputs=[status_output, model_dropdown, server_name],
        )

        model_dropdown.change(
            fn=handle_model_change_persisted,
            inputs=[model_dropdown, server_dropdown],
            outputs=[status_output],
            queue=False,
        )

        stop_button.click(
            fn=stop_response,
            inputs=[],
            outputs=[question_input, status_output],
            queue=False,
        )

        export_chat_button.click(
            fn=export_chat_ui,
            inputs=[],
            outputs=[status_output],
            queue=False,
        )

        session_list_outputs = [
            *session_buttons,
            *session_button_states,
            session_id_state,
            pin_session_button,
            chatbot,
            chat_interface.chatbot_state,
            preset_dropdown,
            preset_name_input,
            persona_dropdown,
            persona_name_input,
            persona_description_input,
            persona_system_prompt_input,
            persona_default_model_input,
            persona_default_preset_dropdown,
            model_dropdown,
            llm_temperature_view,
            llm_max_tokens_view,
            llm_top_p_view,
            llm_typical_p_view,
            llm_num_ctx_view,
            llm_temperature,
            llm_max_tokens,
            llm_top_p,
            llm_typical_p,
            llm_num_ctx,
            status_output,
        ]

        session_search_input.change(
            fn=handle_session_search_change,
            inputs=[session_search_input, session_id_state],
            outputs=[*session_buttons, *session_button_states, session_id_state, pin_session_button, session_search_state],
            queue=False,
        )

        pin_session_button.click(
            fn=handle_toggle_pin_session,
            inputs=[session_id_state, session_search_state],
            outputs=[*session_buttons, *session_button_states, session_id_state, pin_session_button, status_output],
            queue=False,
        )

        for button, button_state in zip(session_buttons, session_button_states):
            button.click(
                fn=handle_switch_session_button,
                inputs=[button_state, session_search_state],
                outputs=session_list_outputs,
                queue=False,
            )

        new_session_button.click(
            fn=handle_create_session_button,
            inputs=[session_search_state],
            outputs=session_list_outputs,
            queue=False,
        )

        rename_session_button.click(
            fn=handle_rename_session_button,
            inputs=[session_id_state, rename_session_input, session_search_state],
            outputs=[*session_buttons, *session_button_states, session_id_state, status_output],
            queue=False,
        )

        delete_session_button.click(
            fn=handle_delete_session_button,
            inputs=[session_id_state, session_search_state],
            outputs=session_list_outputs,
            queue=False,
        )

        clean_answer_button.click(
            fn=handle_clear_current_chat_button,
            inputs=[session_search_state],
            outputs=session_list_outputs,
            queue=False,
        )

        def on_language_change(selected_language, web_search_enabled_value, settings_open_value, current_summary_length, current_session_value):
            new_translations = update_language(selected_language)
            normalized_summary_length = _normalize_summary_length(str(current_summary_length))
            session_updates, _session_ids, _selected_id = build_session_button_updates(current_session_value)
            return (
                gr.update(value=new_translations.get("title", "# Ollama WebUI")),
                gr.update(value=new_translations.get("sessions_heading", "### Sessions")),
                *session_updates,
                gr.update(
                    label=new_translations.get("session_search", "Search Sessions"),
                    placeholder=new_translations.get("session_search_placeholder", "Filter by title or time"),
                ),
                gr.update(
                    label=new_translations.get("session_title", "Session Title"),
                    placeholder=new_translations.get("rename_current_chat", "Rename current chat"),
                ),
                gr.update(
                    value=pin_button_label(is_session_pinned(current_session_value))
                ),
                gr.update(value=new_translations.get("new_chat_button", "New Chat")),
                gr.update(value=new_translations.get("rename_button", "Rename")),
                gr.update(value=new_translations.get("delete_button", "Delete")),
                gr.update(label=new_translations.get("select_server", "Select Server")),
                gr.update(label=new_translations.get("server_address", "Server Address")),
                gr.update(label=new_translations.get("select_model", "Select Model")),
                gr.update(label=new_translations.get("preset_label", "Preset")),
                gr.update(label=new_translations.get("persona_label", "Persona")),
                gr.update(
                    value=new_translations.get("web_search_on", "Web Search: ON")
                    if web_search_enabled_value
                    else new_translations.get("web_search_off", "Web Search: OFF")
                ),
                gr.update(
                    value=new_translations.get("settings_close", "Close Settings")
                    if settings_open_value
                    else new_translations.get("settings_open", "Settings")
                ),
                gr.update(value=new_translations.get("test_connection", "Test Connection")),
                gr.update(value=new_translations.get("export_chat", "Export Chat")),
                gr.update(value=_default_llm_connection_light(new_translations)),
                gr.update(
                    label=new_translations.get("status_display", "Status Display"),
                    placeholder=new_translations.get(
                        "status_display_msg",
                        "Displaying the model's status or displaying an error message.",
                    ),
                ),
                gr.update(
                    label=new_translations.get("enter_question", "Enter Your Question"),
                    placeholder=new_translations.get(
                        "enter_question",
                        "Enter your question or upload image file with question...",
                    ),
                ),
                gr.update(
                    label=new_translations.get("ai_server_name", "Ai Server Name"),
                    placeholder=new_translations.get(
                        "ai_server_name_msg",
                        "Ex: GMK-K9, editable via manual editing, file name for settings: server_settings.json.",
                    ),
                ),
                gr.update(
                    label=new_translations.get("new_server_address", "New Server Address"),
                    placeholder=new_translations.get("new_server_address_msg", "Ex. http://127.0.0.1"),
                ),
                gr.update(label=new_translations.get("new_server_port", "Port")),
                gr.update(label=new_translations.get("default_start_setting", "Make the new host by default.")),
                gr.update(value=new_translations.get("add_ai_server", "Add Ai Server")),
                gr.update(value=new_translations.get("settings_heading", "### Settings")),
                gr.update(value=new_translations.get("settings_note", "<div class='drawer-note'>All settings live here and apply immediately after saving.</div>")),
                gr.update(value=f"<div class='settings-section-label'>{new_translations.get('settings_common', 'Common')}</div>"),
                gr.update(label=new_translations.get("search_accordion", "Search")),
                gr.update(label=new_translations.get("search_provider", "Search Provider")),
                gr.update(
                    label=new_translations.get("search_results_count", "Search Results Count"),
                    info=new_translations.get("search_results_info", "How many web search results to use for answer grounding"),
                ),
                gr.update(
                    choices=_summary_length_choices(new_translations),
                    value=normalized_summary_length,
                    label=new_translations.get("search_summary_length", "Search Summary Length"),
                    info=new_translations.get("search_summary_info", "Controls how detailed search summaries should be"),
                ),
                gr.update(
                    label=new_translations.get("tavily_api_key", "Tavily API Key"),
                    placeholder=new_translations.get("tavily_api_key_placeholder", "Paste Tavily API key"),
                ),
                gr.update(
                    label=new_translations.get("serper_api_key", "Serper.dev API Key"),
                    placeholder=new_translations.get("serper_api_key_placeholder", "Paste Serper.dev API key"),
                ),
                gr.update(value=new_translations.get("save_search_settings", "Save Search Settings")),
                gr.update(label=new_translations.get("server_accordion", "Server")),
                gr.update(label=new_translations.get("presets_accordion", "Presets")),
                gr.update(label=new_translations.get("preset_label", "Preset")),
                gr.update(
                    label=new_translations.get("preset_name", "Preset Name"),
                    placeholder=new_translations.get("preset_placeholder", "Create or update a custom preset"),
                ),
                gr.update(value=new_translations.get("save_preset", "Save Preset")),
                gr.update(value=new_translations.get("delete_preset", "Delete Preset")),
                gr.update(label=new_translations.get("personas_accordion", "Personas")),
                gr.update(label=new_translations.get("persona_label", "Persona")),
                gr.update(
                    label=new_translations.get("persona_name", "Persona Name"),
                    placeholder=new_translations.get("persona_placeholder", "Create or update a persona"),
                ),
                gr.update(
                    label=new_translations.get("persona_description", "Description"),
                    placeholder=new_translations.get("persona_description_placeholder", "Short note about this persona"),
                ),
                gr.update(
                    label=new_translations.get("system_prompt", "System Prompt"),
                    placeholder=new_translations.get("system_prompt_placeholder", "You are a pragmatic assistant..."),
                ),
                gr.update(label=new_translations.get("default_model", "Default Model")),
                gr.update(label=new_translations.get("default_preset", "Default Preset")),
                gr.update(value=new_translations.get("save_persona", "Save Persona")),
                gr.update(value=new_translations.get("delete_persona", "Delete Persona")),
                gr.update(value=f"<div class='settings-section-label'>{new_translations.get('settings_workspace', 'Workspace')}</div>"),
                gr.update(label=new_translations.get("prompt_library_accordion", "Prompt Library")),
                gr.update(label=new_translations.get("prompt_label", "Prompt")),
                gr.update(
                    label=new_translations.get("prompt_name", "Prompt Name"),
                    placeholder=new_translations.get("prompt_placeholder", "Create or update a prompt"),
                ),
                gr.update(
                    label=new_translations.get("category", "Category"),
                    placeholder=new_translations.get("category_placeholder", "general / writing / coding ..."),
                ),
                gr.update(
                    label=new_translations.get("prompt_content", "Prompt Content"),
                    placeholder=new_translations.get("prompt_content_placeholder", "Write the reusable prompt here"),
                ),
                gr.update(label=new_translations.get("favorite", "Favorite")),
                gr.update(value=new_translations.get("save_prompt", "Save Prompt")),
                gr.update(value=new_translations.get("delete_prompt", "Delete Prompt")),
                gr.update(value=new_translations.get("insert_prompt", "Insert to Input")),
                gr.update(value=f"<div class='settings-section-label'>{new_translations.get('settings_system', 'System')}</div>"),
                gr.update(label=new_translations.get("advanced_accordion", "Advanced")),
                gr.update(label=new_translations.get("language_label", "Language"), value=selected_language),
                gr.update(
                    label=new_translations.get("temperature", "Temperature"),
                    info=new_translations.get("temperature_info", "Choose between 0 and 1"),
                ),
                gr.update(label=new_translations.get("max_tokens", "Max Tokens")),
                gr.update(label=new_translations.get("top_p", "Top P")),
                gr.update(label=new_translations.get("typical_p", "Typical P")),
                gr.update(label=new_translations.get("num_ctx", "Num CTX")),
                gr.update(value=new_translations.get("save_llm_settings", "Save LLM Settings")),
                gr.update(value=new_translations.get("tool_commands", UI_DEFAULT_TRANSLATIONS["English"]["tool_commands"])),
                gr.update(value=new_translations.get("stop_button", "Stop")),
                gr.update(value=new_translations.get("clean_answer_button", "Clear Answer")),
            )

        language_dropdown.change(
            on_language_change,
            [language_dropdown, web_search_enabled, settings_open_state, search_summary_length_dropdown, session_id_state],
            [
                title_markdown,
                sessions_heading,
                *session_buttons,
                session_search_input,
                rename_session_input,
                pin_session_button,
                new_session_button,
                rename_session_button,
                delete_session_button,
                server_dropdown,
                server_name,
                model_dropdown,
                preset_dropdown,
                persona_dropdown,
                web_search_toggle,
                settings_toggle_button,
                test_llm_connection_button,
                export_chat_button,
                llm_connection_light,
                status_output,
                question_input,
                new_server_name_input,
                new_address_input,
                new_port_input,
                new_default_server,
                add_server_button,
                settings_heading,
                settings_note,
                common_section_label,
                search_accordion,
                search_provider_dropdown,
                search_num_results_slider,
                search_summary_length_dropdown,
                tavily_api_key_input,
                serper_api_key_input,
                save_search_setting_button,
                server_accordion,
                presets_accordion,
                preset_dropdown,
                preset_name_input,
                save_preset_button,
                delete_preset_button,
                personas_accordion,
                persona_dropdown,
                persona_name_input,
                persona_description_input,
                persona_system_prompt_input,
                persona_default_model_input,
                persona_default_preset_dropdown,
                save_persona_button,
                delete_persona_button,
                workspace_section_label,
                prompt_library_accordion,
                prompt_dropdown,
                prompt_name_input,
                prompt_category_input,
                prompt_content_input,
                prompt_favorite_input,
                save_prompt_button,
                delete_prompt_button,
                insert_prompt_button,
                system_section_label,
                advanced_accordion,
                language_dropdown,
                llm_temperature_view,
                llm_max_tokens_view,
                llm_top_p_view,
                llm_typical_p_view,
                llm_num_ctx_view,
                save_llm_setting_button,
                tool_commands_markdown,
                stop_button,
                clean_answer_button,
            ],
        )

    return demo

