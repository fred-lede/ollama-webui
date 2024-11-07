
# 使用Gradio在Windows與本地或遠端的Ollama進行聊天

中文 | [English](./README_en.md)

## 功能

- 介面語言有中文、英文及泰文，用戶也可以自行增加或減少，有需要的可修改language_settings.json檔案，但務必維持內容格式。
- 本地或遠端的Ollama服務會存放在server_settings.json檔案，在使用者介面是以選單選取，程式會自動抓取該主機Ollama可提供的模型，用戶依需求選擇使用，但此程式僅支持語言模型。
- 在設定的頁面可以增加Ollama的服務主機，但目前不提供編輯及修改，請直接編輯server_settings.json檔案。
- 設定頁面也提供Temperature, Num Predict(Max Tokens), Top_p, Typical_p, Num CTX模型參數設定。
- 虛擬助手頁面為聊天介面，如果選擇的模型提供Vision功能，則可上傳圖面來問答。

## 環境要求
- Windows 10/11
- Python 3.10.11(我開發的環境)
- Gradio 5.1.0(我使用的版本)

## 安裝和運行

### 可直接下載封裝好的，解壓縮後即可使用
https://github.com/fred-lede/ollama-webui/releases

### 使用介面
![1](https://github.com/user-attachments/assets/e45f43d0-0767-4713-a0ae-c6bc673d751e)

![2](https://github.com/user-attachments/assets/311d588d-45b5-48ff-92a2-1facad868dc4)

### 採用源碼的
1. `pip install virtualenv`
2. `virtualenv -p python3.10.11 myenv`
3. `myenv\scripts\activate`
4. 克隆或下載本項目代碼 https://github.com/fred-lede/ollama-webui.git 
5. 安裝依賴庫 `pip install -r requirements.txt`
6. 運行程序 `python ollama-webui.py`

## 貢獻者

- Fred

## 許可證

本項目採用 [MIT License](LICENSE) 進行許可。
