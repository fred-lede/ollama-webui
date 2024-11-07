import gradio as gr
import requests
import os
import json
import logging
from urllib.parse import urlparse
import re
import time
import threading
from PIL import Image, ImageTk
import base64

css = """
.my-button {
    width: 100px !important;
}
"""

# 設定日誌紀錄
logging.basicConfig(
    filename='log-webui.log',
    filemode='a',
    #level=logging.DEBUG,  		# 使用 DEBUG 級別以獲取詳細日誌
    level=logging.INFO,  		# 使用 INFO 級別以獲取詳細日誌
    #level=logging.WARNING,		# 使用 WARNING 級別以獲取詳細日誌
    #level=logging.ERROR,		# 使用 ERROR 級別以獲取詳細日誌
    #level=logging.CRITICAL,	# 使用 CRITICAL 級別以獲取詳細日誌
    format='%(asctime)s:%(levelname)s:%(message)s',
    encoding='utf-8'  # 確保這裡指定 utf-8
)

CONFIG_FILE = "server_settings.json"

LANGUAGE_FILE = "language_settings.json"

# 載入語言設定
def load_language_settings():
    if not os.path.exists(LANGUAGE_FILE):
        raise FileNotFoundError(f"{LANGUAGE_FILE} not found.")
    with open(LANGUAGE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# 設定語言
language_settings = load_language_settings()
current_language = language_settings.get("default_language", "en")
translations = language_settings["languages"].get(current_language, {})

# 更新語言
def update_language(selected_language):
    global current_language, translations
    current_language = selected_language
    translations = language_settings["languages"].get(selected_language, {})
    return translations

# 新增伺服器地址驗證函數
def is_valid_url(url):
    # 簡單的正則表達式驗證 URL
    regex = re.compile(
        r'^(http:\/\/|https:\/\/)'
        r'(\d{1,3}\.){3}\d{1,3}'
        r'(:\d+)?$'
    )
    return re.match(regex, url) is not None

# 載入伺服器設定
def load_settings():
    if not os.path.exists(CONFIG_FILE):
        logging.info(f"{CONFIG_FILE} 不存在。返回空列表。")
        return [], None  # 返回空的主機列表和 None 作為預設
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            hosts = config.get("hosts", [])
            default_host = next((host for host in hosts if host.get("default")), None)
            llm_parameters = config.get("llm_parameters", [{}])[0]
            logging.info(f"從設定檔載入了 {len(hosts)} 個 Ollama 主機。")
            logging.info(f"從設定檔載入了 {len(llm_parameters)} 個模型參數設定。")
            logging.debug(f"載入設定: {config}")
            return hosts, default_host, llm_parameters, config
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"載入設定時出錯: {e}")
        return [], None, "", ""

# 儲存伺服器設定
def save_settings(hosts):
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        config = {"hosts": []}

    config["hosts"] = hosts

    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)  # 添加縮排以提高可讀性
        logging.info(f"已保存設定: {config}")
    except IOError as e:
        logging.error(f"儲存設定時出錯: {e}")

# 從伺服器獲取可用模型
def fetch_models(server_address, server_port):
    try:
        url = f"{server_address}:{server_port}/v1/models"
        logging.info(f"請求模型列表: {url}")
        response = requests.get(url)
        response.raise_for_status()
        models_data = response.json().get("data", [])
        models = [model["id"] for model in models_data if "id" in model]
        logging.info(f"從 {server_address}:{server_port} 獲取的模型: {models}")
        return models
    except requests.RequestException as e:
        logging.warning(f"從 {server_address}:{server_port} 獲取模型失敗，請檢查Ollama服務主機是否開啟或變更server_settings.json預設主機: {e}")
        return []
    except json.JSONDecodeError as e:
        logging.error(f"JSON 解碼失敗: {e}")
        return []

# 處理保存設定（添加新伺服器）
def handle_add_server(new_server_name, new_address, new_port, new_default_server):
    logging.info(f"新增伺服器: {new_server_name}, {new_address}, {new_port}, {new_default_server}")
    
    # 驗證 URL 格式
    if not is_valid_url(new_address):
        logging.error(f"無效的伺服器地址: {new_address}")
        return "無效的伺服器地址。請使用正確的格式，例如 http://192.168.1.173", gr.update(), gr.update(), "", ""
    
    hosts, default_host, _, config = load_settings()
    
    # 檢查新的地址和端口是否已存在於 hosts
    for host in hosts:
        if host['address'] == new_address and host['port'] == new_port:
            logging.info("主機已存在。")
            return "主機已存在。", "主機已存在。", gr.update(), gr.update(), "", ""
    
    # 如果不存在，則將新主機添加到列表中
    hosts.append({"server_name": new_server_name, "address": new_address, "port": new_port, "default": new_default_server})
    
    # 將新的主機設置為預設，並取消其他主機的預設
    if new_default_server:
        for host in hosts:
            host['default'] = (host['address'] == new_address and host['port'] == new_port)
        
    save_settings(hosts)
    
    
    # 更新伺服器選擇下拉選單
    server_choices = [(host['server_name'], f"{host['address']}:{host['port']}") for host in hosts]
    current_host = f"{new_address}:{new_port}"
    
    # 從新保存的主機獲取模型
    models = fetch_models(new_address, new_port)
    
    if models:
        status = "設定已成功保存，已獲取模型、更新伺服器選擇下拉選單、更新模型下拉選單。"
        status_tab_setting = "設定已成功保存，已獲取模型、更新伺服器選擇下拉選單、更新模型下拉選單。"
    else:
        status = "設定已保存，但未獲取到模型，請確認新建主機是否提供LLM服務。"
        status_tab_setting = "設定已保存，但未獲取到模型，請確認新建主機是否提供Ollama服務。"
    
    logging.info(f"handle_add_server 回傳: {status}")
        
    return (
        status, status_tab_setting,
        gr.update(choices=server_choices, value=current_host),
        gr.update(value=new_server_name),  # 更新 server_name 文本框
        "", "",                            # 讓new_server_name, new_address的value清空
    )

# 處理伺服器選擇變更
def handle_server_change(selected_server):
    logging.info(f"選擇伺服器: {selected_server}")
    if not selected_server:
        return "未選擇伺服器。", gr.update(choices=[], value=None), ""
    try:
        parsed_url = urlparse(selected_server)
        address = f"{parsed_url.scheme}://{parsed_url.hostname}"
        port = parsed_url.port
        if port is None:
            port = 11434  # default port if not specified
        models = fetch_models(address, port)
        logging.debug(f"handle_server_change --> address: {address}, port: {port}, models: {models}")
        if models:
            status = "已獲取模型並更新模型下拉選單。"
            status_tab_setting = "已獲取模型並更新模型下拉選單。"
        else:
            status = "未獲取到模型，請確認主機是否提供Ollama服務。"
            status_tab_setting = "未獲取到模型，請確認主機是否提供Ollama服務。"
        logging.info(f"handle_server_change 回傳: {status}")
        logging.debug(f"handle_server_change 回傳的值:\nstatus: {status}\nmodel: {gr.update(choices=models, value=models[0] if models else None)}\nserver: {selected_server}")
        return status, status_tab_setting, gr.update(choices=models, value=models[0] if models else None), selected_server
    except Exception as e:
        logging.error(f"處理伺服器選擇變更時出錯: {e}")
        return "處理伺服器選擇變更時出錯。", gr.update(choices=[], value=None), ""

# 全局停止事件
stop_event = threading.Event()

def generate_prompt(question, history, image=None):
    prompt = "Previous conversation:\n" if history else ""
    for msg in history:
        prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    prompt += f"\nUser: {question}\n"

    if image:
        return {"prompt": prompt, "images": [encode_image(image)]}
    return {"prompt": prompt}
    
def encode_image(self, image):
    buffered = io.BytesIO()
    image = image.convert("RGB")  # Convert image to RGB format
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# 問答功能（流式）
def ask_question_stream(question, history, model, selected_server, llm_temperature, llm_max_tokens, llm_top_p, llm_typical_p, llm_num_ctx):
    logging.debug(
        f"ask_question_stream()傳入的值：\n"
        f"question：{question}\n"
        f"history：{history}\n"
        f"model：{model}\n"
        f"selected_server：{selected_server}\n"
        f"llm_temperature：{llm_temperature}\n"
        f"llm_max_tokens：{llm_max_tokens}\n"
        f"llm_top_p：{llm_top_p}\n"
        f"llm_typical_p：{llm_typical_p}\n"
        f"llm_num_ctx：{llm_num_ctx}\n"
    )
    stop_event.clear()

    prompt = "Previous conversation:\n" if history else ""
    for msg in history:
        prompt += f"{msg['role'].capitalize()}: {msg['content']}\n"
    prompt += f"\nUser: {question.get('text')}\n"
    logging.debug(f"處理prompt履歷: {prompt}")

    new_messages = history.copy()
    new_messages.append({"role": "user", "content": question.get('text')})

    assistant_message = {"role": "assistant", "content": ""}
    new_messages.append(assistant_message)

    if model is None:
        assistant_message['content'] = "請選擇模型。"
        yield from new_messages  # Yield the entire new_messages for the initial state
        return

    if selected_server is None:
        assistant_message['content'] = "請選擇伺服器。"
        yield from new_messages  # Yield the entire new_messages for the initial state
        return

    try:
        parsed_url = urlparse(selected_server)
        address = f"{parsed_url.scheme}://{parsed_url.hostname}"
        port = parsed_url.port if parsed_url.port else 11434
        url = f"{address}:{port}/api/chat"
        headers = {"Content-Type": "application/json"}

        logging.info(f"發送請求到: {url}，問題: {question.get('text')}, 模型: {model}")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {
                "temperature": llm_temperature,     # 調整這個參數來控制答案的隨機性
                "num_predict": llm_max_tokens,      # 增加這個參數來獲得更長的答案
                "top_p": llm_top_p,                 # top_p(核取樣): 控制生成的多樣性。設置為 1 表示使用最高概率的標記，較低的值會使生成的文字更加多樣化。範圍通常是0.5到1.0。
                "typical_p": llm_typical_p,         # 控制生成過程中的典型性，較低值會讓生成內容更符合預期。
                "num_ctx": llm_num_ctx              # 設定單次生成可以使用的最大上下文長度。
            }
        }
        
        
        if len(question.get('files', [])) > 0:
            file_path = question['files'][0]

            try:
                with open(file_path, 'rb') as img_file:
                    encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                    payload["messages"][0]["images"] = [encoded_image]
                    logging.debug(f"確認payload是否含image: {payload}")
            except Exception as e:
                logging.error(f"處理圖片文件時發生錯誤: {e}")
        else:
            logging.info("沒有上傳圖片文件")

        response = requests.post(url, headers=headers, json=payload, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if stop_event.is_set():
                logging.info("收到停止信號，停止回應。")
                break

            if line:
                try:
                    data = json.loads(line.decode('utf-8'))
                    content = data.get("message", {}).get("content", "")
                    if content:
                        assistant_message['content'] += content
                        new_messages[-1]['content'] = assistant_message['content']
                        # Yield only the latest message
                        yield new_messages[-1]  # Yield the new assistant message
                        time.sleep(0.1)  # Simulate delay
                    else:
                        logging.info("收到助手回答完成。")
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logging.error(f"解析回應時出錯: {e}")
                    assistant_message['content'] = "解析回應時出錯。"
                    yield new_messages[-1]  # Yield the new assistant message

        logging.info(f"助手回答: {new_messages[-1]['content']}")

    except requests.RequestException as e:
        logging.error(f"向 {selected_server} 發送請求失敗: {e}")
        assistant_message['content'] = "請求失敗。"
        yield new_messages[-1]  # Yield the new assistant message
    except Exception as e:
        logging.error(f"處理回應時出錯: {e}")
        assistant_message['content'] = "處理回應時出錯。"
        yield new_messages[-1]  # Yield the new assistant message

   

# 處理停止回應
def stop_response():
    logging.debug("停止回應功能被觸發。")
    stop_event.set()
    return gr.update(interactive=True)  # 重新啟用 question_input
    
# 處理prompt傳送回應
def send_prompt(prompt):
    logging.debug("Prompt傳送功能被觸發。")
    
    return prompt

#在tab_chatbot顯示主機名稱
def extract_address(address):
    # 將地址解析為基本URL部分（去掉端口和協議部分）
    return address.split(":")[0] if "://" not in address else address.split("://")[1].split(":")[0]

def find_server_name_by_address(hosts, address):
    address_base = extract_address(address)
    for host in hosts:
        host_address_base = extract_address(host['address'])
        if host_address_base == address_base:
            return host['server_name']
    return "無此主機"

def update_textbox_server(selected_server):
    hosts, _, _, _ = load_settings()
    server_name = find_server_name_by_address(hosts, selected_server)
    logging.debug(f"選擇的伺服器地址: {selected_server}, 對應的伺服器名稱: {server_name}")
    return server_name
     
def get_model_info(server_name, model_name):
    #print(f"get_model_info-{server_name}--{model_name}")
    url=server_name+"/api/show"
    data = json.dumps({"name": model_name})
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            model_info = response.json()
            #return model_info['details']
            return model_info['model_info']
            
        else:
            logging.error(f"獲取模型信息失敗，狀態碼: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"向 {server_name} 獲取模型信息失敗: {e}")
        return None
        
# 格式化模型信息為逐行顯示
def format_model_info(model_info):
    if model_info:
        return "\n".join([f"{key}: {value}" for key, value in model_info.items()])
    else:
        return "無法獲取模型信息"
        

# 載入初始伺服器設定
hosts, default_host, llm_parameters, _ = load_settings()
server_addresses = [host['address'] for host in hosts]
server_choices = [(host['server_name'], f"{host['address']}:{host['port']}") for host in hosts]
current_host = f"{default_host['address']}:{default_host['port']}" if default_host else None
current_server_name = f"{default_host['server_name']}" if default_host else None
models = fetch_models(default_host['address'], default_host['port']) if default_host else []
logging.debug(f"初始伺服器列表: {server_choices}, 初始模型列表: {models}")

# 提取大模型參數設定
llm_temperature = llm_parameters.get('llm_temperature', None)
llm_max_tokens = llm_parameters.get('llm_max_tokens', None)
llm_top_p = llm_parameters.get('llm_top_p', None)
llm_typical_p = llm_parameters.get('llm_typical_p', None)
llm_num_ctx = llm_parameters.get('llm_num_ctx', None)
logging.debug(f"初始大模型參數設定-->temperature:{llm_temperature}, max_tokens: {llm_max_tokens}, top_p: {llm_top_p}, typical_p: {llm_typical_p}, num_ctx: {llm_num_ctx}")

# 獲取模型信息
model_info = get_model_info(current_host, models[0] if models else None)

def clear_chatbot():
    return gr.update(value=[])
    
# 定義 clean_answer_button，這是為了移動 clean_answer_button 到 chatbot 之前並保持運作正常
clean_answer_button = gr.ClearButton(None, value=translations.get("clean_answer_button", "Clear Answer"), elem_classes="my-button")
# 定義 chatbot 並將按鈕與 chatbot 綁定
chatbot = gr.Chatbot(
        type="messages",
        show_copy_all_button=True,
        show_copy_button=True,
        height=400,
        autoscroll=True
        )

# 定義 Gradio 介面
with gr.Blocks(css=css, theme=gr.themes.Default()) as demo:
    gr.Markdown("# Ollama WebUI")
    with gr.Tabs(selected="Chatbot") as tabs:  # 指定程式啟動時的默認標籤頁
        #定義setting標籤頁
        tab_setting = gr.TabItem(label=translations.get("tab_setting", "Settings"), id="Settings")
        with tab_setting:
            with gr.Row():
                with gr.Column():
                        # 語言選單
                        language_dropdown = gr.Dropdown(
                            choices=list(language_settings["languages"].keys()),
                            label="Language",
                            value=current_language,
                            interactive=True
                        )
                        tab_status_output = gr.Textbox(
                                label=translations.get("status_display", "Status Display"),
                                show_label=True,
                                placeholder=translations.get("status_display_msg", "Displaying the model's status or displaying an error message."),
                                lines=1
                            )  
                        model_info = gr.Textbox(
                                label=translations.get("model_info", "Model Info"),
                                show_label=True,
                                lines=25,
                                show_copy_button=True,
                                autoscroll=True,
                                value=format_model_info(model_info)
                            )               
                
                with gr.Column():
                    # 伺服器選擇下拉選單
                    server_dropdown = gr.Dropdown(
                            choices=server_choices,
                            allow_custom_value=True,
                            label=translations.get("select_server", "Select Server"),
                            value=current_host
                        )            
                        
                    server_name = gr.Textbox(
                            label=translations.get("server_address", "Server Address"),
                            value=current_host,
                            interactive=False,
                            visible=True
                        )
                         
                    #模型下拉選單
                    model_dropdown = gr.Dropdown(
                            choices=models,
                            allow_custom_value=True,
                            label=translations.get("select_model", "Select Model"),
                            value=models[0] if models else None,
                        )
                    llm_temperature = gr.Slider(0, 2, step=0.1, value=llm_temperature, label="Temperature", info="Select a value between 0 and 2", interactive=True, visible=True)
                    llm_max_tokens = gr.Number(label="Max Tokens", interactive=True, visible=True, value=llm_max_tokens)
                    llm_top_p = gr.Slider(0, 1, step=0.1, value=llm_top_p, label="top_p", info="Select a value with a typical range of 0.5 to 1.0.", interactive=True, visible=True)
                    llm_typical_p = gr.Slider(0, 1, step=0.1, value=llm_typical_p, label="typical_p", info="Select a value with a range of 0 to 1.0.", interactive=True, visible=True)
                    llm_num_ctx = gr.Number(label="Num CTX", interactive=True, visible=True, value=llm_num_ctx)

                                        
                with gr.Column():
                    # 新增伺服器輸入
                    new_server_name_input = gr.Textbox(label=translations.get("ai_server_name", "Ai Server Name"),
                                                       placeholder=translations.get("ai_server_name_msg", "Ex: GMK-K9, editable via manual editing, file name for settings: server_settings.json."))
                    new_address_input = gr.Textbox(label=translations.get("new_server_address", "New Server Address"),
                                                   placeholder=translations.get("new_server_address_msg", "Ex. http://127.0.0.1"))
                    new_port_input = gr.Number(label=translations.get("new_server_port", "Port"), value=11434)
                    new_default_server = gr.Checkbox(label=translations.get("default_start_setting", "Make the new host by default."), value=False)
                                    
                    add_server_button = gr.Button(value=translations.get("add_ai_server", "Add Ai Server"))        
        
        #定義chatbot標籤頁
        tab_chatbot = gr.TabItem(label=translations.get("tab_chatbot", "Chatbot"), id="Chatbot")
        with tab_chatbot:
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                current_server = gr.Textbox(
                                        label=translations.get("current_server", "Current Server"),
                                        show_label=True,
                                        value=current_server_name,
                                        lines=1
                                    )
                            with gr.Column():                
                                current_model = gr.Textbox(
                                        label=translations.get("current_model", "Current Model"),
                                        show_label=True,
                                        value=model_dropdown.value,
                                        lines=1
                                    )
                        with gr.Row():                
                                status_output = gr.Textbox(
                                        label=translations.get("status_display", "Status Display"),
                                        show_label=True,
                                        placeholder=translations.get("status_display_msg", "Displaying the model's status or displaying an error message."),
                                        lines=1    
                                    )
                        with gr.Row():
                            stop_button = gr.Button(value=translations.get("stop_button", "Stop Answer"), elem_classes="my-button")
                            clean_answer_button = gr.Button(value=translations.get("clean_answer_button", "Clear Answer"), elem_classes="my-button") 

                    with gr.Column():
                        with gr.Row():
                            prompt_temp_area = gr.Textbox(
                                    label="Prompt workspace",
                                    show_label=False,
                                    lines=7.75,
                                    autofocus=True,
                                    show_copy_button=True,
                                    interactive=True,
                                    placeholder=translations.get("prompt_temp_area", "Temporary workspace for the prompt.")
                                )
                                
                        with gr.Row():
                            clean_button = gr.ClearButton(prompt_temp_area, value=translations.get("clean_button", "Clean"))
                            prompt_button = gr.Button(value=translations.get("prompt_button", "Send Prompt"))
                    
                with gr.Row():
                    chatbot.render()  
                with gr.Row():
                    with gr.Column(scale=3):
                        question_input = gr.MultimodalTextbox(
                                interactive=True,
                                autoscroll=True,
                                file_count="single",
                                file_types=['.jpg', '.jpeg', '.png', '.bmp'],
                                placeholder=translations.get("enter_question","Enter your question or upload image file with question..."),
                                show_label=False
                            )
                
                        answer_output = gr.ChatInterface(
                                fn=ask_question_stream,
                                type="messages",
                                multimodal = True,
                                additional_inputs=[model_dropdown, server_dropdown, llm_temperature, llm_max_tokens, llm_top_p, llm_typical_p, llm_num_ctx],
                                fill_width = True,
                                fill_height = True,
                                chatbot = chatbot,
                                autofocus = False,
                                show_progress = "full",
                                textbox = question_input
                            )
                        
    # 綁定新增伺服器按鈕事件
    add_server_button.click(
        handle_add_server,
        inputs=[new_server_name_input, new_address_input, new_port_input, new_default_server],
        outputs=[status_output, tab_status_output, server_dropdown, model_dropdown, new_server_name_input, new_address_input]
    )
    
    # 綁定伺服器選擇下拉選單變更事件
    server_dropdown.change(
        handle_server_change,
        inputs=[server_dropdown],
        outputs=[status_output, tab_status_output, model_dropdown, server_name]
    )
    
    # 綁定主機、模型選擇下拉選單變更事件
    server_dropdown.change(fn=update_textbox_server, inputs=server_dropdown, outputs=current_server)
    
    def update_textbox_model(server_dropdown, selected_option):
        model_info = get_model_info(server_dropdown, selected_option)
        model_info = format_model_info(model_info)
        return selected_option, model_info
    model_dropdown.change(fn=update_textbox_model, inputs=[server_dropdown, model_dropdown], outputs=[current_model, model_info])
        
    
    # 綁定停止按鈕事件
    stop_button.click(
        fn=stop_response,
        inputs=[],
        outputs=[question_input],  
        queue=False  # 停止操作不需要排隊
    )
    
    # 綁定清除按鈕事件
    clean_answer_button.click(fn=clear_chatbot, outputs=chatbot)
    
    # 綁定傳送按鈕事件
    prompt_button.click(
        fn=send_prompt,
        inputs=[prompt_temp_area],
        outputs=[question_input],  # 將更新映射到 question_input
        queue=False  # 停止操作不需要排隊
    )
    
        
    # 當語言選單改變時，更新顯示文本
    def on_language_change(selected_language):
        new_translations = update_language(selected_language)
        return (
            gr.update(label=new_translations.get("select_server", "Select Server")),
            gr.update(label=new_translations.get("server_address", "Server Address")),
            gr.update(label=new_translations.get("select_model", "Select Model")),
            gr.update(label=new_translations.get("status_display", "Status Display"),
                      placeholder=new_translations.get("status_display_msg", "Displaying the model's status or displaying an error message.")),
            gr.update(label=new_translations.get("status_display", "Status Display"),
                      placeholder=new_translations.get("status_display_msg", "Displaying the model's status or displaying an error message.")),
            gr.update(label=new_translations.get("enter_question", "Enter Your Question"),
                      placeholder=translations.get("enter_question", "Enter your question or upload image file with question...")),
            gr.update(label=translations.get("ai_server_name", "Ai Server Name"),
                      placeholder=translations.get("ai_server_name_msg", "Ex: GMK-K9, editable via manual editing, file name for settings: server_settings.json.")),
            gr.update(label=translations.get("new_server_address", "New Server Address"),
                      placeholder=translations.get("new_server_address_msg", "Ex. http://127.0.0.1")),
            gr.update(label=new_translations.get("new_server_port", "Port")),
            gr.update(label=new_translations.get("default_start_setting", "Make the new host by default.")),
            gr.update(value=new_translations.get("add_ai_server", "Add Ai Server")),
            gr.update(value=new_translations.get("clean_button", "Clean")),
            gr.update(value=new_translations.get("stop_button", "Stop")),
            gr.update(value=new_translations.get("clean_answer_button", "Clear Answer")),
            gr.update(value=new_translations.get("prompt_button", "Send Prompt")),
            gr.update(placeholder=new_translations.get("prompt_temp_area", "Temporary workspace for the prompt.")),
            gr.update(value=selected_language),
            gr.update(label=translations.get("current_server", "Current Server")),
            gr.update(label=translations.get("current_model", "Current Model")),
            gr.update(label=translations.get("tab_setting", "Settings")),
            gr.update(label=translations.get("tab_chatbot", "Chatbot")),
            gr.update(label=translations.get("model_info", "Model Info"))
        )
    
    # 設定語言選擇事件
    language_dropdown.change(
            on_language_change, [language_dropdown], 
            [server_dropdown, server_name, model_dropdown, status_output, tab_status_output, question_input,
            new_server_name_input, new_address_input, new_port_input, new_default_server,
            add_server_button, clean_button, stop_button, clean_answer_button, prompt_button,
            prompt_temp_area, language_dropdown, current_server, current_model, tab_setting,
            tab_chatbot, model_info]
        )
    
if __name__ == "__main__":
    demo.launch(show_error=True, inbrowser=True)
