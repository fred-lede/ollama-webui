
# Chatting with Local or Remote Ollama on Windows Using Gradio

[Chinese](./README.md) | English

## Features

- The interface languages include Chinese, English, and Thai. Users can also add or remove them as needed. If necessary, the language_settings.json file can be modified, but it is essential to maintain the content format.
- The local or remote Ollama services are stored in the server_settings.json file. They are selected via a menu in the user interface. The program will automatically retrieve the models that Ollama on the host can provide, and users can choose to use them according to their needs. However, this program only supports language models.
- The Ollama service host can be added on the settings page, but currently, editing and modification are not available. Please directly edit the server_settings.json file.
- The settings page also provides the setting of model parameters such as Temperature, Num Predict (Max Tokens), Top_p, Typical_p, and Num CTX.
- The virtual assistant page is a chatting interface. If the selected model provides the Vision function, images can be uploaded for Q&A.

## Environment Requirements
- Windows 10/11
- Python 3.10.11 (the development environment I used)
- Gradio 5.1.0 (the version I used)

## Installation and Running

### You can directly download the packaged version and use it after decompression
https://github.com/fred-lede/ollama-webui/releases

### Using the Source Code
1. `pip install virtualenv`
2. `virtualenv -p python3.10.11 myenv`
3. `myenv\scripts\activate`
4. Clone or download the code of this project: https://github.com/fred-lede/ollama-webui.git
5. Install the dependent libraries: `pip install -r requirements.txt`
6. Run the program: `python ollama-webui.py`

## Contributors

- Fred

## License

This project is licensed under the [MIT License](LICENSE).
