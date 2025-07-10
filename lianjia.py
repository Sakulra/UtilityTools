import requests
from bs4 import BeautifulSoup
import re
import json

# 目标网页 URL
url = "https://sh.lianjia.com/chengjiao/pg3"

# 发送请求获取网页内容
response = requests.get(url)
html_content = response.text

# 使用 BeautifulSoup 解析 HTML
soup = BeautifulSoup(html_content, "html.parser")

# 查找所有 <script> 标签
scripts = soup.find_all("script")

# 遍历 <script> 标签，查找目标 JSON 数据
for script in scripts:
    if "window.__PRELOADED_STATE__" in script.text:
        # 使用正则表达式提取 JSON 数据
        match = re.search(r"window\.__PRELOADED_STATE__\s*=\s*({.*?});", script.text)
        if match:
            json_data = match.group(1)
            # 将 JSON 字符串转换为 Python 字典
            data = json.loads(json_data)
            print(json.dumps(data, indent=2, ensure_ascii=False))
            break