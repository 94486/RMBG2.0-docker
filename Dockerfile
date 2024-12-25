FROM quay.io/fedora/python-311

WORKDIR /app

COPY requirements.txt .
RUN pip install gradio_imageslider  -i https://pypi.mirrors.ustc.edu.cn/simple --verbose
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --verbose

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
