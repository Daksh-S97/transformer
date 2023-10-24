FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

EXPOSE 80

COPY config.py config.py
COPY model.py model.py
COPY requirements.txt requirements.txt
COPY weights/tf_29.pt weights/tf_29.pt
COPY dataset.py dataset.py
COPY train.py train.py

COPY tokenizer_en.json tokenizer_en.json
COPY tokenizer_sv.json tokenizer_sv.json

RUN pip install -r requirements.txt

CMD ["python3 train.py"]