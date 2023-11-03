FROM python
WORKDIR /app
COPY . /app/
RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 8501

#python3 -m streamlit run app.py
CMD [ "python3","-m", "streamlit", "run", "app.py" ]