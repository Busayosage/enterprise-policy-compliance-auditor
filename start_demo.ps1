$env:CHROMA_DB_PATH = "chroma_db_demo"
$env:CHROMA_TEMP_PATH = "temp_demo"
.\venv311\Scripts\python.exe -m streamlit run app.py --server.port 8501
