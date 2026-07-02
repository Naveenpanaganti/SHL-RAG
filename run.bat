@echo off
REM Use the shl-rag conda environment to run the server
C:\Users\navee\.conda\envs\shl-rag\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
