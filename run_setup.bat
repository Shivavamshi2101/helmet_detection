@echo off
echo ==========================================
echo Helmet Detection - RTX 4060 Setup
echo ==========================================
py -3.11 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo.
echo IMPORTANT:
echo Install CUDA-enabled PyTorch from https://pytorch.org/get-started/locally/
echo before continuing with requirements.txt
echo.
pip install -r requirements.txt
python check_gpu.py
pause
