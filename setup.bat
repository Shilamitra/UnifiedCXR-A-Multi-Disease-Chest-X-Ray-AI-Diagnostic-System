@echo off
echo Setting up Medical Scan Project...

echo Creating virtual environment...
python -m venv mediscan_env

echo Activating environment...
call mediscan_env\Scripts\activate

echo Installing dependencies...
pip install flask==2.3.3
pip install flask-cors==4.0.0
pip install tensorflow==2.16.2
pip install numpy==1.26.4
pip install pillow==10.0.0
pip install opencv-python-headless==4.10.0.84
pip install python-dotenv==1.0.0

echo Creating models folder...
mkdir models 2>nul

echo.
echo ✅ Setup completed!
echo.
echo To run the application:
echo 1. Activate environment: mediscan_env\Scripts\activate
echo 2. Run: python app.py
echo.
pause