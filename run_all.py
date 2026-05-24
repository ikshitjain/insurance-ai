import subprocess
import sys
import time
import os

def run_services():
    print("Starting Smart-Insure AI Services...")
    
    # Start FastAPI Backend
    print("🚀 Starting FastAPI Backend on http://localhost:8000")
    api_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"])
    
    # Give API a moment to start
    time.sleep(2)
    
    # Start Streamlit Frontend
    print("🎨 Starting Streamlit Frontend on http://localhost:8501")
    try:
        streamlit_process = subprocess.Popen(["streamlit", "run", "app.py"])
        
        # Keep the script running
        api_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\nStopping services...")
        api_process.terminate()
        streamlit_process.terminate()
    except Exception as e:
        print(f"Error: {e}")
        api_process.terminate()

if __name__ == "__main__":
    run_services()
