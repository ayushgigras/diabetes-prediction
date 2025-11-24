import subprocess
import time
import os
import sys

def run_app():
    # Get absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(base_dir, 'backend')
    frontend_dir = os.path.join(base_dir, 'frontend')

    print("Starting Diabetes Prediction System...")

    # Start Backend
    print("Starting Backend (Flask)...")
    backend_process = subprocess.Popen(
        [sys.executable, 'app.py'],
        cwd=backend_dir,
        shell=True
    )

    # Start Frontend
    print("Starting Frontend (Vite)...")
    # Use cmd /c to bypass PowerShell restrictions if needed
    frontend_cmd = 'npm run dev'
    if sys.platform == 'win32':
        frontend_cmd = f'cmd /c "{frontend_cmd}"'
        
    frontend_process = subprocess.Popen(
        frontend_cmd,
        cwd=frontend_dir,
        shell=True
    )

    print("\nSystem is running!")
    print("Backend: http://localhost:5000")
    print("Frontend: http://localhost:5173")
    print("\nPress Ctrl+C to stop all services.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
        backend_process.terminate()
        frontend_process.terminate()
        sys.exit(0)

if __name__ == "__main__":
    run_app()
