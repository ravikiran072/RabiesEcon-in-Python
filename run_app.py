#!/usr/bin/env python3
"""
Rabies Economic Analysis - Streamlit App Launcher
==================================================

Launch script for the rabies economic analysis Streamlit application.
Run with: python run_app.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit application."""
    
    # Get the path to the comprehensive app
    app_path = Path(__file__).parent / "app" / "comprehensive_rabies_app.py"
    
    if not app_path.exists():
        print(f"❌ Error: Application file not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Starting Rabies Economic Analysis Dashboard...")
    print(f"📁 App location: {app_path}")
    print("🌐 The app will open in your default browser")
    print("⏹️  Press Ctrl+C to stop the application")
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.headless", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()