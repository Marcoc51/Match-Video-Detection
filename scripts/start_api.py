#!/usr/bin/env python3
"""
Script to start the Match Video Detection API server.
"""

import sys
from pathlib import Path

def start_api():
    """Start the API server."""
    try:
        # Get project root
        project_root = Path(__file__).parent.parent
        
        # Add project root to Python path
        sys.path.insert(0, str(project_root))
        
        # Import configuration
        from src.api.config import get_config, validate_configuration
        
        config = get_config()
        
        print("🚀 Starting Match Video Detection API...")
        print("=" * 50)
        
        # Validate configuration
        print("📋 Validating configuration...")
        if not validate_configuration():
            print("❌ Configuration validation failed!")
            print("Please check your configuration and try again.")
            return False
        
        print("✅ Configuration validated successfully")
        
        # Display configuration
        print(f"📊 Configuration:")
        print(f"   - Host: {config.host}")
        print(f"   - Port: {config.port}")
        print(f"   - Model: {config.model_path}")
        print(f"   - Output: {config.output_dir}")
        print(f"   - Max concurrent jobs: {config.max_concurrent_jobs}")
        
        # Check if model exists
        if config.model_path_absolute.exists():
            print(f"✅ Model found: {config.model_path_absolute}")
        else:
            print(f"⚠️  Model not found: {config.model_path_absolute}")
            print("   The API will start but video processing may fail.")
        
        print("\n🌐 Starting server...")
        print(f"   API will be available at: http://{config.host}:{config.port}")
        print(f"   Documentation: http://{config.host}:{config.port}/docs")
        print(f"   Health check: http://{config.host}:{config.port}/health")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 50)
        
        # Start the server directly instead of using subprocess
        import uvicorn
        uvicorn.run(
            "src.api.main:app",
            host=config.host,
            port=config.port,
            reload=config.reload
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting API: {e}")
        return False
    
    return True

if __name__ == "__main__":
    start_api() 