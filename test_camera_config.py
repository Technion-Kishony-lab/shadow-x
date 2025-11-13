"""Test camera configuration"""
import os
from env_loader import get_active_env, get_environment_name

print("=" * 60)
print("Camera Configuration Test")
print("=" * 60)

# Check environment
env_name = get_environment_name()
print(f"\n1. Active environment: {env_name}")

# Check environment variable
shadow_x_env = os.environ.get('SHADOW_X_ENV', 'not set')
print(f"2. SHADOW_X_ENV variable: {shadow_x_env}")

# Check USE_DIGICAM flag
try:
    env = get_active_env()
    use_digicam = getattr(env, 'USE_DIGICAM', False)
    print(f"3. USE_DIGICAM flag: {use_digicam}")
    
    if use_digicam:
        print("   -> Will use digiCamControl")
    else:
        print("   -> Will use OpenCV VideoCapture")
except Exception as e:
    print(f"3. Error checking USE_DIGICAM: {e}")

# Test digiCamControl import
print("\n4. Testing digiCamControl adapter:")
try:
    from resources.camera_digicam import get_or_create_digicam_camera
    print("   [OK] DigiCamCamera adapter imported successfully")
except ImportError as e:
    print(f"   [FAIL] Cannot import DigiCamCamera: {e}")

print("\n" + "=" * 60)

