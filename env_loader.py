"""
Environment configuration loader.
Allows switching between different environment configurations (shadow-x, macroscope, etc.)
"""
import os
from typing import Optional

# Environment variable to control which config to load
ENV_CONFIG_VAR = 'SHADOW_X_ENV'

# Available environments
ENVIRONMENTS = {
    'shadow-x': 'env',
    'macroscope': 'env_macroscope',
}


def get_environment_name() -> str:
    """
    Determine which environment configuration to use.
    
    Priority:
    1. Environment variable SHADOW_X_ENV
    2. Default to 'macroscope' (macroscope-adaptation branch default)
    """
    env_name = os.environ.get(ENV_CONFIG_VAR, 'macroscope')
    if env_name not in ENVIRONMENTS:
        raise ValueError(
            f"Unknown environment '{env_name}'. "
            f"Available options: {list(ENVIRONMENTS.keys())}. "
            f"Set {ENV_CONFIG_VAR} environment variable to switch."
        )
    return env_name


def load_environment_config(env_name: Optional[str] = None):
    """
    Dynamically load the appropriate environment configuration module.
    
    Args:
        env_name: Name of the environment to load. If None, uses get_environment_name()
    
    Returns:
        The loaded environment module
    """
    if env_name is None:
        env_name = get_environment_name()
    
    module_name = ENVIRONMENTS[env_name]
    
    # Import the appropriate environment module
    if env_name == 'shadow-x':
        import env as env_module
    elif env_name == 'macroscope':
        import env_macroscope as env_module
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    return env_module


# Load the active environment configuration
_active_env = None


def get_active_env():
    """Get the currently active environment configuration module."""
    global _active_env
    if _active_env is None:
        _active_env = load_environment_config()
    return _active_env


# Export the configuration values for backward compatibility
def _export_config():
    """Export configuration values from the active environment."""
    env = get_active_env()
    return {
        'BACKGROUND_SCREEN_INDEX': env.BACKGROUND_SCREEN_INDEX,
        'BACKGROUND_SCREEN_BIN_SIZE': env.BACKGROUND_SCREEN_BIN_SIZE,
        'DISPLAY_SCREEN_INDEX': env.DISPLAY_SCREEN_INDEX,
        'OVERHEAD_CAMERA': env.OVERHEAD_CAMERA,
        'BACKEND': env.BACKEND,
        'SCREENS_TO_COORDS': env.SCREENS_TO_COORDS,
    }


# Make configuration values available at module level for easy importing
_config = _export_config()
BACKGROUND_SCREEN_INDEX = _config['BACKGROUND_SCREEN_INDEX']
BACKGROUND_SCREEN_BIN_SIZE = _config['BACKGROUND_SCREEN_BIN_SIZE']
DISPLAY_SCREEN_INDEX = _config['DISPLAY_SCREEN_INDEX']
OVERHEAD_CAMERA = _config['OVERHEAD_CAMERA']
BACKEND = _config['BACKEND']
SCREENS_TO_COORDS = _config['SCREENS_TO_COORDS']

