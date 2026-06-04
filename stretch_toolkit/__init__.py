"""
Stretch Toolkit - Unified interface for physical and simulated robot control.

Environment auto-detection:
- If stretch_body is available → Physical robot mode
- If USE_SIM=1 environment variable → Simulation mode
- Otherwise → Simulation mode (default for dev environments)

Usage:
    from stretch_toolkit import controller, teleop
    
    while True:
        velocities = teleop.get_normalized_velocities()
        controller.set_velocities(velocities)
"""
import os
import sys

# Determine which backend to use
USE_PHYSICAL = False
BACKEND_NAME = "simulation"

# Check for explicit simulation flag
if os.getenv('USE_SIM', '0') == '1':
    USE_PHYSICAL = False
    BACKEND_NAME = "simulation"
else:
    # Try to import stretch_body to detect physical robot
    try:
        import stretch_body.robot
        USE_PHYSICAL = True
        BACKEND_NAME = "physical"
    except ImportError:
        USE_PHYSICAL = False
        BACKEND_NAME = "simulation"

print(f"[stretch_toolkit] Loading {BACKEND_NAME} backend")

# Import base classes (always available)
from .base import TeleopProvider, JointController, merge_proportional

# Import state control
from .state_control import StateController
if USE_PHYSICAL:
    try:
        from .physical import (
            PhysicalJointController, 
            HEAD_CAMERA, WRIST_CAMERA, NAVIGATION_CAMERA,
            HEAD_RGB_CAMERA, HEAD_DEPTH_CAMERA,
            WRIST_RGB_CAMERA, WRIST_DEPTH_CAMERA
        )
        import stretch_body.robot as rb
        
        # Create robot instance
        robot = rb.Robot()
        robot.startup()
        robot.enable_collision_mgmt()
        
        # Create controller
        controller = PhysicalJointController(robot=robot)
        teleop = TeleopProvider(is_stretch_env=True)
        
        print("[stretch_toolkit] Physical robot initialized")
    except Exception as e:
        print(f"[stretch_toolkit] ERROR: Failed to initialize physical robot: {e}")
        print("[stretch_toolkit] Falling back to simulation mode")
        USE_PHYSICAL = False

if not USE_PHYSICAL:
    from .sim import (
        SimulatedJointController, 
        HEAD_CAMERA, WRIST_CAMERA, NAVIGATION_CAMERA,
        HEAD_RGB_CAMERA, HEAD_DEPTH_CAMERA,
        WRIST_RGB_CAMERA, WRIST_DEPTH_CAMERA
    )
    from stretch_mujoco import StretchMujocoSimulator
    from stretch_mujoco.enums.stretch_cameras import StretchCameras
    
    # Lazy initialization - only create sim when first accessed
    _sim = None
    _controller = None
    
    def _load_robocasa_config():
        """Load robocasa configuration from JSON file if it exists."""
        import json
        from pathlib import Path
        
        config_path = Path(__file__).parent / "robocasa_config.json"
        
        if not config_path.exists():
            return None
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config if config.get('enabled', False) else None
        except Exception as e:
            print(f"[stretch_toolkit] Warning: Failed to load robocasa config: {e}")
            return None
    
    def _get_controller():
        global _sim, _controller
        if _controller is None:
            # Check for robocasa environment configuration
            robocasa_config = _load_robocasa_config()
            
            # Prepare simulator initialization kwargs
            camera_hz = robocasa_config.get('camera_hz', 30) if robocasa_config else 30
            sim_kwargs = {'cameras_to_use': [], 'camera_hz': camera_hz}
            
            if robocasa_config:
                # Generate robocasa model from config parameters
                try:
                    from stretch_mujoco.robocasa_gen import model_generation_wizard
                    
                    task = robocasa_config.get('task', 'PnPCounterToCab')
                    layout = robocasa_config.get('layout', 0)
                    style = robocasa_config.get('style', 0)
                    
                    print(f"[stretch_toolkit] Loading RoboCasa environment: {task} (layout={layout}, style={style})")
                    model, xml, objects_info = model_generation_wizard(
                        task=task,
                        layout=layout,
                        style=style,
                    )
                    sim_kwargs['model'] = model  # Pass generated model to simulator
                    
                except Exception as e:
                    print(f"[stretch_toolkit] Warning: Failed to load RoboCasa scene: {e}")
                    print("[stretch_toolkit] Falling back to default environment")
            
            # Initialize simulator (with or without robocasa model)
            headless = robocasa_config.get('headless', False) if robocasa_config else False
            _sim = StretchMujocoSimulator(**sim_kwargs)
            _sim.start(headless=headless)
            _controller = SimulatedJointController(sim=_sim)
            
            # Start camera watchdog thread
            from . import sim
            sim._start_watchdog()
            
            if robocasa_config:
                print("[stretch_toolkit] RoboCasa environment initialized")
            else:
                print("[stretch_toolkit] MuJoCo simulation initialized (default environment)")
        return _controller
    
    # Create a proxy object that initializes on first use
    class _ControllerProxy:
        def __getattr__(self, name):
            return getattr(_get_controller(), name)
    
    controller = _ControllerProxy()
    teleop = TeleopProvider(is_stretch_env=False)
    
    print("[stretch_toolkit] Simulation mode ready (lazy init)")

# Export public API
__all__ = [
    'controller',
    'teleop',
    'TeleopProvider',
    'JointController',
    'StateController',
    'merge_proportional',
    'USE_PHYSICAL',
    'BACKEND_NAME',
    'HEAD_CAMERA',
    'WRIST_CAMERA',
    'NAVIGATION_CAMERA',
    'HEAD_RGB_CAMERA',
    'HEAD_DEPTH_CAMERA',
    'WRIST_RGB_CAMERA',
    'WRIST_DEPTH_CAMERA',
]
