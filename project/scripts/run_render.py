import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import the rendering function
from render import render_sac_hockey

# Define the checkpoint path
checkpoint_path = (
    "C:/Users/killa/Documents/GitHub/LaserHockeyProject/results/checkpoint_1000.pt"
)

# Call the rendering function
render_sac_hockey(checkpoint=checkpoint_path, deterministic=True, strong_opponent=False)
