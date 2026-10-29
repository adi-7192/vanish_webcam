# PyInstaller runtime hook. MediaPipe imports matplotlib.pyplot only for a 3D
# plotting helper VanishCam never calls. Bundled matplotlib rebuilds its font
# cache on every launch, delaying startup, so an empty module stands in for it.
import sys
import types

matplotlib = types.ModuleType("matplotlib")
matplotlib.pyplot = types.ModuleType("matplotlib.pyplot")
sys.modules.setdefault("matplotlib", matplotlib)
sys.modules.setdefault("matplotlib.pyplot", matplotlib.pyplot)
