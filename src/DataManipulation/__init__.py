# __init__.py (Root Level)

from .temporal_graphs import *  # Make temporal_graphs available at package level

# Import submodules so they can be accessed directly
from .socio_patterns import *

print("Package initialized: Available submodules -> ipc, mimic, sfhh, temporal_graphs")
