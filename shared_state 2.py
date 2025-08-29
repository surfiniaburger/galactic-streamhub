# /Users/surfiniaburger/Desktop/app/shared_state.py
from typing import Any, Dict

# In-memory store for transient data like the latest audio chunk.
transient_data_store: Dict[str, Dict[str, Any]] = {}