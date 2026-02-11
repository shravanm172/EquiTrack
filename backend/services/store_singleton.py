# services/store_singleton.py
from services.analysis_store import AnalysisStore

# 30 min TTL, adjust as needed for dev
analysis_store = AnalysisStore(ttl_seconds=1800, max_items=5000)