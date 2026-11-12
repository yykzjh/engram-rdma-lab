"""独立的 GPU 发请求 / CPU SGE WRITE Engram 实现，不依赖 deep_ep。"""
from .buffer import EngramBuffer

__all__ = ["EngramBuffer"]
