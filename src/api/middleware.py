import time
import logging
from starlette.requests import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("uvicorn")

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
