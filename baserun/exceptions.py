from typing import Optional


class BaserunException(Exception):
    detail: str = "Something went wrong"

    def __init__(self, detail: Optional[str] = None) -> None:
        self.detail = detail or self.detail
        super().__init__(self.detail)


class NotInitializedException(BaserunException):
    detail = "Baserun not initialized"
