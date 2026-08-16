class LLMTwinException(Exception):
    pass


class ImproperlyConfigured(LLMTwinException):
    pass


class LLMGenerationError(LLMTwinException):
    pass
