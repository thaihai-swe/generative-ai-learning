"""Input validation utilities"""


def validate_query(query: str) -> bool:
    """Validate query input"""
    return query and len(query.strip()) > 0


def validate_source(source: str) -> bool:
    """Validate source format"""
    return source and len(source) > 0


def validate_text(text: str, min_length: int = 1) -> bool:
    """Validate text has minimum length"""
    return text and len(text.strip()) >= min_length
