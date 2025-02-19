from typing import get_type_hints, Dict
from config import ANALYZABLE_PREFIX

def get_field_descriptions(doc_state_class: any, analyzable_only: bool = False) -> Dict[str, str]:
    """
    Extract field descriptions from DocState annotations.
    
    Args:
        doc_state_class: The DocState class to analyze
        analyzable_only: If True, only return fields with ANALYZABLE_PREFIX
    """
    hints = get_type_hints(doc_state_class, include_extras=True)
    descriptions = {}

    for field_name, type_hint in hints.items():
        if hasattr(type_hint, "__metadata__"):
            description = type_hint.__metadata__[0]
            if not analyzable_only or (
                analyzable_only and description.startswith(ANALYZABLE_PREFIX)
            ):
                descriptions[field_name] = description

    return descriptions 