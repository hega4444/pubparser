from typing import get_type_hints, Dict
from config import ANALYZABLE_PREFIX
from datetime import datetime
import os
import json
import re
import spacy
from pathlib import Path
from typing import List, Tuple, Literal
from config import OUTPUT_DIR

# Initialize spaCy
nlp = spacy.load("en_core_web_sm")

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

def save_document_json(state) -> Tuple[bool, str]:
    """
    Save the document state as JSON and return success status and message.
    """
    try:
        # Set creation timestamp
        state.created_at = datetime.now()

        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Get original filename without extension
        original_name = Path(state.raw_html_path).stem if hasattr(state, 'raw_html_path') else "doc"
        
        # Generate filename with timestamp
        timestamp = state.created_at.strftime("%y%m%d_%H%M%S")
        filename = os.path.join(OUTPUT_DIR, f"doc_{original_name}_{timestamp}.json")

        # Get JSON representation
        json_output = state.to_json()

        # Save to file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json_output)

        return True, f"💾 Document successfully saved to {filename}"

    except Exception as e:
        return False, f"❌ Failed to save document: {str(e)}"

def validate_title_text(title: str) -> Tuple[bool, List[str]]:
    """
    Validate a title string against various criteria.
    Returns (is_valid, list_of_validation_failures)
    """
    validation_failures = []
    
    try:
        doc = nlp(title)
        
        # 1. Basic length checks
        word_count = len(doc)
        if word_count < 2 or word_count > 30:
            validation_failures.append(f"Word count {word_count} outside acceptable range (2-30)")

        # 2. Check for date-like patterns
        date_patterns = [
            r'\d{4}',           # Year
            r'\d{1,2}/\d{1,2}', # MM/DD
            r'January|February|March|April|May|June|July|August|September|October|November|December'
        ]
        if any(re.search(pattern, title, re.IGNORECASE) for pattern in date_patterns):
            validation_failures.append("Contains date-like patterns")

        # 3. POS analysis
        pos_counts = {
            "nouns": sum(1 for token in doc if token.pos_ in ["NOUN", "PROPN"]),
            "verbs": sum(1 for token in doc if token.pos_ == "VERB"),
            "adj": sum(1 for token in doc if token.pos_ == "ADJ"),
            "punct": sum(1 for token in doc if token.pos_ == "PUNCT")
        }

        if pos_counts["nouns"] == 0:
            validation_failures.append("No nouns found")

        verb_ratio = pos_counts["verbs"] / word_count
        if verb_ratio > 0.3:
            validation_failures.append(f"Too many verbs (ratio: {verb_ratio:.2f})")

        # 4. Check for common title patterns
        first_token = doc[0]
        last_token = doc[-1]

        if first_token.pos_ in ["DET", "ADP"]:
            validation_failures.append(f"Starts with {first_token.pos_}")

        if last_token.text not in ["?", "!"]:
            if last_token.pos_ == "PUNCT":
                validation_failures.append("Ends with punctuation")

        return len(validation_failures) == 0, validation_failures

    except Exception as e:
        return False, [f"Validation error: {str(e)}"] 

def append_analysis_status(state, message: str) -> None:
    """
    Append a message to the analysis_status list and print it to terminal.
    Handles both success and error/warning messages with different formatting.
    """
    # Only print if this is a new message (not already in analysis_status)
    if message not in state.analysis_status:
        # Format the message based on its content
        if any(error_term in message.lower() for error_term in ["error", "failed", "warning", "penalty"]):
            print(f"\033[91m❌ {message}\033[0m")  # Red text for errors/warnings
        elif "success" in message.lower():
            print(f"\033[92m✓ {message}\033[0m")  # Green text for success
        else:
            print(f"\033[94mℹ {message}\033[0m")  # Blue text for info
    
    # Always append to the state
    state.analysis_status.append(message) 