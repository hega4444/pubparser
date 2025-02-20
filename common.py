# common.py

# This file contains auxiliary functions used throughout the project.

from typing import get_type_hints, Dict
from config import ANALYZABLE_PREFIX
from datetime import datetime
import os
import json
import re
import spacy
from pathlib import Path
from typing import List, Tuple
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

def update_analysis_status(state, message: str, print_to_terminal: bool = True) -> None:
    """
    Append a message to the analysis_status list and print it to terminal.
    Handles both success and error/warning messages with different formatting.
    """
    # Only print if this is a new message (not already in analysis_status)

    # Format the message based on its content
    message_format = (
        "\033[91m❌" if any(error_term in message.lower() for error_term in ["error", "failed", "warning", "penalty"]) else
        "\033[92m✓" if "success" in message.lower() else
        "\033[94mℹ"
    )
    if print_to_terminal:
        print(f"{message_format} {message}\033[0m")
    
    # Add to messages for the LLM, may be used for debugging
    state.messages.append({
            "role": "system",
            "content": message
        })
        
    # Always append to the state
    state.analysis_status.append(message) 

def prepare_field_prompt(field_name: str, description: str, text_pieces: list) -> tuple[dict, dict]:
    """
    Prepare field-specific prompt messages for document analysis.
    
    Args:
        field_name: Name of the field to analyze
        description: Description of what to extract (with ANALYZABLE_PREFIX removed)
        text_pieces: List of (tag, text) tuples containing document content
        
    Returns:
        tuple[dict, dict]: A tuple of (context_message, user_message) dictionaries
    """
    context_message = {
        "role": "system",
        "content": f"""
        You are a document analysis assistant. Extract ONLY the {field_name} from the text.
        
        {
        '''For subheadings, extract the full heading including any descriptive text after dashes.
        Return as a JSON array of strings.
        If a heading contains a dash, keep the entire text.''' if field_name == 'subheadings' else
        f'Return in this exact JSON format: {{"{field_name}": "your extracted value here"}}'
        }
        
        Example format for subheadings:
        {{
            "subheadings": [
                "Abstract - A Brief Overview",
                "Introduction - Main Concepts",
                "Methods - Experimental Setup",
                "Results - Key Findings"
            ]
        }}
        
        Task: {description.replace(ANALYZABLE_PREFIX, '')}
        """
    }

    user_message = {
        "role": "user",
        "content": f"""
        Text to analyze:
        {' '.join(text for _, text in text_pieces[:10])}
        """
    }

    return context_message, user_message 

def parse_and_update_field(state, field_name: str, response_text: str) -> tuple[bool, str]:
    """
    Parse LLM response and update state with the extracted field value.
    
    Args:
        state: The DocState instance to update
        field_name: Name of the field being processed
        response_text: Raw response text from LLM
        
    Returns:
        tuple[bool, str]: Success status and message
    """
    try:
        # Handle markdown code blocks
        if response_text.startswith('```'):
            # Remove the first line (```json) and last line (```)
            response_text = '\n'.join(response_text.split('\n')[1:-1])

        # Parse the JSON response
        parsed_response = json.loads(response_text)
        field_value = parsed_response.get(field_name)

        if not field_value:
            return False, f"No value found for {field_name}"

        # Process field value based on field type
        if field_name == "date":
            # Handle date parsing
            if len(field_value) == 7:  # Format like "2025-02"
                state.date = datetime.strptime(field_value, "%Y-%m").replace(day=1)
            else:
                state.date = datetime.fromisoformat(field_value)
        elif field_name == "subheadings":
            # Ensure subheadings is always a list
            if isinstance(field_value, str):
                # Convert comma-separated string to list, preserving dash content
                subheadings = [h.strip() for h in field_value.split(',')]
            elif isinstance(field_value, list):
                subheadings = field_value
            else:
                raise ValueError(f"Unexpected subheadings format: {type(field_value)}")
            
            # Clean up each subheading while preserving descriptive text
            cleaned_subheadings = []
            for heading in subheadings:
                # Remove any extra whitespace but keep the dash and description
                cleaned = ' '.join(heading.split())
                if cleaned:
                    cleaned_subheadings.append(cleaned)
            
            setattr(state, field_name, cleaned_subheadings)
        else:
            # Handle other fields
            setattr(state, field_name, field_value)

        return True, f"Successfully processed {field_name}"

    except json.JSONDecodeError:
        return False, f"Failed to parse JSON for {field_name}"
    except ValueError as e:
        return False, f"Error processing {field_name}: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error processing {field_name}: {str(e)}" 