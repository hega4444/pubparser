# graph.py

# This file contains the graph definition for the document processing workflow
# It defines the nodes and edges for the workflow, and the state class
# It also contains the LLM initialization and the example usage

import json
import spacy
from typing import List, Literal, Tuple
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from bs4 import BeautifulSoup
from config import (
    GOOGLE_API_KEY,
    MAX_TITLE_WORDS,
    COMPLETION_THRESHOLD,
    ANALYZABLE_PREFIX,
    OUTPUT_DIR,
    LLM_MODEL,
)
from common import (
    get_field_descriptions, 
    validate_title_text, 
    save_document_json,
    append_analysis_status
)
import os
from pathlib import Path
import re

from state import DocState

# Initialize spaCy - this should be at the top level with other imports
nlp = spacy.load("en_core_web_sm")

# Keeping tools structure for future use
tools = []
tool_node = ToolNode(tools)

# Initialize the LLM with API key from environment
llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GOOGLE_API_KEY,
)

# Node functions
def parse_html(state: DocState) -> DocState:
    """Parse the raw HTML content using BeautifulSoup to extract all text elements"""
    state.current_step = "parsing_html"

    try:
        # Input validation
        if not state.raw_html:
            raise ValueError("Empty HTML input")

        if not isinstance(state.raw_html, str):
            raise TypeError("HTML input must be a string")

        if len(state.raw_html) > 10_000_000:  # 10MB limit
            raise ValueError("HTML input too large (>10MB)")

        if "<html" not in state.raw_html.lower():
            raise ValueError("Input does not appear to be HTML")

        soup = BeautifulSoup(state.raw_html, "html.parser")

        # Find all heading elements (h1-h6) and paragraph elements
        relevant_tags = ["h1", "h2", "h3", "h4", "h5", "h6", "p"]

        # Extract text from each element along with its tag name
        for tag in soup.find_all(relevant_tags):
            # Clean the text: remove extra whitespace and newlines
            text = " ".join(tag.get_text().strip().split())
            if text:  # Only add non-empty text
                # Validate text piece size
                if len(text) > 10_000:  # 10K chars limit per piece
                    state.messages.append(
                        {
                            "role": "system",
                            "content": f"Warning: Large text piece truncated ({len(text)} chars)",
                        }
                    )
                    text = text[:10_000]
                state.text_pieces.append((tag.name, text))

        if not state.text_pieces:
            raise ValueError("No text content found in HTML")

        state.processing_status = "processing"

    except Exception as e:
        state.error_message = f"Error parsing HTML: {str(e)}"
        state.processing_status = "error"

    return state


def find_title(state: DocState) -> DocState:
    """
    Extract and set the title from parsed content.
    Looks for text pieces with fewer than MAX_TITLE_WORDS words.
    Takes the first 5 candidates in order of appearance.
    """
    def count_words(text: str) -> int:
        """Helper function to count words in a text string"""
        return len(text.split())

    def get_candidate_titles(
        text_pieces: List[Tuple[str, str]], max_words: int
    ) -> List[Tuple[str, str]]:
        """Get potential title candidates with fewer than max_words words"""
        # Get all candidates that meet the word count criteria
        all_candidates = [
            (tag, text) for tag, text in text_pieces 
            if count_words(text) <= max_words
        ]
        
        # Split into h1 and non-h1 candidates
        h1_candidates = [(tag, text) for tag, text in all_candidates if tag == "h1"]
        other_candidates = [(tag, text) for tag, text in all_candidates if tag != "h1"]
        
        # Combine h1s first, then others, limiting to 5 total
        return (h1_candidates + other_candidates)[:5]

    state.current_step = "finding_title"

    try:
        # Get candidate titles using the nested function
        candidates = get_candidate_titles(state.text_pieces, MAX_TITLE_WORDS)

        if candidates:
            # Take the first candidate as the title
            state.title = candidates[0][1]

            # Store all candidates in messages for potential refinement later
            state.messages.append(
                {
                    "role": "system",
                    "content": f"Found {len(candidates)} title candidates:\n"
                    + "\n".join([f"- ({tag}) {text}" for tag, text in candidates]),
                }
            )
            # Don't add status here - let validate_title handle it
        else:
            state.error_message = "No suitable title candidates found"
            state.processing_status = "error"
            # Don't add status here - let validate_title handle it

    except Exception as e:
        state.error_message = f"Error finding title: {str(e)}"
        state.processing_status = "error"
        # Don't add status here - let validate_title handle it

    return state


def validate_title(state: DocState) -> Literal["refine_title", "analyze_body"]:
    """
    Verify if the extracted title is valid by checking multiple criteria.
    Updates completion rate if validation fails.
    """
    if not state.title:
        state.completion_rate = max(0.0, state.completion_rate - 0.2)
        state.analysis_status = [s for s in state.analysis_status if not s.startswith(("Successfully processed title", "Title needed"))]
        # Just add to messages, don't print
        state.messages.append({
            "role": "system",
            "content": "Title needed refinement (-10% penalty)"
        })
        return "refine_title"

    # Use auxiliary function to check if the title is valid
    is_valid, validation_failures = validate_title_text(state.title)

    if not is_valid:
        # Penalize completion rate for each validation failure
        penalty = min(0.1 * len(validation_failures), 0.3)  # Cap penalty at 0.3
        state.completion_rate = max(0.0, state.completion_rate - penalty)
        
        # Add validation failures to messages
        failure_msg = "Title needed refinement (-10% penalty): " + "; ".join(validation_failures)
        state.messages.append({
            "role": "system",
            "content": failure_msg
        })
        
        # Clear any existing title status
        state.analysis_status = [s for s in state.analysis_status if not s.startswith(("Successfully processed title", "Title needed"))]
        return "refine_title"

    # Clear any existing title status
    state.analysis_status = [s for s in state.analysis_status if not s.startswith(("Successfully processed title", "Title needed"))]
    # Just add to messages, don't print
    state.messages.append({
        "role": "system",
        "content": "Successfully processed title on first attempt"
    })
    return "analyze_body"


def refine_title(state: DocState) -> DocState:
    """
    Improve or fix the title using LLM analysis.
    The completion rate has already been penalized in validate_title.
    """
    state.current_step = "refining_title"

    try:
        # Prepare context from previous steps and text pieces
        first_paragraphs = [text for tag, text in state.text_pieces[:5] if tag == "p"]

        # Create prompt for the LLM
        prompt = {
            "role": "system",
            "content": """You are a title refinement expert. Your task is to analyze 
            the existing title candidates and either:
            1. Select the best candidate if it meets the criteria
            2. Improve one of the candidates if they're close but need adjustment
            3. Create a new title only if none of the candidates are suitable

            A good title should:
            - Be concise (typically 5-10 words)
            - Capture the main topic
            - Use proper capitalization
            - Avoid sentence-like structures (prefer noun phrases)
            - Not end with punctuation
            - Be engaging but not clickbait

            First evaluate the existing candidates before creating something new.""",
        }

        # Add context from previous validation if it exists
        validation_msgs = [
            msg
            for msg in state.messages
            if isinstance(msg, dict) and "Title validation" in msg.get("content", "")
        ]

        # Add context about previous candidates if they exist
        candidate_msgs = [
            msg
            for msg in state.messages
            if isinstance(msg, dict) and "title candidates" in msg.get("content", "")
        ]

        # Prepare the specific request with all context
        context_message = {
            "role": "user",
            "content": f"""Context for title refinement:

            Previous title candidates:
            {candidate_msgs[-1]['content'] if candidate_msgs else 'No previous candidates'}

            Current title: {state.title or 'None'}

            Validation feedback:
            {validation_msgs[-1]['content'] if validation_msgs else 'No validation feedback'}

            First few paragraphs for context:
            {' '.join(first_paragraphs[:2])}

            Please analyze the existing candidates first. If one of them is already good or 
            can be improved with minor changes, use that as a base.
            Only create a completely new title if none of the candidates are suitable.

            Explain your choice briefly, then provide the final title on a new line.""",
        }

        # Create messages list for LLM
        messages = [prompt, context_message]

        # Get LLM response
        response = llm.invoke(messages)

        # Extract title from the last line of the response
        new_title = response.content.strip().split("\n")[-1]

        # Update state
        state.messages.append(
            {
                "role": "system",
                "content": f"Title refinement:\nReasoning: {response.content}\nChosen/Generated title: {new_title}",
            }
        )
        state.title = new_title
        
        # Don't add status here - let validate_title handle it when it's called next

    except Exception as e:
        state.error_message = f"Error refining title: {str(e)}"
        state.processing_status = "error"
        state.completion_rate = max(0.0, state.completion_rate - 0.1)

    return state


def analyze_body(state: DocState) -> DocState:
    """Analyze the document by extracting information for each annotated field."""
    state.current_step = "analyzing_body"

    try:
        # Get only the analyzable fields
        analyzable_fields = get_field_descriptions(DocState, analyzable_only=True)

        for field_name, description in analyzable_fields.items():
            try:
                # Prepare context of what we have so far
                current_state = {
                    k: v
                    for k, v in state.__dict__.items()
                    if k not in ["raw_html", "messages", "text_pieces"]
                    and v is not None
                }

                # Create a field-specific prompt with format guidance
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
                    {' '.join(text for _, text in state.text_pieces[:10])}
                    """
                }

                # Get LLM response
                response = llm.invoke([context_message, user_message])
                response_text = response.content.strip()

                # Handle markdown code blocks
                if response_text.startswith('```'):
                    # Remove the first line (```json) and last line (```)
                    response_text = '\n'.join(response_text.split('\n')[1:-1])

                try:
                    # Parse the JSON response
                    parsed_response = json.loads(response_text)
                    field_value = parsed_response.get(field_name)

                    if field_value:
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
                        
                        # Don't append status here - let validate_json handle it
                    else:
                        state.messages.append({
                            "role": "system",
                            "content": f"No value found for {field_name}"
                        })

                except json.JSONDecodeError:
                    state.messages.append({
                        "role": "system",
                        "content": f"Failed to parse JSON for {field_name}"
                    })
                except ValueError as e:
                    state.messages.append({
                        "role": "system",
                        "content": f"Error processing {field_name}: {str(e)}"
                    })

            except Exception as e:
                state.messages.append({
                    "role": "system",
                    "content": f"Error processing {field_name}: {str(e)}"
                })

    except Exception as e:
        state.error_message = f"Error in analyze_body: {str(e)}"
        state.processing_status = "error"

    return state


def validate_json(state: DocState) -> DocState:
    """
    Validate the processed document and calculate completion rate.
    Takes into account previous penalties from title validation/refinement.
    """
    try:
        # Get total number of analyzable fields
        analyzable_fields = get_field_descriptions(DocState, analyzable_only=True)

        # Find the most recent title status from messages
        title_status_messages = [
            msg for msg in state.messages  
            if isinstance(msg, dict) and 
            any(title_msg in msg.get("content", "") 
                for title_msg in ["Title needed refinement", "Successfully processed title"])
        ]
        
        # Get the latest title status message
        latest_title_msg = title_status_messages[-1]["content"] if title_status_messages else None

        # Reset analysis status and start fresh
        state.analysis_status = []

        # First handle title status
        title_status = None
        if latest_title_msg:
            if "Title needed refinement" in latest_title_msg:
                title_status = "Title needed refinement (-10% penalty)"
                state.completion_rate = 0.9  # Apply penalty
            else:
                title_status = "Successfully processed title on first attempt"
                state.completion_rate = 1.0
        else:
            # If no title status found, check if title exists and is valid
            if state.title:
                title_status = "Successfully processed title on first attempt"
                state.completion_rate = 1.0
            else:
                title_status = "Title needed refinement (-10% penalty)"
                state.completion_rate = 0.9

        # Add title status to analysis_status and print it
        append_analysis_status(state, title_status)

        # Add status for each field
        for field_name in analyzable_fields:
            if field_name != "title":  # Skip title as we handled it above
                field_value = getattr(state, field_name, None)
                if field_value is not None:
                    append_analysis_status(state, f"Successfully processed {field_name}")
                else:
                    append_analysis_status(state, f"Failed to process {field_name}")
                    state.completion_rate = max(0.0, state.completion_rate - 0.1)

        # Set processing status
        state.processing_status = (
            "complete" 
            if state.completion_rate >= COMPLETION_THRESHOLD
            else "incomplete"
        )

        # Add final completion status
        append_analysis_status(
            state,
            f"Parsing completed with {state.completion_rate:.2f} completion rate"
        )

    except Exception as e:
        error_msg = f"Error in validate_json: {str(e)}"
        state.error_message = error_msg
        state.processing_status = "error"
        state.completion_rate = 0.0
        state.messages.append(
            {"role": "system", "content": f"Validation failed with error: {str(e)}"}
        )

    return state


def save_json(state: DocState) -> DocState:
    """Save the final processed state as JSON and log the action."""
    state.current_step = "saving_json"

    success, message = save_document_json(state)
    
    if success:
        state.messages.append({"role": "system", "content": message})
    else:
        state.error_message = "Error saving JSON: " + message
        state.processing_status = "error"
        state.messages.append({"role": "system", "content": message})

    return state


def complete_workflow(state: DocState) -> DocState:
    """Final node to mark workflow completion"""
    state.current_step = "completing"
    return state


def create_conversation_graph():
    """
    Create the conversation graph for the document processing workflow.
    """
    workflow = StateGraph(DocState)

    # Add nodes
    workflow.add_node("parse_html", parse_html)
    workflow.add_node("find_title", find_title)
    workflow.add_node("refine_title", refine_title)
    workflow.add_node("analyze_body", analyze_body)
    workflow.add_node("validate_json", validate_json)
    workflow.add_node("save_json", save_json)
    workflow.add_node("complete", complete_workflow)

    # Add edges
    workflow.add_edge(START, "parse_html")
    workflow.add_edge("parse_html", "find_title")
    workflow.add_conditional_edges("find_title", validate_title)
    workflow.add_edge("refine_title", "analyze_body")
    workflow.add_edge("analyze_body", "validate_json")
    workflow.add_edge("validate_json", "save_json")  # Always save, regardless of completion
    workflow.add_edge("save_json", "complete")

    return workflow.compile()
