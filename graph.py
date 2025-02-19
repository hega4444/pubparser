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
    validate_env,
)
from common import get_field_descriptions
import os

from state import DocState

# Initialize spaCy - this should be at the top level with other imports
nlp = spacy.load("en_core_web_sm")

# Keeping tools structure for future use
tools = []
tool_node = ToolNode(tools)

# Initialize the LLM with API key from environment
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
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
        candidates = [
            (tag, text) for tag, text in text_pieces if count_words(text) <= max_words
        ]
        return candidates[:5]

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
        else:
            state.error_message = "No suitable title candidates found"
            state.processing_status = "error"

    except Exception as e:
        state.error_message = f"Error finding title: {str(e)}"
        state.processing_status = "error"

    return state


def validate_title(state: DocState) -> Literal["refine_title", "analyze_body"]:
    """
    Verify if the extracted title is valid by checking if it has more nouns/adjectives than verbs.
    This helps identify phrase-like titles vs sentence-like text.
    """
    if not state.title:
        return "refine_title"

    try:
        doc = nlp(state.title)

        # Count nouns, proper nouns, and adjectives
        content_words = sum(
            1 for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"]
        )
        # Count verbs
        verbs = sum(1 for token in doc if token.pos_ == "VERB")

        # Add validation details to messages for potential refinement
        state.messages.append(
            {
                "role": "system",
                "content": (
                    f"Title validation:\n"
                    f"- Content words (nouns/adjectives): {content_words}\n"
                    f"- Verbs: {verbs}\n"
                    f"- Title: {state.title}"
                ),
            }
        )

        # If we have more content words than verbs, it's likely a good title
        if content_words > verbs:
            return "analyze_body"

    except Exception as e:
        state.messages.append(
            {"role": "system", "content": f"Title validation failed: {str(e)}"}
        )

    # If anything fails or title isn't good enough, go to refinement
    return "refine_title"


def refine_title(state: DocState) -> DocState:
    """
    Improve or fix the title using LLM analysis of the text pieces and previous candidates.
    Uses context from validation and previous title attempts to generate a better title.
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

    except Exception as e:
        state.error_message = f"Error refining title: {str(e)}"
        state.processing_status = "error"

    return state


def analyze_body(state: DocState) -> DocState:
    """
    Analyze the document by extracting information for each annotated field.
    Uses LLM to process each field based on its annotation description.
    """
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

                # Simulate RAG queries using the current state and LLM (TODO: Implement RAG)
                context_message = {
                    "role": "user",
                    "content": f"""
                    Current document state:
                    {json.dumps(current_state, default=str, indent=2)}
                    
                    Task: {description.replace(ANALYZABLE_PREFIX, '')}
                    
                    Available text:
                    {' '.join(text for _, text in state.text_pieces[:10])}  # First 10 pieces for context
                    """,
                }

                # Get LLM response
                response = llm.invoke([context_message])

                # Update state with response
                if field_name == "date":
                    # Parse date string to datetime
                    setattr(
                        state,
                        field_name,
                        datetime.fromisoformat(response.content.strip()),
                    )
                else:
                    setattr(state, field_name, response.content.strip())

                state.analysis_status.append(f"Successfully processed {field_name}")

            except Exception as e:
                state.analysis_status.append(f"Error processing {field_name}: {str(e)}")

    except Exception as e:
        state.error_message = f"Error in analyze_body: {str(e)}"
        state.processing_status = "error"

    return state


def validate_json(state: DocState) -> Literal["save_json", "__end__"]:
    """
    Validate the processed document and calculate completion rate.
    Returns save_json if completion rate meets threshold, otherwise ends the graph with a detailed report.
    """
    try:
        # Get total number of analyzable fields
        analyzable_fields = get_field_descriptions(DocState, analyzable_only=True)
        total_fields = len(analyzable_fields)

        # Analyze each field's status
        field_status = []
        successful_fields = 0
        error_count = 0

        # Check each analyzable field
        for field_name in analyzable_fields:
            field_value = getattr(state, field_name)
            success_msg = next(
                (
                    msg
                    for msg in state.analysis_status
                    if f"Successfully processed {field_name}" in msg
                ),
                None,
            )
            error_msg = next(
                (
                    msg
                    for msg in state.analysis_status
                    if f"Error processing {field_name}" in msg
                ),
                None,
            )

            if success_msg:
                successful_fields += 1
                status = "✓ Success"
            elif error_msg:
                error_count += 1
                status = f"✗ Error: {error_msg}"
            else:
                error_count += 1
                status = "✗ Not processed"

            field_status.append(f"{field_name}: {status}")

        # Title quality factor
        title_quality = 1.0 if "refine_title" not in state.current_step else 0.8

        # Calculate completion rate
        state.completion_rate = (
            (successful_fields / total_fields) * 0.6
            + title_quality * 0.3  # Weight for successful fields
            + (1 - error_count / total_fields)  # Weight for title quality
            * 0.1  # Weight for error ratio
        )

        # Prepare validation report
        validation_report = [
            "📊 Document Processing Validation Report",
            "=====================================",
            f"✨ Completion Rate: {state.completion_rate:.2f} (Threshold: {COMPLETION_THRESHOLD})",
            f"📝 Title Quality: {'✅ Good' if title_quality == 1.0 else '⚠️ Needed Refinement'}",
            f"📈 Successful Fields: {successful_fields}/{total_fields}",
            f"❌ Errors: {error_count}",
            "",
            "🔍 Field Status:",
            "------------",
        ] + field_status

        # Add report to messages
        state.messages.append(
            {"role": "system", "content": "\n".join(validation_report)}
        )

        # Determine next step
        if state.completion_rate >= COMPLETION_THRESHOLD:
            return "save_json"
        else:
            state.error_message = (
                f"Document failed validation. See processing report for details.\n"
                f"Completion rate {state.completion_rate:.2f} below threshold {COMPLETION_THRESHOLD}"
            )
            state.processing_status = "error"
            return END or "__end__"

    except Exception as e:
        state.error_message = f"Error in validate_json: {str(e)}"
        state.processing_status = "error"
        state.messages.append(
            {"role": "system", "content": f"Validation failed with error: {str(e)}"}
        )
        return END or "__end__"


def save_json(state: DocState) -> DocState:
    """
    Save the final processed state as JSON and log the action.
    Uses DocState's to_json method to serialize the state.
    """
    state.current_step = "saving_json"

    try:
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Get JSON representation using DocState's method
        json_output = state.to_json()

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(OUTPUT_DIR, f"processed_doc_{timestamp}.json")

        # Save to file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json_output)

        # Log success
        state.messages.append(
            {"role": "system", "content": f"💾 Document successfully saved to {filename}"}
        )
        state.processing_status = "completed"

    except Exception as e:
        state.error_message = f"Error saving JSON: {str(e)}"
        state.processing_status = "error"
        state.messages.append(
            {"role": "system", "content": f"❌ Failed to save document: {str(e)}"}
        )

    return state


def create_conversation_graph():
    """
    Create the conversation graph for the document processing workflow.
    
    The graph defines the following processing flow:
    1. Starts with parsing raw HTML to extract text elements
    2. Attempts to find a title from the parsed content
    3. If title needs refinement, goes to refine_title step, otherwise proceeds
    4. Analyzes the document body for key information
    5. Validates the extracted JSON data
    6. If validation passes, saves the JSON output
    7. Ends processing with either success or error status
    
    The graph handles errors at each step by updating the DocState with 
    error messages and status. Conditional edges allow for branching based
    on validation results.
    """
    workflow = StateGraph(DocState)

    # Add nodes
    workflow.add_node("parse_html", parse_html)
    workflow.add_node("find_title", find_title)
    workflow.add_node("refine_title", refine_title)
    workflow.add_node("analyze_body", analyze_body)
    workflow.add_node("validate_json", validate_json)
    workflow.add_node("save_json", save_json)

    # Add edges
    workflow.add_edge(START, "parse_html")
    workflow.add_edge("parse_html", "find_title")
    workflow.add_conditional_edges("find_title", validate_title)
    workflow.add_edge("refine_title", "analyze_body")
    workflow.add_edge("analyze_body", "validate_json")

    # Add conditional edge for validation
    workflow.add_conditional_edges("validate_json", validate_json)

    return workflow.compile()
