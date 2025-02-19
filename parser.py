# parser.py

# This file contains the main function for the document parser
# It is responsible for validating the environment, creating the workflow graph,
# and processing the HTML document

import asyncio
from pathlib import Path
from typing import Optional, Union
from config import validate_env
from graph import create_conversation_graph
from state import DocState

async def process_html_document(
    html_path: Union[str, Path], 
    output_dir: Optional[Union[str, Path]] = None
) -> DocState:
    """
    Process an HTML document through the document analysis workflow.
    
    Args:
        html_path: Path to the HTML file to process
        output_dir: Optional custom output directory for the JSON result
        
    Returns:
        DocState: The final state after processing
        
    Raises:
        FileNotFoundError: If the HTML file doesn't exist
        ValueError: If the HTML file is empty or invalid
    """
    # Convert to Path object
    html_path = Path(html_path)
    
    # Validate input file
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")
        
    # Read HTML content
    html_content = html_path.read_text(encoding="utf-8")
    if not html_content.strip():
        raise ValueError("HTML file is empty")
    
    # Initialize workflow graph
    graph = create_conversation_graph()
    
    # Create initial state
    initial_state = DocState(raw_html=html_content)
    if output_dir:
        initial_state.output_dir = str(Path(output_dir))
    
    # Process document
    final_state = await graph.ainvoke(initial_state)
    return final_state

async def main():
    """Main entry point for the document parser"""
    # Validate environment before starting
    validate_env()
    
    # Process a single document
    try:
        final_state = await process_html_document("article.html")
        print("Processing completed!")
        print("Final state:", final_state.to_json())
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")

def run_parser():
    """Helper function to run the async main function"""
    asyncio.run(main())

if __name__ == "__main__":
    run_parser()
