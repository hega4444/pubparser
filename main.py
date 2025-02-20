# main.py

# This file contains the main function for the document parser
# It is responsible for validating the environment, creating the workflow graph,
# and processing the HTML document

import os
os.environ["GRPC_PYTHON_LOG_LEVEL"] = "error"
os.environ["GRPC_VERBOSITY"] = "ERROR"

import logging
logging.getLogger().setLevel(logging.ERROR)

import asyncio
from pathlib import Path
from typing import Optional, Union
from config import validate_env
from graph import create_conversation_graph
from state import DocState
from absl import logging as absl_logging
import json
from langgraph.graph import START, END

# Suppress all warnings
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('grpc').setLevel(logging.ERROR)
logging.getLogger('google.generativeai').setLevel(logging.ERROR)

async def process_html_document(
    html_path: Union[str, Path], 
    output_dir: Optional[Union[str, Path]] = None
) -> DocState:
    """Process an HTML document through the document analysis workflow."""
    
    # Initial setup
    html_path_obj = Path(html_path)
    print("🔄 Starting document processing...")
    
    # Validate input file
    if not html_path_obj.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path_obj}")
        
    html_content = html_path_obj.read_text(encoding="utf-8")
    if not html_content.strip():
        raise ValueError("HTML file is empty")
    
    print("📄 HTML file loaded successfully")
    print("🔄 Parsing into JSON...")
    
    # Initialize workflow
    graph = create_conversation_graph()
    initial_state = DocState(raw_html=html_content)
    initial_state.raw_html_path = str(html_path)
    if output_dir:
        initial_state.output_dir = str(Path(output_dir))

    # Process document and track progress
    final_state = await graph.ainvoke(initial_state)
    
    # Print progress logs
    if hasattr(final_state, 'messages'):
        for msg in final_state.messages:
            if isinstance(msg, dict):
                content = msg.get('content', '')
                if content.startswith('📊 Document Processing Validation Report'):
                    print(f"\n{content}")
                elif '✓' in content or '❌' in content:
                    print(content)

    # Convert final state back to DocState if needed
    if not isinstance(final_state, DocState):
        final_state_obj = DocState(raw_html=html_content)
        final_state_obj.raw_html_path = str(html_path)
        for key, value in dict(final_state).items():
            if hasattr(final_state_obj, key):
                setattr(final_state_obj, key, value)
        final_state = final_state_obj
    
    return final_state

async def main():
    """Main entry point for the document parser"""
    validate_env()
    
    # Get all HTML files from examples directory
    examples_dir = Path("examples")
    html_files = list(examples_dir.glob("*.html"))
    
    if not html_files:
        print("❌ No HTML files found in examples directory")
        return
        
    print(f"📁 Found {len(html_files)} HTML files to process")
    
    # Process each file
    for html_file in html_files:
        print(f"\n{'='*50}")
        print(f"📄 Processing {html_file.name}")
        print('='*50)
        
        try:
            final_state = await process_html_document(html_file)
            print("\n✅ Processing completed!")
            
            # Print final summary
            print("\n📄 Final State Summary:")
            summary = {
                "Title": final_state.title or "No title",
                "Status": final_state.processing_status,
                "Completion Rate": f"{final_state.completion_rate:.2f}",
                "Analysis Status": [
                    msg for msg in final_state.analysis_status
                    if msg.startswith(("Successfully", "Error", "Could not"))
                ]
            }
            print(json.dumps(summary, indent=2))
            
        except Exception as e:
            print(f"❌ Error processing {html_file.name}: {str(e)}")
            continue  # Move to next file
    
    print("\n✨ All files processed!")

def run_parser():
    """Helper function to run the async main function"""
    asyncio.run(main())

if __name__ == "__main__":
    run_parser()
