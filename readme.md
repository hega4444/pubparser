# 💡 PubParser - Publishing Parser

A Python tool that transforms HTML documents into structured data using Google's Gemini LLM, LangGraph, Beautiful Soup, and spaCy. Extract titles, authors, content, and generate summaries with ease! Combines powerful HTML parsing with NLP capabilities for robust document analysis.

## 🚀 Features

- Extract structured data from HTML documents
- Parse document metadata (title, date, author)
- Generate document summaries
- Extract main content and subheadings
- Generate clean JSON output
- Intelligent content analysis using Gemini LLM
- Quality ranking of parsed documents based on extraction confidence
- Robust error handling and validation

## 📋 Requirements

- Python 3.11 (recommended for best compatibility)
- Google API key for Gemini LLM

## 🛠️ Installation

1. Clone the repository:

        git clone https://github.com/hega4444/pubparser
        cd pubparser

2. Create and activate virtual environment:

        python3.11 -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:

        pip install -r requirements.txt

4. Install spaCy English language model:

        python -m spacy download en_core_web_sm

5. Configure your environment:

        cp .env.example .env
        # Add your Google API key to .env:
        # GOOGLE_API_KEY=your_api_key_here

## 🎯 Usage

Place HTML files in the `examples` directory and run:

    python main.py

The parser will process all HTML files and evaluate their parsing quality. When a document's completion rate exceeds the `COMPLETION_THRESHOLD` (default 0.7), the structured JSON result will be saved in the `OUTPUT_DIR` directory. This ensures only high-quality parsing results are preserved.

The project comes with three example HTML files demonstrating different parsing challenges:

- `01_good_article.html`: A well-structured article that's easy to parse
- `02_bad_article.html`: An article with some structural issues
- `03_ugly_article.html`: A challenging article with complex formatting

You can add your own HTML files to the `examples` directory - all files will be processed when running `main.py`.

## 🔧 Configuration

Customize the parser behavior in `.env`:

    GOOGLE_API_KEY=your_api_key_here
    MAX_TITLE_WORDS=10
    COMPLETION_THRESHOLD=0.7
    OUTPUT_DIR=output

## 📚 Project Structure

- `main.py`: Main entry point and document processing
- `graph.py`: LangGraph workflow definition
- `state.py`: Document state management
- `config.py`: Configuration settings
- `examples/`: Directory for HTML files to process
- `output/`: Directory for processed JSON output

View the complete source code on [GitHub](https://github.com/hega4444/pubparser)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ‍💻 Author


Made with ❤️ by [@hega4444](https://github.com/hega4444)