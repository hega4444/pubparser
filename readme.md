# 🎯 Academic Graph Extractor

A sophisticated Python tool that transforms academic HTML documents into structured knowledge graphs. Extract citations, authors, references, and research relationships from scientific papers with ease!

## 🚀 Features

- Extract structured data from academic HTML documents
- Parse complex publication metadata
- Identify author networks and collaborations
- Extract citation graphs and reference networks
- Generate clean JSON output
- Support for multiple academic publisher formats
- Intelligent entity recognition
- Robust error handling

## 📋 Requirements

Install all required dependencies using:

    pip install -r requirements.txt

## 🛠️ Installation

1. Clone the repository:

    git clone https://github.com/hega4444/pubparser
    cd pubparser

2. Install dependencies:

    pip install -r requirements.txt

3. Install spaCy English language model:

    python -m spacy download en_core_web_sm

4. Configure your environment variables:

    cp .env.example .env

## 🎯 Usage

Basic usage example:

    from main import HTMLParser
    from graph import PublicationGraph

    # Initialize parser
    parser = HTMLParser()

    # Parse academic HTML document
    publication_data = parser.parse("path/to/paper.html")

    # Get structured JSON output
    json_output = publication_data.to_json()

    # Analyze citation network
    citations = publication_data.get_citations()
    authors = publication_data.get_authors()

## 🔧 Configuration

You can customize the parser behavior by modifying `config.py`:

    PARSER_CONFIG = {
        "extract_references": True,
        "extract_citations": True,
        "extract_authors": True,
        "detect_institutions": True,
        "output_format": "json"
    }

## 📚 Documentation

The project consists of several key components:

- `parser.py`: HTML parsing and data extraction logic
- `graph.py`: Knowledge graph construction
- `state.py`: Parser state management
- `common.py`: Shared utilities
- `config.py`: Configuration settings

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ✨ Acknowledgments

- Thanks to all contributors
- Inspired by the need for better academic data extraction
- Built with Python 🐍

## 📞 Contact

For any questions or feedback, please open an issue or reach out to the maintainers.

---
Made with ❤️ by hega4444