from typing import List, Optional, Annotated, TypeAlias, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from langchain_core.messages import BaseMessage
from config import ANALYZABLE_PREFIX
import json

# Define field types with annotations, used to help the
# LLM understand the context of the workflow
# elements with ANALYZABLE_PREFIX are used to help the LLM understand the context of the workflow
# and the fields that it can use to generate the output
RawHtmlType: TypeAlias = Annotated[
    str, "Raw HTML content of the article to be processed"
]
MessagesType: TypeAlias = Annotated[
    List[BaseMessage], "List of messages for LLM interaction history"
]
TitleType: TypeAlias = Annotated[
    Optional[str],
    "Main title of the article, should be concise and representative of the content",
]
DateType: TypeAlias = Annotated[
    Optional[datetime], 
    f"{ANALYZABLE_PREFIX} Find a single publication date in ISO format. Look for dates in headers, bylines, or metadata."
]
AuthorType: TypeAlias = Annotated[
    Optional[str],
    f"{ANALYZABLE_PREFIX} Extract author name(s). Look for bylines or author sections. Multiple authors should be comma-separated."
]
SubheadingsType: TypeAlias = Annotated[
    List[str],
    f"{ANALYZABLE_PREFIX} Find all section headings in order of appearance. Include only clear section dividers or topic markers."
]
BodyType: TypeAlias = Annotated[
    Optional[str],
    f"{ANALYZABLE_PREFIX} Extract the main article content as clean text. Remove any navigation, ads, or irrelevant elements."
]
SummaryType: TypeAlias = Annotated[
    Optional[str], 
    f"{ANALYZABLE_PREFIX} Create a concise summary of the main points. If an existing summary exists, use that instead."
]
StepType: TypeAlias = Annotated[
    str,
    "Current processing step in the workflow: init, parsing_html, finding_title, etc.",
]
ErrorType: TypeAlias = Annotated[
    Optional[str], "Error message if any step fails during processing"
]
StatusType: TypeAlias = Annotated[
    str, "Current status of document processing: pending, processing, completed, error"
]
TextPieceType: TypeAlias = Annotated[
    List[Tuple[str, str]],
    "List of tuples containing (tag_name, text_content) extracted from HTML elements",
]
CreatedAtType: TypeAlias = Annotated[
    Optional[datetime], 
    "Timestamp when the document was processed"
]


@dataclass
class DocState:
    """
    State for the document processing workflow
    contains the declaration of the fields that will be used in the workflow
    which can be shared with the LLM to help it understand the context of the workflow
    and the fields that it can use to generate the output
    """

    raw_html: RawHtmlType
    messages: MessagesType = field(default_factory=list)
    text_pieces: TextPieceType = None

    # Document fields
    title: TitleType = None
    date: DateType = None
    author: AuthorType = None
    subheadings: SubheadingsType = None
    body: BodyType = None
    summary: SummaryType = None

    # Processing status fields
    current_step: StepType = "init"
    error_message: ErrorType = None
    processing_status: StatusType = "pending"

    # Analysis status fields
    analysis_status: List[str] = field(default_factory=list)
    completion_rate: float = 0.0  # Initialize here only

    # Additional fields
    created_at: CreatedAtType = None
    raw_html_path: Optional[str] = None

    def __init__(
        self,
        raw_html: str,
        messages: Optional[List[BaseMessage]] = None,
        text_pieces: Optional[List[Tuple[str, str]]] = None,
        title: Optional[str] = None,
        date: Optional[datetime] = None,
        author: Optional[str] = None,
        subheadings: Optional[List[str]] = None,
        body: Optional[str] = None,
        summary: Optional[str] = None,
        current_step: str = "init",
        error_message: Optional[str] = None,
        processing_status: str = "pending",
        analysis_status: Optional[List[str]] = None,
        completion_rate: float = 0.0,
        created_at: Optional[datetime] = None,
        raw_html_path: Optional[str] = None,
    ):
        """Initialize a new DocState instance."""
        self.raw_html = raw_html
        self.messages = messages or []
        self.text_pieces = text_pieces or []
        self.title = title
        self.date = date
        self.author = author
        self.subheadings = subheadings or []
        self.body = body
        self.summary = summary
        self.current_step = current_step
        self.error_message = error_message
        self.processing_status = processing_status
        self.analysis_status = analysis_status or []
        self.completion_rate = completion_rate  # Keep this one
        self.created_at = created_at
        self.raw_html_path = raw_html_path

    def to_json(self) -> str:
        """Convert the document state to JSON format"""
        return json.dumps(
            {
                "title": self.title,
                "date": self.date.isoformat() if self.date else None,
                "author": self.author,
                "subheadings": self.subheadings,
                "body": self.body,
                "summary": self.summary,
                "processing_status": self.processing_status,
                "error_message": self.error_message,
                "completion_rate": self.completion_rate,
                "analysis_status": self.analysis_status,
                "created_at": self.created_at.isoformat() if self.created_at else None
            },
            indent=2,
        ) 