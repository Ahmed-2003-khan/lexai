from typing import Optional
from langchain_core.tools import tool

@tool
def format_legal_citation(title: str, source: str, doc_type: str, year: Optional[str] = None) -> str:
    """
    Formats a given legal source into proper citation styles.
    """
    # Default to current year or placeholder if missing
    display_year = year if year else "[Year]"
    
    if doc_type == "case_law":
        pakistani_style = f"{title} [{display_year}] {source}"
        bluebook_style = f"{title}, {source} ({display_year})"
    elif doc_type == "statute":
        pakistani_style = f"The {title}, {display_year} ({source})"
        bluebook_style = f"{title} act, {source} ({display_year})"
    else:
        pakistani_style = f"{title} ({source})"
        bluebook_style = f"{title}, {source}"

    output = (
        f"Pakistani Citation Style: {pakistani_style}\n"
        f"Bluebook Citation Style: {bluebook_style}"
    )
    return output