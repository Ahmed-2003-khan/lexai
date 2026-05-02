import re
from datetime import datetime, timedelta
from langchain.tools import tool

@tool
def legal_calculator(expression: str) -> str:
    """
    Calculate dates, deadlines, limitation periods, and other numeric legal computations.
    Provide natural language expressions like '90 days from 2024-01-15'.
    """
    expression_lower = expression.lower()
    
    # Regex to capture patterns for adding time to a specific base date
    add_match = re.search(r'(\d+)\s+(days|years)\s+from\s+(\d{4}-\d{2}-\d{2})', expression_lower)
    
    if add_match:
        amount = int(add_match.group(1))
        unit = add_match.group(2)
        base_date_str = add_match.group(3)
        
        try:
            base_date = datetime.strptime(base_date_str, "%Y-%m-%d")
            
            if unit == "days":
                target_date = base_date + timedelta(days=amount)
            elif unit == "years":
                # Calculate leap-year safe year addition
                try:
                    target_date = base_date.replace(year=base_date.year + amount)
                except ValueError:
                    target_date = base_date + timedelta(days=365 * amount)
                    
            return f"Calculated Deadline: {target_date.strftime('%Y-%m-%d')}"
        except Exception as e:
            return f"Error computing date: {str(e)}"
            
    return "Could not parse computation. Please use a format like 'X days from YYYY-MM-DD'."