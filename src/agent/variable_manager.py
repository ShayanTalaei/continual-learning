"""Variable Manager for compact memory storage.

Based on CUGA's implementation - stores execution outputs as variables
and provides compact summaries instead of full values.
"""
import json
from typing import Any, Dict, Optional
from datetime import datetime


class VariableMetadata:
    """Metadata wrapper for stored variables."""
    
    def __init__(self, value: Any, description: Optional[str] = None, created_at: Optional[datetime] = None):
        self.value = value
        self.description = description or ""
        self.type = type(value).__name__
        self.created_at = created_at if created_at is not None else datetime.now()
        self.count_items = self._calculate_count(value)

    def _calculate_count(self, value: Any) -> int:
        """Calculate the count of items in the value based on its type."""
        if isinstance(value, (list, tuple, set)):
            return len(value)
        elif isinstance(value, dict):
            return len(value)
        elif isinstance(value, str):
            return len(value)
        elif hasattr(value, '__len__'):
            try:
                return len(value)
            except Exception:
                return 1
        else:
            return 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return {
            "value": self.value,
            "description": self.description,
            "type": self.type,
            "created_at": self.created_at.isoformat(),
            "count_items": self.count_items,
        }


class VariablesManager:
    """Singleton manager for storing and retrieving execution variables.
    
    Provides compact summaries of variables to reduce prompt size.
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(VariablesManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.variables: Dict[str, VariableMetadata] = {}
        self.variable_counter: int = 0
        self._creation_order: list = []
        self._initialized = True

    def add_variable(self, value: Any, name: Optional[str] = None, description: Optional[str] = None) -> str:
        """Add a new variable with an optional name or auto-generated name.
        
        Args:
            value: The value to store
            name: Optional custom name, if None will auto-generate
            description: Optional description of the variable
            
        Returns:
            str: The name of the variable that was created
        """
        if name is None:
            self.variable_counter += 1
            name = f"variable_{self.variable_counter}"
        else:
            # If a custom name is provided and it's a 'variable_X' format,
            # update the counter to avoid future collisions.
            if name.startswith("variable_") and name[9:].isdigit():
                num = int(name[9:])
                if num >= self.variable_counter:
                    self.variable_counter = num

        self.variables[name] = VariableMetadata(value, description)

        # Track creation order
        if name not in self._creation_order:
            self._creation_order.append(name)

        return name

    def get_variable(self, name: str) -> Any:
        """Get a variable value by name.
        
        Args:
            name: The name of the variable
            
        Returns:
            Any: The value of the variable, or None if not found
        """
        metadata = self.variables.get(name)
        return metadata.value if metadata else None

    def get_variable_metadata(self, name: str) -> Optional[VariableMetadata]:
        """Get complete metadata for a variable by name."""
        return self.variables.get(name)

    def get_variables_summary(
        self, 
        variable_names: Optional[list] = None, 
        last_n: Optional[int] = None, 
        max_preview_length: int = 200
    ) -> str:
        """Get a formatted summary of variables with their metadata.
        
        Args:
            variable_names: Optional list of variable names to include
            last_n: Optional number of last created variables to include
            max_preview_length: Maximum length for value preview
            
        Returns:
            str: Formatted string with variable summaries
        """
        if not self.variables:
            return "# No variables stored"

        # Determine which variables to include
        if last_n is not None:
            if last_n <= 0:
                return "# Invalid last_n value: must be greater than 0"
            last_n_names = (
                self._creation_order[-last_n:]
                if len(self._creation_order) >= last_n
                else self._creation_order[:]
            )
            filtered_variables = {
                name: metadata for name, metadata in self.variables.items() if name in last_n_names
            }
        elif variable_names is not None:
            filtered_variables = {
                name: metadata for name, metadata in self.variables.items() if name in variable_names
            }
        else:
            filtered_variables = self.variables

        if not filtered_variables:
            return "# No matching variables found"

        # Build summary
        summary_lines = ["# Variables Summary\n"]
        for name, metadata in filtered_variables.items():
            summary_lines.append(f"## {name}")
            summary_lines.append(f"- Type: {metadata.type}")
            summary_lines.append(f"- Items: {metadata.count_items}")
            if metadata.description:
                summary_lines.append(f"- Description: {metadata.description}")
            
            # Create compact preview
            preview = self._create_preview(metadata.value, max_preview_length)
            summary_lines.append(f"- Value Preview: {preview}")
            summary_lines.append("")

        return "\n".join(summary_lines)

    def _create_preview(self, value: Any, max_length: int) -> str:
        """Create a compact preview of a value."""
        try:
            if isinstance(value, str):
                if len(value) > max_length:
                    return f'"{value[:max_length]}..." (truncated, {len(value)} chars total)'
                return f'"{value}"'
            elif isinstance(value, (list, tuple)):
                if len(value) == 0:
                    return "[]" if isinstance(value, list) else "()"
                elif len(value) > 3:
                    preview_items = [self._create_preview(item, 50) for item in value[:3]]
                    return f"[{', '.join(preview_items)}, ...] ({len(value)} items total)"
                else:
                    preview_items = [self._create_preview(item, 50) for item in value]
                    return f"[{', '.join(preview_items)}]"
            elif isinstance(value, dict):
                if len(value) == 0:
                    return "{}"
                elif len(value) > 3:
                    keys = list(value.keys())[:3]
                    preview_dict = {k: "..." for k in keys}
                    return f"{preview_dict} (+ {len(value) - 3} more keys)"
                else:
                    # Truncate each value
                    preview_dict = {k: self._create_preview(v, 30) for k, v in list(value.items())[:3]}
                    return str(preview_dict)
            else:
                str_val = str(value)
                if len(str_val) > max_length:
                    return f"{str_val[:max_length]}..."
                return str_val
        except Exception:
            return f"<{type(value).__name__}>"

    def reset(self):
        """Reset all variables and counter."""
        self.variables.clear()
        self.variable_counter = 0
        self._creation_order.clear()

    def get_variable_count(self) -> int:
        """Get the total number of stored variables."""
        return len(self.variables)

