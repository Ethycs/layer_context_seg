from dataclasses import dataclass, field
from typing import Dict, Any

from typing import Dict, Any, Optional, List

@dataclass
class EnrichedSegment:
    """A structured data class for a document segment."""
    id: str
    content: str
    start_pos: int
    end_pos: int
    has_math: bool = False
    has_code: bool = False
    tag: str = 'track'
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
