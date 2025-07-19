from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

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

class EdgeType(Enum):
    """Types of relationships between segments."""
    NO_RELATION = 0
    EXPLAINS = 1
    ELABORATES = 2
    CONTRADICTS = 3
    IS_EXAMPLE_OF = 4
    IS_CONSEQUENCE_OF = 5
    DEPENDS_ON = 6
    SUMMARIZES = 7
    REFERENCES = 8
    CONTINUES = 9

@dataclass
class GraphAwareSegment(EnrichedSegment):
    """Segment with graph-aware properties and GAP enhancements."""
    incoming_edges: List[Tuple[str, EdgeType]] = field(default_factory=list)
    outgoing_edges: List[Tuple[str, EdgeType]] = field(default_factory=list)
    attention_density: float = 0.0
    type_distribution: Dict[EdgeType, int] = field(default_factory=dict)
    
    # GAP-specific fields
    cohesion_score: float = 0.0
    node_attention_strength: float = 0.0
    is_hub_node: bool = False
    dual_level_boundaries: List[int] = field(default_factory=list)
    
    def add_incoming_edge(self, from_id: str, edge_type: EdgeType):
        """Add an incoming edge with type."""
        self.incoming_edges.append((from_id, edge_type))
        self.type_distribution[edge_type] = self.type_distribution.get(edge_type, 0) + 1
    
    def add_outgoing_edge(self, to_id: str, edge_type: EdgeType):
        """Add an outgoing edge with type."""
        self.outgoing_edges.append((to_id, edge_type))

@dataclass
class TypeEmbedding:
    """Lightweight type representation for edges."""
    edge_type: EdgeType
    embedding_value: float
    learnable: bool = True
    
    @classmethod
    def create_default_embeddings(cls) -> Dict[EdgeType, 'TypeEmbedding']:
        """Create default type embeddings with small values."""
        default_values = {
            EdgeType.NO_RELATION: 0.0,
            EdgeType.EXPLAINS: 0.05,
            EdgeType.ELABORATES: 0.03,
            EdgeType.CONTRADICTS: -0.1,
            EdgeType.IS_EXAMPLE_OF: 0.04,
            EdgeType.IS_CONSEQUENCE_OF: 0.06,
            EdgeType.DEPENDS_ON: 0.08,
            EdgeType.SUMMARIZES: 0.07,
            EdgeType.REFERENCES: 0.02,
            EdgeType.CONTINUES: 0.01
        }
        return {
            edge_type: cls(edge_type, value) 
            for edge_type, value in default_values.items()
        }


@dataclass
class GAPAnalysisResult:
    """Results from GAP dual-level analysis."""
    segment_id: str
    token_attention_entropy: float = 0.0
    node_attention_patterns: Dict[str, Any] = field(default_factory=dict)
    cross_level_consistency: float = 0.0
    detected_boundaries: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_boundary(self, position: int, strength: float, boundary_type: str):
        """Add a detected boundary."""
        self.detected_boundaries.append({
            'position': position,
            'strength': strength,
            'type': boundary_type
        })


@dataclass 
class DualLevelEdge:
    """Edge with both token and node-level attention information."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float
    token_level_attention: float = 0.0
    node_level_attention: float = 0.0
    cross_level_agreement: float = 0.0
    
    @property
    def combined_strength(self) -> float:
        """Compute combined strength from both levels."""
        return (self.weight + self.token_level_attention + self.node_level_attention) / 3.0
