import re
import logging
import uuid
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional

from models.qwq_model import QwQModel
from models.baai_model import BAAIModel
from graph.graph_objects import EnrichedSegment, EdgeType, GraphAwareSegment, TypeEmbedding

logger = logging.getLogger(__name__)

class PartitionManager:
    """
    A stateful, central object that manages the hierarchical graph of a text.
    It acts as a pure state manager, with all model operations orchestrated
    by an external process.
    
    Supports tree-first construction followed by graph enhancement.
    """

    def __init__(self, similarity_threshold: float = 0.75, use_graph_aware: bool = False):
        """Initializes the PartitionManager.
        
        Args:
            similarity_threshold: Threshold for semantic similarity edges
            use_graph_aware: Whether to use GraphAwareSegment instead of EnrichedSegment
        """
        self.similarity_threshold = similarity_threshold
        self.use_graph_aware = use_graph_aware
        
        # State attributes
        self.segments: Dict[str, EnrichedSegment] = {}
        self.graph = nx.DiGraph()
        self.reassembled_outputs: Dict[str, str] = {}
        self.root_id: Optional[str] = None
        self.processing_queue: List[str] = []
        
        # Tree construction state
        self.tree_construction_complete: bool = False
        self.tree_depth: int = 0
        self.nodes_by_level: Dict[int, List[str]] = {}
        self.processed_by_rule: Dict[str, set] = {}  # Track which nodes processed by which rules

    def initialize_graph(self, text: str):
        """
        Creates the root node of the graph from the initial text.
        """
        self.segments = {}
        self.graph = nx.DiGraph()
        self.tree_construction_complete = False
        self.tree_depth = 0
        self.nodes_by_level = {0: []}
        self.processed_by_rule = {}
        
        self.root_id = str(uuid.uuid4())
        root_segment = self._create_enriched_segment(self.root_id, text, 0, parent_id=None)
        self.segments[self.root_id] = root_segment
        self.graph.add_node(self.root_id, segment=root_segment)
        self.processing_queue = [self.root_id]
        self.nodes_by_level[0] = [self.root_id]
        logger.info(f"Graph initialized with root node {self.root_id}.")

    def get_leaves_to_process(self) -> List[EnrichedSegment]:
        """
        Dispenses the current list of leaf nodes that require processing.
        The internal queue is cleared after dispensing.
        """
        leaves = [self.segments[seg_id] for seg_id in self.processing_queue]
        logger.info(f"Dispensing {len(leaves)} leaf nodes for processing.")
        self.processing_queue = []
        return leaves

    def add_child_segments(self, parent_id: str, new_content_pieces: List[str], level: int = None):
        """
        Adds new child segments to a parent node and wires up the graph.
        The new children are added to the processing queue for the next level.
        Code blocks are marked as atomic and not queued for further processing.
        """
        if parent_id not in self.segments:
            logger.error(f"Parent ID {parent_id} not found in segments.")
            return

        parent_segment = self.segments[parent_id]
        
        # Determine level if not provided
        if level is None:
            level = self._get_node_level(parent_id) + 1
        
        # Update tree depth
        if level > self.tree_depth:
            self.tree_depth = level
            self.nodes_by_level[level] = []
        
        if len(new_content_pieces) > 1:
            child_segments = self._create_child_segments(new_content_pieces, parent_segment)
            parent_segment.children = [child.id for child in child_segments]
            
            # Track segments for creating explains edges
            code_segments = []
            text_segments = []
            
            for child in child_segments:
                self.graph.add_node(child.id, segment=child)
                self.graph.add_edge(parent_id, child.id, type='hierarchical')
                self.nodes_by_level[level].append(child.id)
                
                if child.has_code:
                    child.tag = 'KEEP'
                    child.metadata['atomic'] = True  # Mark as atomic - no further segmentation
                    code_segments.append(child)
                elif child.has_math:
                    child.tag = 'KEEP'
                    self.processing_queue.append(child.id)
                else:
                    self.processing_queue.append(child.id)
                    text_segments.append(child)
            
            # Create explains edges between adjacent text and code segments
            self._create_explains_edges(text_segments, code_segments)
            
            logger.info(f"Added {len(child_segments)} children to parent {parent_id} at level {level}.")
        else:
            # If no new children were created, check if parent is atomic
            if not parent_segment.metadata.get('atomic', False):
                self.processing_queue.append(parent_id)

    def get_all_segments_for_embedding(self) -> List[Dict[str, str]]:
        """
        Dispenses all segments for the embedding process.
        """
        return [{'id': seg.id, 'content': seg.content} for seg in self.segments.values()]

    def add_edges_from_similarity(self, source_id: str, target_id: str, similarity: float):
        """
        Receives a similarity score and adds an edge if it meets the threshold.
        Only works after tree construction is complete.
        """
        if not self.tree_construction_complete:
            logger.warning("Cannot add similarity edges before tree construction is complete.")
            return
            
        if similarity > self.similarity_threshold:
            self.graph.add_edge(
                source_id, 
                target_id, 
                type='semantic_similarity',
                weight=float(similarity)
            )

    def classify(self):
        """Classifies nodes as KEEP, DELETE, or TRACK based on content and importance."""
        if not self.segments:
            return

        pagerank_graph = self.graph.to_undirected()
        pagerank = nx.pagerank(pagerank_graph, weight='weight') if pagerank_graph.nodes else {}

        for seg_id, segment in self.segments.items():
            score = pagerank.get(seg_id, 0)
            segment.metadata['importance'] = score
            
            if 'TODO' in segment.content or 'FIXME' in segment.content:
                segment.tag = 'TRACK'
            elif segment.has_code or segment.has_math:
                segment.tag = 'KEEP'
            elif score < (1.0 / (len(self.segments) + 1e-6)) * 0.5:
                segment.tag = 'DELETE'
            else:
                segment.tag = 'KEEP'
        logger.info("Node classification complete.")

    def reassemble(self, prompt: str, key: str, synthesis_model: QwQModel) -> str:
        """
        Generates a new text output by traversing the graph using a provided synthesis model.
        """
        if not self.segments:
            return ""

        nodes_to_reassemble = [
            self.segments[node_id] for node_id in nx.dfs_preorder_nodes(self.graph, source=self.root_id)
            if self.segments[node_id].tag == 'KEEP'
        ]
        
        context_for_synthesis = "\\n\\n---\\n\\n".join([seg.content for seg in nodes_to_reassemble])
        
        final_prompt = f"{prompt}\\n\\nUse the following context to inform your response:\\n\\n{context_for_synthesis}"
        
        reassembled_text = synthesis_model.generate(final_prompt, max_tokens=2048)
        self.reassembled_outputs[key] = reassembled_text
        logger.info(f"Reassembly for '{key}' complete.")
        return reassembled_text

    def _create_enriched_segment(self, id_str: str, content: str, start_pos: int, parent_id: Optional[str]) -> EnrichedSegment:
        """Creates a single EnrichedSegment or GraphAwareSegment."""
        # Check for code patterns beyond just ```
        has_code = ('```' in content or 
                   bool(re.search(r'^\s*(def|class|function|import|from|var|let|const|public|private|return)\s+', content, re.MULTILINE)) or
                   bool(re.search(r'[{};]\s*$', content, re.MULTILINE)))  # Common code line endings
        
        # Enhanced math detection
        has_math = bool(re.search(r'\\\[.*\\\]|\\\(.*\\\)|[$]{1,2}[^$]+[$]{1,2}|\\\\\w+{', content))
        
        if self.use_graph_aware:
            return GraphAwareSegment(
                id=id_str,
                content=content,
                start_pos=start_pos,
                end_pos=start_pos + len(content),
                has_math=has_math,
                has_code=has_code,
                tag='track',
                parent=parent_id,
                children=[]
            )
        else:
            return EnrichedSegment(
                id=id_str,
                content=content,
                start_pos=start_pos,
                end_pos=start_pos + len(content),
                has_math=has_math,
                has_code=has_code,
                tag='track',
                parent=parent_id,
                children=[]
            )

    def _create_child_segments(self, content_pieces: List[str], parent_segment: EnrichedSegment) -> List[EnrichedSegment]:
        """Creates child segments and adds them to the master segment list."""
        children = []
        current_pos_in_parent = 0
        for piece in content_pieces:
            if not piece.strip():
                continue
            
            start_pos = parent_segment.content.find(piece, current_pos_in_parent)
            if start_pos == -1:
                continue
            
            child_id = str(uuid.uuid4())
            child_segment = self._create_enriched_segment(
                id_str=child_id,
                content=piece,
                start_pos=parent_segment.start_pos + start_pos,
                parent_id=parent_segment.id
            )
            self.segments[child_id] = child_segment
            children.append(child_segment)
            current_pos_in_parent = start_pos + len(piece)
        return children
    
    def _get_node_level(self, node_id: str) -> int:
        """Get the level of a node in the tree."""
        for level, nodes in self.nodes_by_level.items():
            if node_id in nodes:
                return level
        return 0
    
    def complete_tree_construction(self):
        """
        Mark tree construction as complete.
        This enables graph enhancement with semantic edges.
        """
        self.tree_construction_complete = True
        logger.info(f"Tree construction complete. Depth: {self.tree_depth}, Total nodes: {len(self.segments)}")
    
    def get_tree_statistics(self) -> Dict[str, Any]:
        """Get statistics about the constructed tree."""
        stats = {
            'total_nodes': len(self.segments),
            'tree_depth': self.tree_depth,
            'nodes_per_level': {level: len(nodes) for level, nodes in self.nodes_by_level.items()},
            'leaf_nodes': sum(1 for seg in self.segments.values() if not seg.children),
            'branching_factors': {}
        }
        
        # Calculate average branching factor per level
        for level in range(self.tree_depth):
            level_nodes = self.nodes_by_level.get(level, [])
            total_children = sum(len(self.segments[node_id].children) for node_id in level_nodes)
            if level_nodes:
                stats['branching_factors'][level] = total_children / len(level_nodes)
        
        return stats
    
    def has_unprocessed_leaves(self) -> bool:
        """Check if there are still leaves to process."""
        return len(self.processing_queue) > 0
    
    def mark_processed(self, node_id: str, rule: str):
        """Mark a node as processed by a specific rule."""
        if node_id not in self.processed_by_rule:
            self.processed_by_rule[node_id] = set()
        self.processed_by_rule[node_id].add(rule)
    
    def is_fully_processed(self, node_id: str, rules: List[str]) -> bool:
        """Check if a node has been processed by all rules."""
        if node_id not in self.processed_by_rule:
            return False
        return all(rule in self.processed_by_rule[node_id] for rule in rules)
    
    def convert_tree_to_graph(self, embeddings: Dict[str, np.ndarray], similarity_threshold: float = None):
        """
        Convert the tree to a graph by adding semantic similarity edges.
        This should only be called after tree construction is complete.
        
        Args:
            embeddings: Dictionary mapping segment IDs to embedding vectors
            similarity_threshold: Optional override for similarity threshold
        """
        if not self.tree_construction_complete:
            logger.error("Cannot convert to graph before tree construction is complete.")
            return
        
        threshold = similarity_threshold or self.similarity_threshold
        edges_added = 0
        
        segment_ids = list(embeddings.keys())
        for i in range(len(segment_ids)):
            for j in range(i + 1, len(segment_ids)):
                id_i, id_j = segment_ids[i], segment_ids[j]
                
                # Skip if edge already exists (hierarchical)
                if self.graph.has_edge(id_i, id_j) or self.graph.has_edge(id_j, id_i):
                    continue
                
                # Calculate similarity
                similarity = np.dot(embeddings[id_i], embeddings[id_j])
                
                if similarity > threshold:
                    self.graph.add_edge(
                        id_i, id_j,
                        type='semantic_similarity',
                        weight=float(similarity)
                    )
                    edges_added += 1
        
        logger.info(f"Added {edges_added} semantic similarity edges to the graph.")
        
        # Update GraphAwareSegment edge information if using graph-aware mode
        if self.use_graph_aware:
            self._update_graph_aware_edges()
        
        return edges_added
    
    def _update_graph_aware_edges(self):
        """Update GraphAwareSegment objects with edge information."""
        # Clear existing edge information
        for segment in self.segments.values():
            if isinstance(segment, GraphAwareSegment):
                segment.incoming_edges.clear()
                segment.outgoing_edges.clear()
                segment.type_distribution.clear()
        
        # Populate edge information
        for source_id, target_id, data in self.graph.edges(data=True):
            edge_type_value = data.get('edge_type', EdgeType.NO_RELATION.value)
            edge_type = EdgeType(edge_type_value)
            
            source_seg = self.segments.get(source_id)
            target_seg = self.segments.get(target_id)
            
            if isinstance(source_seg, GraphAwareSegment):
                source_seg.add_outgoing_edge(target_id, edge_type)
            
            if isinstance(target_seg, GraphAwareSegment):
                target_seg.add_incoming_edge(source_id, edge_type)
        
        # Calculate attention density for each segment
        for segment in self.segments.values():
            if isinstance(segment, GraphAwareSegment):
                total_edges = len(segment.incoming_edges) + len(segment.outgoing_edges)
                segment.attention_density = total_edges / max(1, len(self.segments) - 1)
                
                # Mark hub nodes (high connectivity)
                if total_edges > 2 * (self.graph.number_of_edges() / len(self.segments)):
                    segment.is_hub_node = True
    
    def _create_explains_edges(self, text_segments: List[EnrichedSegment], 
                              code_segments: List[EnrichedSegment]):
        """
        Create 'explains' edges between text and code segments based on proximity.
        Text immediately before or after code is assumed to explain it.
        """
        if not text_segments or not code_segments:
            return
        
        # Sort segments by position to find adjacency
        all_segments = text_segments + code_segments
        all_segments.sort(key=lambda s: s.start_pos)
        
        # For each code segment, find adjacent text segments
        for code_seg in code_segments:
            code_idx = all_segments.index(code_seg)
            
            # Check previous segment
            if code_idx > 0:
                prev_seg = all_segments[code_idx - 1]
                if prev_seg in text_segments:
                    # Text before code typically introduces/explains it
                    self.graph.add_edge(
                        prev_seg.id,
                        code_seg.id,
                        type='semantic',
                        edge_type=EdgeType.EXPLAINS.value,
                        weight=0.9
                    )
                    logger.debug(f"Added EXPLAINS edge: {prev_seg.id} -> {code_seg.id}")
            
            # Check next segment
            if code_idx < len(all_segments) - 1:
                next_seg = all_segments[code_idx + 1]
                if next_seg in text_segments:
                    # Text after code might explain output or usage
                    self.graph.add_edge(
                        next_seg.id,
                        code_seg.id,
                        type='semantic',
                        edge_type=EdgeType.EXPLAINS.value,
                        weight=0.7
                    )
                    logger.debug(f"Added EXPLAINS edge: {next_seg.id} -> {code_seg.id}")
