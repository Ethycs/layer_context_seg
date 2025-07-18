import re
import logging
import uuid
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional

from models.qwq_model import QwQModel
from models.baai_model import BAAIModel
from graph.graph_objects import EnrichedSegment, EdgeType

logger = logging.getLogger(__name__)

class PartitionManager:
    """
    A stateful, central object that manages the hierarchical graph of a text.
    It acts as a pure state manager, with all model operations orchestrated
    by an external process.
    """

    def __init__(self, similarity_threshold: float = 0.75):
        """Initializes the PartitionManager."""
        self.similarity_threshold = similarity_threshold
        
        # State attributes
        self.segments: Dict[str, EnrichedSegment] = {}
        self.graph = nx.DiGraph()
        self.reassembled_outputs: Dict[str, str] = {}
        self.root_id: Optional[str] = None
        self.processing_queue: List[str] = []

    def initialize_graph(self, text: str):
        """
        Creates the root node of the graph from the initial text.
        """
        self.segments = {}
        self.graph = nx.DiGraph()
        
        self.root_id = str(uuid.uuid4())
        root_segment = self._create_enriched_segment(self.root_id, text, 0, parent_id=None)
        self.segments[self.root_id] = root_segment
        self.graph.add_node(self.root_id, segment=root_segment)
        self.processing_queue = [self.root_id]
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

    def add_child_segments(self, parent_id: str, new_content_pieces: List[str]):
        """
        Adds new child segments to a parent node and wires up the graph.
        The new children are added to the processing queue for the next level.
        """
        if parent_id not in self.segments:
            logger.error(f"Parent ID {parent_id} not found in segments.")
            return

        parent_segment = self.segments[parent_id]
        
        if len(new_content_pieces) > 1:
            child_segments = self._create_child_segments(new_content_pieces, parent_segment)
            parent_segment.children = [child.id for child in child_segments]
            for child in child_segments:
                self.graph.add_node(child.id, segment=child)
                self.graph.add_edge(parent_id, child.id, type='hierarchical')
                self.processing_queue.append(child.id)
            logger.info(f"Added {len(child_segments)} children to parent {parent_id}.")
        else:
            # If no new children were created, the parent remains a leaf for the next level.
            self.processing_queue.append(parent_id)

    def get_all_segments_for_embedding(self) -> List[Dict[str, str]]:
        """
        Dispenses all segments for the embedding process.
        """
        return [{'id': seg.id, 'content': seg.content} for seg in self.segments.values()]

    def add_edges_from_similarity(self, source_id: str, target_id: str, similarity: float):
        """
        Receives a similarity score and adds an edge if it meets the threshold.
        """
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
        """Creates a single EnrichedSegment."""
        return EnrichedSegment(
            id=id_str,
            content=content,
            start_pos=start_pos,
            end_pos=start_pos + len(content),
            has_math=bool(re.search(r'\\\[.*\\\]|\\\(.*\\\)|[$]{1,2}[^$]+[$]{1,2}', content)),
            has_code='```' in content,
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
