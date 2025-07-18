import re
import logging
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple
from models.baai_model import BAAIModel
from graph.graph_objects import EnrichedSegment

logger = logging.getLogger(__name__)

class PartitionManager:
    """
    A definitive, model-driven partition manager that creates a hierarchical,
    classified graph of enriched segments from a source text.
    """

    def __init__(self, embedding_model: BAAIModel, min_segment_len: int = 100, 
                 similarity_threshold: float = 0.7, cohesion_threshold: float = 0.5):
        self.embedding_model = embedding_model
        self.min_segment_len = min_segment_len
        self.similarity_threshold = similarity_threshold
        self.cohesion_threshold = cohesion_threshold

    def create_partition_graph(self, text: str) -> Dict[str, List[Any]]:
        """
        Creates a hierarchical graph of enriched partitions from the text.
        """
        # K1: Initial Disassembly
        initial_segments = self._initial_segmentation(text)

        # K2/K3: Refine into sentences
        refined_segments = self._refine_to_sentences(initial_segments)

        # K4: Merge short segments
        merged_segments = self._merge_short_segments(refined_segments)

        # Build the graph from the final segments
        graph = self._build_graph(merged_segments)
        
        # Build the hierarchy within the graph
        nodes, edges = self._build_hierarchy(graph)

        # Classify the final nodes
        nodes = self._classify_nodes(nodes, edges)

        logger.info(f"Graph construction complete. Produced {len(nodes)} final nodes.")
        return {'nodes': nodes, 'edges': edges}

    def _create_enriched_segment(self, id_str: str, content: str, start: int) -> EnrichedSegment:
        """Creates a single EnrichedSegment with metadata."""
        return EnrichedSegment(
            id=id_str,
            content=content,
            start_pos=start,
            end_pos=start + len(content),
            has_math=bool(re.search(r'\\\[.*\\\]|\\\(.*\\\)|[$]{1,2}[^$]+[$]{1,2}', content)),
            has_code='```' in content,
            tag='track'
        )

    def _initial_segmentation(self, text: str) -> List[Dict]:
        """(K1) Isolate special blocks and perform initial semantic segmentation."""
        # This logic is a placeholder for the more complex rule-based segmentation
        # For now, we split by paragraphs as a baseline
        paragraphs = text.split('\n\n')
        segments = []
        current_pos = 0
        for p in paragraphs:
            if p.strip():
                segments.append({'content': p, 'start': current_pos})
            current_pos += len(p) + 2
        return segments

    def _refine_to_sentences(self, segments: List[Dict]) -> List[Dict]:
        """(K2/K3) Refines larger segments into sentences."""
        refined = []
        for seg in segments:
            content = seg['content']
            start_offset = seg['start']
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if s.strip()]
            current_pos_in_seg = 0
            for s in sentences:
                start_pos = content.find(s, current_pos_in_seg)
                if start_pos != -1:
                    refined.append({'content': s, 'start': start_offset + start_pos})
                    current_pos_in_seg = start_pos + len(s)
        return refined

    def _merge_short_segments(self, segments: List[Dict]) -> List[Dict]:
        """(K4) Merges segments that are shorter than the minimum length."""
        if not segments: return []
        merged = []
        buffer = segments[0]
        for i in range(1, len(segments)):
            next_seg = segments[i]
            if len(buffer['content']) < self.min_segment_len:
                buffer['content'] += " " + next_seg['content']
            else:
                merged.append(buffer)
                buffer = next_seg
        if buffer:
            merged.append(buffer)
        return merged

    def _build_graph(self, segments: List[Dict]) -> nx.Graph:
        """Builds a NetworkX graph from segments, processing embeddings in batches."""
        G = nx.Graph()
        node_list = []
        for i, seg_dict in enumerate(segments):
            node_id = f"seg_{i}"
            G.add_node(node_id, **seg_dict)
            node_list.append(node_id)

        if self.embedding_model and len(segments) > 1:
            # Process in batches to manage memory
            batch_size = 64 
            all_embeddings = []
            for i in range(0, len(segments), batch_size):
                batch_contents = [seg['content'] for seg in segments[i:i+batch_size]]
                batch_embeddings = self.embedding_model.encode(batch_contents)
                all_embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(all_embeddings)

            for i in range(len(node_list)):
                for j in range(i + 1, len(node_list)):
                    similarity = np.dot(embeddings[i], embeddings[j])
                    if similarity > self.similarity_threshold:
                        G.add_edge(node_list[i], node_list[j], weight=float(similarity), type='semantic_similarity')
        return G

    def _build_hierarchy(self, G: nx.Graph) -> Tuple[List[Dict], List[Dict]]:
        """Builds a hierarchy from the graph and updates node attributes."""
        if not G.nodes:
            return [], []
            
        # Use a minimum spanning tree to find the core structure
        mst = nx.maximum_spanning_tree(G, weight='weight')
        
        # Convert back to node and edge lists, adding parent/child attributes
        nodes = []
        edges = list(mst.edges(data=True))
        
        # Create a map of nodes for easy access
        node_map = {node_id: data for node_id, data in G.nodes(data=True)}

        # Set parent/child relationships based on the MST
        for u, v in mst.edges():
            # A simple heuristic: the node that appears earlier is the parent
            if node_map[u]['start'] < node_map[v]['start']:
                parent, child = u, v
            else:
                parent, child = v, u
            
            if 'children' not in node_map[parent]:
                node_map[parent]['children'] = []
            node_map[parent]['children'].append(child)
            node_map[child]['parent'] = parent
        
        # Ensure the 'id' is part of the returned node dictionary
        final_nodes = []
        for node_id, data in node_map.items():
            data['id'] = node_id
            final_nodes.append(data)

        return final_nodes, edges

    def _classify_nodes(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Classifies nodes as KEEP, TRACK, or DELETE."""
        if not nodes: return []
        
        graph_dict = {'nodes': nodes, 'edges': edges}
        nx_graph = self._create_networkx_graph(graph_dict)
        pagerank = nx.pagerank(nx_graph) if nx_graph.nodes else {}

        for node in nodes:
            score = pagerank.get(node['id'], 0)
            node['importance'] = score
            
            if 'TODO' in node['content'] or 'FIXME' in node['content'] or '?' in node['content']:
                node['tag'] = 'TRACK'
            elif node.get('has_code') or node.get('has_math'):
                node['tag'] = 'KEEP'
            elif score < (1.0 / (len(nodes) + 1e-6)) * 0.5:
                node['tag'] = 'DELETE'
            else:
                node['tag'] = 'KEEP'
        return nodes

    def _create_networkx_graph(self, graph_dict: Dict) -> nx.Graph:
        """Converts a dictionary graph to a NetworkX graph."""
        G = nx.Graph()
        for node_data in graph_dict.get('nodes', []):
            G.add_node(node_data['id'], **node_data)
        
        for edge_data in graph_dict.get('edges', []):
            if isinstance(edge_data, tuple):
                # Handle tuple format from networkx
                if len(edge_data) == 3:
                    u, v, data = edge_data
                else:
                    u, v = edge_data
                    data = {}
            elif isinstance(edge_data, dict):
                # Handle dict format
                u = edge_data.get('source', edge_data.get(0))
                v = edge_data.get('target', edge_data.get(1))
                data = {k: val for k, val in edge_data.items() 
                       if k not in ['source', 'target', 0, 1]}
            else:
                # Handle list format
                if len(edge_data) >= 2:
                    u, v = edge_data[0], edge_data[1]
                    data = edge_data[2] if len(edge_data) > 2 else {}
                else:
                    continue
            
            G.add_edge(u, v, **data)
        
        return G
