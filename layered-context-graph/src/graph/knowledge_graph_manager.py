import networkx as nx
import re
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)

class KnowledgeGraphManager:
    """
    Manages the creation, enrichment, and classification of nodes and edges
    within a knowledge graph. This class consolidates the logic from the
    previous AttentionGraphBuilder and EdgeDetector.
    """

    def __init__(self, embedding_model=None, similarity_threshold=0.7):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold

    def build_initial_graph(self, segments: List[Any]) -> Dict[str, List[Dict]]:
        """Builds the initial graph from enriched segments."""
        nodes = self._create_nodes_from_segments(segments)
        edges = self._create_initial_edges(nodes)
        return {'nodes': nodes, 'edges': edges}

    def _create_nodes_from_segments(self, segments: List[Any]) -> List[Dict]:
        """Creates graph nodes from EnrichedSegment objects."""
        return [
            {
                'id': seg.id,
                'content': seg.content,
                'has_math': seg.has_math,
                'has_code': seg.has_code,
                'tag': seg.tag,
                'start_pos': seg.start_pos,
                'end_pos': seg.end_pos,
                'importance': 0.0 # To be calculated
            } for seg in segments
        ]

    def _create_initial_edges(self, nodes: List[Dict]) -> List[Dict]:
        """Creates sequential and semantic similarity edges."""
        edges = []
        # Sequential edges
        for i in range(len(nodes) - 1):
            edges.append({'source': nodes[i]['id'], 'target': nodes[i+1]['id'], 'weight': 0.8, 'type': 'sequential'})

        # Semantic similarity edges
        if self.embedding_model and len(nodes) > 1:
            contents = [node['content'] for node in nodes]
            embeddings = self.embedding_model.encode(contents)
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    similarity = np.dot(embeddings[i], embeddings[j])
                    if similarity > self.similarity_threshold:
                        edges.append({'source': nodes[i]['id'], 'target': nodes[j]['id'], 'weight': float(similarity), 'type': 'semantic_similarity'})
        return self._deduplicate_edges(edges)

    def classify_nodes(self, graph: Dict) -> List[Dict]:
        """Classifies nodes as KEEP, TRACK, or DELETE."""
        nodes = graph.get('nodes', [])
        if not nodes: return []
        
        nx_graph = self._create_networkx_graph(graph)
        pagerank = nx.pagerank(nx_graph) if nx_graph.nodes else {}

        for node in nodes:
            score = pagerank.get(node['id'], 0)
            node['importance'] = score
            
            # Classification logic
            if 'TODO' in node['content'] or 'FIXME' in node['content'] or '?' in node['content']:
                node['tag'] = 'TRACK'
            elif node['has_code'] or node['has_math']:
                node['tag'] = 'KEEP'
            elif score < (1.0 / (len(nodes) + 1e-6)) * 0.5: # Low importance
                node['tag'] = 'DELETE'
            else:
                node['tag'] = 'KEEP'
        return nodes

    async def condense_graph(self, nodes: List[Dict], edges: List[Dict], llm_client) -> Tuple[List[Dict], List[Dict]]:
        """Merges clusters of similar text nodes using an LLM."""
        nodes_to_keep = [n for n in nodes if n.get('tag') != 'DELETE']
        if len(nodes_to_keep) < 2 or not self.embedding_model:
            return nodes_to_keep, edges

        contents = [node['content'] for node in nodes_to_keep]
        embeddings = self.embedding_model.encode(contents)
        
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0 - self.similarity_threshold)
        labels = clustering.fit_predict(embeddings)
        
        new_nodes, mappings = [], {}
        for cluster_id in set(labels):
            cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
            if len(cluster_indices) == 1:
                new_nodes.append(nodes_to_keep[cluster_indices[0]])
                continue

            cluster_nodes = [nodes_to_keep[i] for i in cluster_indices]
            new_content = await self._synthesize_content(cluster_nodes, llm_client)
            
            # Create a new node representing the cluster
            representative_node = max(cluster_nodes, key=lambda n: n.get('importance', 0))
            new_node = representative_node.copy()
            new_node.update({
                'id': f"condensed_{cluster_id}",
                'content': new_content,
                'metadata': {'condensed_from': [n['id'] for n in cluster_nodes]}
            })
            new_nodes.append(new_node)
            for old_node in cluster_nodes:
                mappings[old_node['id']] = new_node['id']
                
        final_edges = self._rewire_edges(edges, mappings)
        return new_nodes, final_edges

    async def _synthesize_content(self, nodes: List[Dict], llm_client) -> str:
        """Use LLM to synthesize content from a list of nodes."""
        if not llm_client:
            return "\n\n".join([n['content'] for n in nodes])
        
        prompt = "Synthesize the following similar text segments into a single, coherent paragraph:\n\n"
        for i, node in enumerate(nodes):
            prompt += f"--- Segment {i+1} ---\n{node['content']}\n\n"
        prompt += "--- Synthesized Paragraph ---\n"
        
        response = await llm_client.generate(prompt, max_tokens=512)
        return response.strip()

    def _rewire_edges(self, edges: List[Dict], mappings: Dict) -> List[Dict]:
        """Update edges to point to new, condensed nodes."""
        new_edges = []
        seen_edges = set()
        for edge in edges:
            source = mappings.get(edge['source'], edge['source'])
            target = mappings.get(edge['target'], edge['target'])
            if source != target:
                key = tuple(sorted((source, target)))
                if key not in seen_edges:
                    edge['source'], edge['target'] = source, target
                    new_edges.append(edge)
                    seen_edges.add(key)
        return new_edges

    def _create_networkx_graph(self, graph_dict: Dict) -> nx.Graph:
        """Converts a dictionary graph to a NetworkX graph."""
        G = nx.Graph()
        for node_data in graph_dict.get('nodes', []):
            G.add_node(node_data['id'], **node_data)
        for edge_data in graph_dict.get('edges', []):
            G.add_edge(edge_data['source'], edge_data['target'], weight=edge_data.get('weight', 1.0))
        return G

    def _deduplicate_edges(self, edges: List[Dict]) -> List[Dict]:
        """Remove duplicate edges, keeping the one with the highest weight."""
        edge_map = {}
        for edge in edges:
            key = tuple(sorted((edge['source'], edge['target'])))
            if key not in edge_map or edge['weight'] > edge_map[key]['weight']:
                edge_map[key] = edge
        return list(edge_map.values())
