# Enhanced GraphReassembler with conversation tracking capabilities
import networkx as nx
import re
from typing import Dict, List, Tuple, Optional

class GraphReassembler:
    def __init__(self):
        self.strategies = {
            "default": self._linear_reassembly,
            "logical_flow": self._topological_sort_reassembly,
            "group_by_topic": self._thematic_clustering_reassembly,
            "summary": self._executive_summary_reassembly,
            # Conversation-specific strategies
            "timeline": self.reassemble_by_timeline,
            "speaker": self.reassemble_by_speaker,
            "evolution": self.reassemble_by_concept_evolution,
            "current_state": self.reassemble_by_current_state
        }
        self.graph = None
        self.reassembly_rules = {
            'importance_ordering': True,
            'conceptual_clustering': True,
            'flow_optimization': True,
            'layered_organization': True
        }

    def reassemble(self, graph, strategy="default", original_document=None):
        reassemble_func = self.strategies.get(strategy, self.strategies["default"])
        return reassemble_func(graph, original_document)
    
    def reassemble_graph(self, nodes, edges, original_document=None):
        """Main reassembly method that works with nodes and edges"""
        # Convert to networkx graph
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node['id'], **node)
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], **edge)
        
        # Default reassembly
        return self.reassemble(G, strategy="default", original_document=original_document)

    def _linear_reassembly(self, graph, original_document):
        # Basic reassembly, can be improved
        content = [data['content'] for _, data in graph.nodes(data=True)]
        return "\n\n".join(content)

    def _topological_sort_reassembly(self, graph, original_document):
        print("Reassembling with topological sort...")
        try:
            # This requires a Directed Acyclic Graph (DAG)
            ordered_nodes = list(nx.topological_sort(graph))
            content = [graph.nodes[node_id]['content'] for node_id in ordered_nodes]
            return "\n\n".join(content)
        except nx.NetworkXUnfeasible:
            print("Warning: Graph is not a DAG. Falling back to linear reassembly.")
            return self._linear_reassembly(graph, original_document)

    def _thematic_clustering_reassembly(self, graph, original_document):
        print("Reassembling by thematic clusters...")
        # Use community detection to find clusters
        if len(graph.nodes()) > 1:
            communities = nx.community.greedy_modularity_communities(graph.to_undirected())
            output = []
            for i, community in enumerate(communities):
                output.append(f"--- Theme Cluster {i+1} ---")
                for node_id in community:
                    output.append(graph.nodes[node_id]['content'])
                output.append("\n")
            return "\n".join(output)
        else:
            return self._linear_reassembly(graph, original_document)

    def _executive_summary_reassembly(self, graph, original_document):
        print("Reassembling an executive summary...")
        if len(graph.nodes()) == 0:
            return ""
        # Use PageRank to find the most important nodes
        pagerank = nx.pagerank(graph)
        # Get top 20% of nodes
        num_summary_nodes = max(1, int(len(graph.nodes) * 0.2))
        important_nodes = sorted(pagerank, key=pagerank.get, reverse=True)[:num_summary_nodes]
        
        content = [graph.nodes[node_id]['content'] for node_id in important_nodes]
        return "\n\n".join(content)
    
    # Conversation-specific reassembly methods
    
    def reassemble_by_timeline(self, nodes_or_graph, edges_or_original=None, original_text=None):
        """
        Chronological view preserving temporal order.
        Shows the natural flow of conversation as it happened.
        """
        # Handle both graph and nodes/edges inputs
        if isinstance(nodes_or_graph, nx.Graph):
            graph = nodes_or_graph
            nodes = [{'id': n, **graph.nodes[n]} for n in graph.nodes()]
            edges = [{'source': u, 'target': v, **graph.edges[u,v]} for u, v in graph.edges()]
        else:
            nodes = nodes_or_graph
            edges = edges_or_original if edges_or_original else []
        
        print("🕐 Reassembling conversation in chronological order...")
        
        # Sort nodes by their original position/order
        timeline_nodes = sorted(nodes, key=lambda n: int(n['id'].split('_')[-1]) if '_' in n['id'] else 0)
        
        content_parts = []
        content_parts.append("# Conversation Timeline View\n")
        content_parts.append("*Chronological development of the conversation*\n\n")
        
        for i, node in enumerate(timeline_nodes):
            # Extract speaker if available
            speaker = self._extract_speaker_from_node(node)
            timestamp = f"[Turn {i+1}]"
            
            content_parts.append(f"## {timestamp} {speaker}\n")
            content_parts.append(node.get('content', '') + "\n\n")
            
            # Add any references to earlier points
            refs = self._find_references_to_node(node, edges)
            if refs:
                content_parts.append(f"*References: {', '.join(refs)}*\n\n")
        
        return {
            'mode': 'timeline',
            'reassembled_text': ''.join(content_parts),
            'node_count': len(timeline_nodes)
        }
    
    def reassemble_by_speaker(self, nodes_or_graph, edges_or_original=None, original_text=None):
        """
        Organize by speaker contributions.
        Groups all statements by each participant.
        """
        # Handle both graph and nodes/edges inputs
        if isinstance(nodes_or_graph, nx.Graph):
            graph = nodes_or_graph
            nodes = [{'id': n, **graph.nodes[n]} for n in graph.nodes()]
        else:
            nodes = nodes_or_graph
        
        print("👥 Reassembling conversation by speaker contributions...")
        
        # Group nodes by speaker
        speaker_groups = {}
        for node in nodes:
            speaker = self._extract_speaker_from_node(node)
            if speaker not in speaker_groups:
                speaker_groups[speaker] = []
            speaker_groups[speaker].append(node)
        
        content_parts = []
        content_parts.append("# Conversation by Speaker\n")
        content_parts.append("*Organized by participant contributions*\n\n")
        
        for speaker, speaker_nodes in speaker_groups.items():
            content_parts.append(f"## {speaker}\n")
            content_parts.append(f"*{len(speaker_nodes)} contributions*\n\n")
            
            # Sort by importance within speaker
            speaker_nodes.sort(key=lambda n: n.get('importance', 0), reverse=True)
            
            for i, node in enumerate(speaker_nodes, 1):
                content_parts.append(f"### Contribution {i} (Importance: {node.get('importance', 0):.2f})\n")
                content_parts.append(node.get('content', '') + "\n\n")
        
        return {
            'mode': 'speaker',
            'reassembled_text': ''.join(content_parts),
            'speaker_count': len(speaker_groups)
        }
    
    def reassemble_by_concept_evolution(self, nodes_or_graph, edges_or_original=None, original_text=None):
        """
        Track how ideas evolved through conversation.
        Shows the development and refinement of concepts.
        """
        # Handle both graph and nodes/edges inputs
        if isinstance(nodes_or_graph, nx.Graph):
            graph = nodes_or_graph
            nodes = [{'id': n, **graph.nodes[n]} for n in graph.nodes()]
            edges = [{'source': u, 'target': v, **graph.edges[u,v]} for u, v in graph.edges()]
        else:
            nodes = nodes_or_graph
            edges = edges_or_original if edges_or_original else []
        
        print("🔄 Reassembling conversation by concept evolution...")
        
        # Build concept chains using edges
        concept_chains = self._build_concept_chains(nodes, edges)
        
        content_parts = []
        content_parts.append("# Concept Evolution View\n")
        content_parts.append("*Tracking how ideas developed through the conversation*\n\n")
        
        for concept_id, chain in enumerate(concept_chains, 1):
            if len(chain) > 1:  # Only show concepts that evolved
                content_parts.append(f"## Concept {concept_id}: Evolution Chain\n\n")
                
                for i, node in enumerate(chain):
                    evolution_stage = self._determine_evolution_stage(i, len(chain))
                    speaker = self._extract_speaker_from_node(node)
                    
                    content_parts.append(f"### {evolution_stage} - {speaker}\n")
                    content_parts.append(node.get('content', '') + "\n")
                    
                    # Show relationship to previous
                    if i > 0:
                        relationship = self._find_relationship_type(chain[i-1], node, edges)
                        content_parts.append(f"*{relationship} previous statement*\n")
                    
                    content_parts.append("\n")
        
        return {
            'mode': 'concept_evolution',
            'reassembled_text': ''.join(content_parts),
            'concept_chains': len(concept_chains)
        }
    
    def reassemble_by_current_state(self, nodes_or_graph, edges_or_original=None, original_text=None):
        """
        Latest understanding only, resolving contradictions.
        Shows the final consensus or current state of each topic.
        """
        # Handle both graph and nodes/edges inputs
        if isinstance(nodes_or_graph, nx.Graph):
            graph = nodes_or_graph
            nodes = [{'id': n, **graph.nodes[n]} for n in graph.nodes()]
            edges = [{'source': u, 'target': v, **graph.edges[u,v]} for u, v in graph.edges()]
        else:
            nodes = nodes_or_graph
            edges = edges_or_original if edges_or_original else []
        
        print("📍 Reassembling conversation to show current state...")
        
        # Group nodes by topic/concept
        topic_groups = self._group_by_topics(nodes)
        
        content_parts = []
        content_parts.append("# Current State Summary\n")
        content_parts.append("*Latest understanding and resolutions*\n\n")
        
        for topic, topic_nodes in topic_groups.items():
            # Find the most recent/authoritative statement for each topic
            latest_node = self._find_latest_consensus(topic_nodes, edges)
            
            content_parts.append(f"## Topic: {topic}\n")
            
            # Show final state
            content_parts.append("### Current Understanding:\n")
            content_parts.append(latest_node.get('content', '') + "\n\n")
            
            # Show any unresolved contradictions
            contradictions = self._find_contradictions(topic_nodes, edges)
            if contradictions:
                content_parts.append("### Unresolved Points:\n")
                for contra in contradictions:
                    content_parts.append(f"- {contra}\n")
                content_parts.append("\n")
        
        return {
            'mode': 'current_state',
            'reassembled_text': ''.join(content_parts),
            'topics_resolved': len(topic_groups)
        }
    
    # Helper methods for conversation reassembly
    
    def _extract_speaker_from_node(self, node):
        """Extract speaker information from node"""
        # Check metadata first
        if 'speaker' in node:
            return node['speaker']
        
        # Try to extract from content
        content = node.get('content', '')
        speaker_match = re.match(r'(Speaker\s+[A-Za-z0-9]+:|^[A-Za-z0-9]+:)', content)
        if speaker_match:
            return speaker_match.group(1).rstrip(':')
        
        return "Unknown Speaker"
    
    def _find_references_to_node(self, node, edges):
        """Find nodes that reference this node"""
        refs = []
        node_id = node['id']
        
        for edge in edges:
            if edge['target'] == node_id and edge.get('type') in ['references', 'returns_to']:
                refs.append(f"Turn {edge['source'].split('_')[-1]}")
        
        return refs
    
    def _build_concept_chains(self, nodes, edges):
        """Build chains of related concepts"""
        # Create adjacency list for evolution relationships
        evolution_graph = {}
        for node in nodes:
            evolution_graph[node['id']] = []
        
        for edge in edges:
            if edge.get('type') in ['builds_on', 'elaborates', 'clarifies', 'answers']:
                evolution_graph[edge['source']].append(edge['target'])
        
        # Find chains using DFS
        chains = []
        visited = set()
        
        for node_id in evolution_graph:
            if node_id not in visited:
                chain = []
                self._dfs_chain(node_id, evolution_graph, visited, chain, nodes)
                if len(chain) > 1:
                    chains.append(chain)
        
        return chains
    
    def _dfs_chain(self, node_id, graph, visited, chain, all_nodes):
        """Depth-first search to build concept chains"""
        visited.add(node_id)
        
        # Find the actual node
        node = next((n for n in all_nodes if n['id'] == node_id), None)
        if node:
            chain.append(node)
        
        for neighbor in graph.get(node_id, []):
            if neighbor not in visited:
                self._dfs_chain(neighbor, graph, visited, chain, all_nodes)
    
    def _determine_evolution_stage(self, position, chain_length):
        """Determine the stage name based on position in evolution chain"""
        if position == 0:
            return "Initial Idea"
        elif position == chain_length - 1:
            return "Final Form"
        elif position < chain_length / 2:
            return "Early Development"
        else:
            return "Later Refinement"
    
    def _find_relationship_type(self, node1, node2, edges):
        """Find the relationship type between two nodes"""
        for edge in edges:
            if edge['source'] == node1['id'] and edge['target'] == node2['id']:
                edge_type = edge.get('type', 'follows')
                return edge_type.replace('_', ' ').title()
        return "Follows"
    
    def _group_by_topics(self, nodes):
        """Group nodes by their main topics"""
        # Simple topic extraction based on common words
        topic_groups = {}
        
        for node in nodes:
            content = node.get('content', '').lower()
            # Extract key terms (simplified - could use TF-IDF or LLM)
            words = content.split()
            key_terms = [w for w in words if len(w) > 5][:3]  # Top 3 long words
            
            topic = ' '.join(key_terms) if key_terms else 'general'
            
            if topic not in topic_groups:
                topic_groups[topic] = []
            topic_groups[topic].append(node)
        
        return topic_groups
    
    def _find_latest_consensus(self, topic_nodes, edges):
        """Find the most recent authoritative statement on a topic"""
        # Sort by position (assuming later = more recent)
        topic_nodes.sort(key=lambda n: int(n['id'].split('_')[-1]) if '_' in n['id'] else 0, reverse=True)
        
        # Find node with most confirmations and least contradictions
        best_node = topic_nodes[0]
        best_score = 0
        
        for node in topic_nodes:
            score = 0
            for edge in edges:
                if edge['target'] == node['id']:
                    if edge.get('type') in ['confirms', 'agrees_with']:
                        score += 1
                    elif edge.get('type') in ['contradicts', 'disagrees_with']:
                        score -= 1
            
            if score > best_score:
                best_score = score
                best_node = node
        
        return best_node
    
    def _find_contradictions(self, topic_nodes, edges):
        """Find unresolved contradictions in a topic"""
        contradictions = []
        
        for edge in edges:
            if edge.get('type') in ['contradicts', 'disagrees_with']:
                source_node = next((n for n in topic_nodes if n['id'] == edge['source']), None)
                target_node = next((n for n in topic_nodes if n['id'] == edge['target']), None)
                
                if source_node and target_node:
                    # Check if this contradiction was resolved
                    resolved = False
                    for resolve_edge in edges:
                        if (resolve_edge.get('type') == 'clarifies' and 
                            resolve_edge['source'] in [source_node['id'], target_node['id']]):
                            resolved = True
                            break
                    
                    if not resolved:
                        speaker1 = self._extract_speaker_from_node(source_node)
                        speaker2 = self._extract_speaker_from_node(target_node)
                        contradictions.append(f"{speaker1} vs {speaker2} on this point")
        
        return contradictions