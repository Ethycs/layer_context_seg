import networkx as nx
import re
from typing import Dict, List, Tuple, Optional

class GraphReassembler:
    def __init__(self):
        self.strategies = {
            "default": self._linear_reassembly,
            "layered_assembly": self.reassemble_graph_layered,
            "logical_flow": self._topological_sort_reassembly,
            "group_by_topic": self._thematic_clustering_reassembly,
            "summary": self._executive_summary_reassembly,
            # Conversation-specific strategies
            "timeline": self.reassemble_by_timeline,
            "speaker": self.reassemble_by_speaker,
            "evolution": self.reassemble_by_concept_evolution,
            "current_state": self.reassemble_by_current_state
        }
        self.reassembly_rules = {
            'importance_ordering': True,
            'conceptual_clustering': True,
            'flow_optimization': True,
            'layered_organization': True
        }

    def reassemble(self, nodes, edges, strategy="default", original_document=None):
        """
        Main reassembly entry point.
        Converts nodes/edges to a graph and calls the specified strategy.
        """
        # Convert to networkx graph
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node['id'], **node)
        for edge in edges:
            G.add_edge(edge['source'], edge['target'], **edge)

        reassemble_func = self.strategies.get(strategy, self.strategies["default"])
        
        # The original layered method has a different signature
        if strategy == "layered_assembly":
             return self.reassemble_graph_layered(nodes, edges, original_document)

        return reassemble_func(G, original_document)

    def reassemble_graph_layered(self, nodes, edges, original_document=None):
        """
        RECONSTRUCTION: Build back up from optimal segments into organized structure
        Uses original document as reconstruction seed/template.
        This is the original reassembly logic, now a specific strategy.
        """
        
        print(f"🔧 Starting layered graph reconstruction with {len(nodes)} nodes and {len(edges)} edges")
        if original_document:
            print(f"   🌱 Using original document as reconstruction seed ({len(original_document)} chars)")
        
        scaffold = self._create_reconstruction_scaffold(original_document) if original_document else None
        segment_analysis = self._analyze_optimal_segments(nodes)
        reconstructed_structure = self._apply_reconstruction_rules(nodes, edges, segment_analysis, scaffold)
        reorganized_content = self._generate_full_reorganized_content(reconstructed_structure, scaffold, original_document)
        compression_metrics = self._calculate_compression_metrics(original_document, nodes, reorganized_content)
        
        return {
            'graph': self._build_graph_structure(reconstructed_structure['nodes'], reconstructed_structure['edges']),
            'nodes': reconstructed_structure['nodes'],
            'edges': reconstructed_structure['edges'],
            'reassembled_text': reorganized_content,
            'reconstruction_metadata': {
                'segment_analysis': segment_analysis,
                'reconstruction_method': 'layered_assembly',
                'content_length': len(reorganized_content),
                'optimization_applied': True,
                'compression_metrics': compression_metrics
            }
        }

    # Methods from the original graph_reassembler.py
    def _calculate_compression_metrics(self, original_document, nodes, reorganized_content):
        if not original_document: return {}
        original_length = len(original_document)
        reorganized_length = len(reorganized_content)
        total_node_content = sum(len(node.get('content', '')) for node in nodes)
        unique_content = set(word for node in nodes for word in node.get('content', '').split())
        unique_word_count = len(unique_content)
        
        return {
            'original_length': original_length,
            'reorganized_length': reorganized_length,
            'total_node_content_length': total_node_content,
            'compression_ratio': round(original_length / reorganized_length, 2) if reorganized_length > 0 else 0,
            'expansion_ratio': round(reorganized_length / original_length, 2) if original_length > 0 else 0,
            'node_redundancy': round((total_node_content - reorganized_length) / total_node_content, 2) if total_node_content > 0 else 0,
            'unique_words': unique_word_count,
            'information_density': round(unique_word_count / (reorganized_length / 5), 2) if reorganized_length > 0 else 0,
            'node_count': len(nodes),
            'avg_node_size': round(total_node_content / len(nodes), 2) if nodes else 0
        }

    def _generate_full_reorganized_content(self, reconstructed_structure, scaffold, original_document):
        content_parts = ["# Reconstructed Document\n", "*Reorganized by semantic importance and relationships*\n\n"]
        layers = {}
        for node in reconstructed_structure['nodes']:
            layer = node.get('reconstruction_layer', 0)
            if layer not in layers:
                layers[layer] = {'name': node.get('layer_name', f'Layer {layer}'), 'nodes': []}
            layers[layer]['nodes'].append(node)
        
        total_content_length = 0
        for layer_num in sorted(layers.keys()):
            layer_data = layers[layer_num]
            if not layer_data['nodes']: continue
            layer_data['nodes'].sort(key=lambda x: x.get('importance', 0), reverse=True)
            
            for i, node in enumerate(layer_data['nodes'], 1):
                if i > 1: content_parts.append("---\n\n")
                node_content = node.get('content', '')
                total_content_length += len(node_content)
                segment_type = node.get('segment_type', 'content')
                importance = node.get('importance', 0)
                content_parts.append(f"## Section {i}: {segment_type.title().replace('_', ' ')}\n")
                content_parts.append(f"*Importance: {importance:.0%}*\n\n")
                content_parts.append(node_content)
                content_parts.append("\n\n")

        return ''.join(content_parts)

    def _analyze_optimal_segments(self, nodes):
        analysis = {'total_segments': len(nodes), 'segment_types': {}}
        for node in nodes:
            content = node.get('content', '')
            segment_type = self._classify_segment_type(content)
            analysis['segment_types'][segment_type] = analysis['segment_types'].get(segment_type, 0) + 1
            node['segment_type'] = segment_type
        return analysis

    def _classify_segment_type(self, content):
        content_lower = content.lower()
        if any(marker in content for marker in ['<MATH>', '<QWQ_REASONING>', 'algorithm', 'implementation']):
            return 'technical_core'
        elif any(marker in content for marker in ['<DIALOGUE>', '<QWQ_EXAMPLE>', 'example', 'demonstration']):
            return 'illustrative'
        elif any(word in content_lower for word in ['definition', 'concept', 'introduction', 'overview']):
            return 'foundational'
        else:
            return 'supporting'

    def _apply_reconstruction_rules(self, nodes, edges, analysis, scaffold=None):
        for node in nodes:
            node['reconstruction_layer'] = 0
            node['layer_name'] = 'Primary Content'
        return {'nodes': nodes, 'edges': edges, 'metadata': {'total_layers': 1, 'reconstruction_strategy': 'layered_assembly'}}

    def _create_reconstruction_scaffold(self, original_document):
        return {'original_length': len(original_document), 'structure_type': 'linear' if '\n' not in original_document else 'multi_paragraph', 'original_document': original_document}

    def _build_graph_structure(self, nodes, edges):
        graph = {}
        for node in nodes:
            graph[node['id']] = {'content': node['content'], 'dependencies': [], 'importance': node.get('importance', 0)}
        for edge in edges:
            if edge['source'] in graph and edge['target'] in graph:
                graph[edge['source']]['dependencies'].append({'target': edge['target'], 'weight': edge.get('weight', 1.0)})
        return graph

    # --- Methods from graph_reassembler_enhanced.py ---

    def _linear_reassembly(self, graph, original_document):
        content_parts = []
        nodes_with_data = sorted(graph.nodes(data=True), key=lambda x: int(x[0].split('_')[-1]) if '_' in x[0] else 0)
        
        for node_id, data in nodes_with_data:
            node_content = data.get('content', '')
            if not node_content.strip(): continue
            content_parts.append(node_content)
        
        return "\n\n".join(content_parts)

    def _topological_sort_reassembly(self, graph, original_document):
        try:
            ordered_nodes = list(nx.topological_sort(graph))
            content = [graph.nodes[node_id]['content'] for node_id in ordered_nodes]
            return "\n\n".join(content)
        except nx.NetworkXUnfeasible:
            return self._linear_reassembly(graph, original_document)

    def _thematic_clustering_reassembly(self, graph, original_document):
        if len(graph.nodes()) <= 1:
            return self._linear_reassembly(graph, original_document)
        communities = nx.community.greedy_modularity_communities(graph.to_undirected())
        output = []
        for i, community in enumerate(communities):
            output.append(f"--- Theme Cluster {i+1} ---")
            for node_id in community:
                output.append(graph.nodes[node_id]['content'])
            output.append("\n")
        return "\n".join(output)

    def _executive_summary_reassembly(self, graph, original_document):
        if not graph.nodes(): return ""
        pagerank = nx.pagerank(graph)
        num_summary_nodes = max(1, int(len(graph.nodes) * 0.2))
        important_nodes = sorted(pagerank, key=pagerank.get, reverse=True)[:num_summary_nodes]
        content = [graph.nodes[node_id]['content'] for node_id in important_nodes]
        return "\n\n".join(content)

    # --- Conversation-specific reassembly methods ---

    def reassemble_by_timeline(self, graph, original_document=None):
        nodes = sorted(graph.nodes(data=True), key=lambda n: int(n[0].split('_')[-1]) if '_' in n[0] else 0)
        content_parts = ["# Conversation Timeline View\n*Chronological development of the conversation*\n\n"]
        for i, (node_id, data) in enumerate(nodes):
            speaker = self._extract_speaker_from_node(data)
            content_parts.append(f"## [Turn {i+1}] {speaker}\n{data.get('content', '')}\n\n")
        return "".join(content_parts)

    def reassemble_by_speaker(self, graph, original_document=None):
        speaker_groups = {}
        for node_id, data in graph.nodes(data=True):
            speaker = self._extract_speaker_from_node(data)
            if speaker not in speaker_groups: speaker_groups[speaker] = []
            speaker_groups[speaker].append(data)
        
        content_parts = ["# Conversation by Speaker\n*Organized by participant contributions*\n\n"]
        for speaker, speaker_nodes in speaker_groups.items():
            content_parts.append(f"## {speaker}\n*{len(speaker_nodes)} contributions*\n\n")
            speaker_nodes.sort(key=lambda n: n.get('importance', 0), reverse=True)
            for i, node in enumerate(speaker_nodes, 1):
                content_parts.append(f"### Contribution {i}\n{node.get('content', '')}\n\n")
        return "".join(content_parts)

    def reassemble_by_concept_evolution(self, graph, original_document=None):
        # This is a simplified version. A full implementation would require edge analysis.
        return self._topological_sort_reassembly(graph, original_document)

    def reassemble_by_current_state(self, graph, original_document=None):
        # This is a simplified version. A full implementation would require contradiction resolution.
        return self._executive_summary_reassembly(graph, original_document)

    def _extract_speaker_from_node(self, node_data):
        if 'speaker' in node_data: return node_data['speaker']
        content = node_data.get('content', '')
        match = re.match(r'([A-Za-z0-9\s]+):', content)
        return match.group(1).strip() if match else "Unknown Speaker"
