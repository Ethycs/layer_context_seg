class GraphReassembler:
    def __init__(self):
        self.graph = None
        self.reassembly_rules = {
            'importance_ordering': True,
            'conceptual_clustering': True,
            'flow_optimization': True,
            'layered_organization': True
        }
        self.reassembled_content = None

    def reassemble_graph(self, nodes, edges, original_document=None):
        """
        RECONSTRUCTION: Build back up from optimal segments into organized structure
        Uses original document as reconstruction seed/template
        """
        
        print(f"🔧 Starting graph reconstruction with {len(nodes)} nodes and {len(edges)} edges")
        if original_document:
            print(f"   🌱 Using original document as reconstruction seed ({len(original_document)} chars)")
        
        # Step 1: Create reconstruction scaffold from original document
        scaffold = self._create_reconstruction_scaffold(original_document) if original_document else None
        
        # Step 2: Analyze the optimal segments we received from iterative disassembly
        segment_analysis = self._analyze_optimal_segments(nodes)
        print(f"   📊 Analyzed {segment_analysis['total_segments']} segments across {len(segment_analysis['segment_types'])} types")
        
        # Step 3: Apply reconstruction rules using scaffold as guide
        reconstructed_structure = self._apply_reconstruction_rules(nodes, edges, segment_analysis, scaffold)
        print(f"   🏗️  Reconstructed into {reconstructed_structure['metadata']['total_layers']} layers")
        
        # Step 4: Generate final reorganized content with FULL text
        reorganized_content = self._generate_full_reorganized_content(reconstructed_structure, scaffold, original_document)
        print(f"   📝 Generated {len(reorganized_content)} characters of reorganized content")
        
        # Calculate compression metrics
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
    
    def _calculate_compression_metrics(self, original_document, nodes, reorganized_content):
        """Calculate comprehensive compression and transformation metrics"""
        if not original_document:
            return {}
        
        original_length = len(original_document)
        reorganized_length = len(reorganized_content)
        
        # Calculate total content in nodes
        total_node_content = sum(len(node.get('content', '')) for node in nodes)
        
        # Calculate unique content (remove overlaps)
        unique_content = set()
        for node in nodes:
            content = node.get('content', '')
            # Split into words to measure unique information
            words = content.split()
            unique_content.update(words)
        unique_word_count = len(unique_content)
        
        # Calculate metrics
        metrics = {
            'original_length': original_length,
            'reorganized_length': reorganized_length,
            'total_node_content_length': total_node_content,
            'compression_ratio': round(original_length / reorganized_length, 2) if reorganized_length > 0 else 0,
            'expansion_ratio': round(reorganized_length / original_length, 2) if original_length > 0 else 0,
            'node_redundancy': round((total_node_content - reorganized_length) / total_node_content, 2) if total_node_content > 0 else 0,
            'unique_words': unique_word_count,
            'information_density': round(unique_word_count / (reorganized_length / 5), 2) if reorganized_length > 0 else 0,  # Assuming avg 5 chars per word
            'node_count': len(nodes),
            'avg_node_size': round(total_node_content / len(nodes), 2) if nodes else 0
        }
        
        return metrics
    
    def _generate_full_reorganized_content(self, reconstructed_structure, scaffold, original_document):
        """Generate final reorganized text with FULL content from reconstruction"""
        
        content_parts = []
        
        # Minimal header for standalone document
        content_parts.append("# Reconstructed Document\n")
        content_parts.append("*Reorganized by semantic importance and relationships*\n\n")
        
        # Skip original document - create standalone output
        
        # Group nodes by reconstruction layer
        layers = {}
        for node in reconstructed_structure['nodes']:
            layer = node.get('reconstruction_layer', 0)
            if layer not in layers:
                layers[layer] = {
                    'name': node.get('layer_name', f'Layer {layer}'),
                    'nodes': []
                }
            layers[layer]['nodes'].append(node)
        
        # Generate content layer by layer
        total_content_length = 0
        
        for layer_num in sorted(layers.keys()):
            layer_data = layers[layer_num]
            layer_name = layer_data['name']
            layer_nodes = layer_data['nodes']
            
            # Skip empty layers
            if not layer_nodes:
                continue
            
            # Sort nodes within layer by importance
            layer_nodes.sort(key=lambda x: x.get('importance', 0), reverse=True)
            
            for i, node in enumerate(layer_nodes, 1):
                # Add section separator for readability
                if i > 1:
                    content_parts.append("---\n\n")
                
                # Add node content directly - minimal metadata
                node_content = node.get('content', '')
                
                # Optional: Add subtle metadata as a header
                segment_type = node.get('segment_type', 'content')
                importance = node.get('importance', 0)
                
                # Add content with light formatting
                content_parts.append(f"## Section {i}: {segment_type.title().replace('_', ' ')}\n")
                content_parts.append(f"*Importance: {importance:.0%}*\n\n")
                content_parts.append(node_content)
                content_parts.append("\n\n")
                
                # Add attention metadata if available
                if 'attention' in node and isinstance(node['attention'], dict):
                    if 'attention_scores' in node['attention']:
                        scores = node['attention']['attention_scores']
                        if isinstance(scores, list) and scores:
                            avg_score = sum(scores) / len(scores)
                            content_parts.append(f"*Attention Score: {avg_score:.3f}*\n\n")
                
                content_parts.append("---\n\n")
        
        # Add comprehensive graph analysis summary
        content_parts.append("## Graph Analysis Summary\n\n")
        content_parts.append(f"- **Total Nodes**: {len(reconstructed_structure['nodes'])}\n")
        content_parts.append(f"- **Total Edges**: {len(reconstructed_structure['edges'])}\n")
        content_parts.append(f"- **Layers**: {len(layers)}\n")
        content_parts.append(f"- **Total Reconstructed Content**: {total_content_length:,} characters\n")
        
        # Add segment type distribution
        segment_types = {}
        for node in reconstructed_structure['nodes']:
            seg_type = node.get('segment_type', 'unknown')
            segment_types[seg_type] = segment_types.get(seg_type, 0) + 1
        
        content_parts.append("\n### Content Distribution by Type:\n")
        for seg_type, count in sorted(segment_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(reconstructed_structure['nodes'])) * 100
            content_parts.append(f"- **{seg_type}**: {count} segments ({percentage:.1f}%)\n")
        
        # Add percolation analysis
        content_parts.append("\n### Percolation Properties:\n")
        avg_connectivity = self._calculate_avg_connectivity(reconstructed_structure)
        graph_density = self._calculate_graph_density(reconstructed_structure)
        content_parts.append(f"- **Average Connectivity**: {avg_connectivity:.2f}\n")
        content_parts.append(f"- **Graph Density**: {graph_density:.2f}\n")
        content_parts.append(f"- **Percolation Threshold**: {'Met' if graph_density > 0.15 else 'Not Met'} (target: 15-30%)\n")
        
        # Add compression summary
        if original_document:
            compression_ratio = len(original_document) / total_content_length if total_content_length > 0 else 0
            content_parts.append("\n### Compression Analysis:\n")
            content_parts.append(f"- **Original Size**: {len(original_document):,} characters\n")
            content_parts.append(f"- **Total Node Content**: {total_content_length:,} characters\n")
            content_parts.append(f"- **Compression Ratio**: {compression_ratio:.2f}:1\n")
            content_parts.append(f"- **Content Preservation**: {(total_content_length / len(original_document) * 100):.1f}%\n")
        
        # Add integration instructions
        content_parts.append("\n---\n")
        content_parts.append("## Integration Instructions\n")
        content_parts.append("This enriched content can be appended back to the original seed document ")
        content_parts.append("to create a multi-dimensional knowledge representation that preserves ")
        content_parts.append("both linear narrative and graph-based semantic relationships.\n\n")
        content_parts.append("The layered structure enables:\n")
        content_parts.append("- **Non-linear navigation** through semantic connections\n")
        content_parts.append("- **Importance-based filtering** for different reading depths\n")
        content_parts.append("- **Type-based organization** for targeted information retrieval\n")
        content_parts.append("- **Percolation-enabled flow** ensuring information connectivity\n")
        
        return ''.join(content_parts)
    
    def _get_node_connections(self, node_id, edges):
        """Get list of connected nodes for a given node"""
        connections = []
        for edge in edges:
            if edge['source'] == node_id:
                connections.append(f"→ {edge['target']}")
            elif edge['target'] == node_id:
                connections.append(f"← {edge['source']}")
        return connections
    
    def _calculate_avg_connectivity(self, structure):
        """Calculate average node connectivity"""
        if not structure['nodes']:
            return 0
        
        node_connections = {}
        for node in structure['nodes']:
            node_connections[node['id']] = 0
        
        for edge in structure['edges']:
            if edge['source'] in node_connections:
                node_connections[edge['source']] += 1
            if edge['target'] in node_connections:
                node_connections[edge['target']] += 1
        
        return sum(node_connections.values()) / len(node_connections)
    
    def _calculate_graph_density(self, structure):
        """Calculate graph density (actual edges / possible edges)"""
        n = len(structure['nodes'])
        if n <= 1:
            return 0
        
        max_edges = n * (n - 1) / 2  # For undirected graph
        actual_edges = len(structure['edges'])
        
        return actual_edges / max_edges if max_edges > 0 else 0
    
    # Include all the other methods from the original GraphReassembler...
    
    def _analyze_optimal_segments(self, nodes):
        """Analyze the final optimal segments from iterative disassembly"""
        analysis = {
            'total_segments': len(nodes),
            'segment_types': {},
            'content_themes': [],
            'length_distribution': [],
            'connectivity_potential': []
        }
        
        for node in nodes:
            content = node.get('content', '')
            
            # Classify segment type for reconstruction planning
            segment_type = self._classify_segment_type(content)
            analysis['segment_types'][segment_type] = analysis['segment_types'].get(segment_type, 0) + 1
            
            # Store segment type in node for later use
            node['segment_type'] = segment_type
            
            # Analyze content characteristics
            analysis['length_distribution'].append(len(content))
            
            # Extract themes for conceptual grouping
            themes = self._extract_content_themes(content)
            analysis['content_themes'].extend(themes)
            
            # Assess connectivity potential
            connectivity = self._assess_connectivity_potential(content, nodes)
            analysis['connectivity_potential'].append(connectivity)
        
        return analysis
    
    def _classify_segment_type(self, content):
        """Classify segment for reconstruction purposes"""
        content_lower = content.lower()
        
        # Look for instruction markers and technical indicators
        if any(marker in content for marker in ['<MATH>', '<QWQ_REASONING>', 'algorithm', 'implementation']):
            return 'technical_core'
        elif any(marker in content for marker in ['<DIALOGUE>', '<QWQ_EXAMPLE>', 'example', 'demonstration']):
            return 'illustrative'
        elif any(word in content_lower for word in ['definition', 'concept', 'introduction', 'overview']):
            return 'foundational'
        elif any(word in content_lower for word in ['application', 'use case', 'practical']):
            return 'application'
        elif any(word in content_lower for word in ['problem', 'issue', 'challenge', 'solution']):
            return 'problem_solving'
        else:
            return 'supporting'
    
    def _extract_content_themes(self, content):
        """Extract key themes for conceptual organization"""
        themes = []
        content_lower = content.lower()
        
        # Domain-specific theme detection
        theme_keywords = {
            'attention_analysis': ['attention', 'head', 'layer', 'weight'],
            'graph_theory': ['graph', 'node', 'edge', 'network', 'connection'],
            'context_processing': ['context', 'window', 'partition', 'segment'],
            'knowledge_representation': ['knowledge', 'semantic', 'representation'],
            'machine_learning': ['model', 'training', 'learning', 'neural'],
            'text_processing': ['text', 'language', 'processing', 'analysis']
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    def _assess_connectivity_potential(self, content, all_nodes):
        """Assess how well this segment connects to others"""
        content_words = set(content.lower().split())
        total_connections = 0
        
        for other_node in all_nodes:
            other_content = other_node.get('content', '')
            other_words = set(other_content.lower().split())
            
            # Calculate word overlap
            overlap = len(content_words.intersection(other_words))
            if overlap > 2:  # Minimum meaningful overlap
                total_connections += overlap
        
        return total_connections / len(content_words) if content_words else 0
    
    def _apply_reconstruction_rules(self, nodes, edges, analysis, scaffold=None):
        """Apply reconstruction rules with scaffold guidance"""
        
        # Store metrics temporarily for use in content generation
        self._temp_metrics = {
            'node_count': len(nodes),
            'original_length': len(scaffold['original_document']) if scaffold and 'original_document' in scaffold else 0,
            'avg_node_size': sum(len(n.get('content', '')) for n in nodes) / len(nodes) if nodes else 0,
            'information_density': len(set(' '.join(n.get('content', '') for n in nodes).split())) / len(' '.join(n.get('content', '') for n in nodes)) * 5 if nodes else 0
        }
        
        # For brevity, using simplified version
        # Full implementation would include all the deduplication and hierarchy building
        
        # Assign all nodes to layer 0 for single-pass mode
        for node in nodes:
            node['reconstruction_layer'] = 0
            node['layer_name'] = 'Primary Content'
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_layers': 1,  # Simplified for single-pass
                'reconstruction_strategy': 'layered_assembly'
            }
        }
    
    def _create_reconstruction_scaffold(self, original_document):
        """Create a structural scaffold from the original document"""
        # Enhanced scaffold with full document
        return {
            'original_length': len(original_document),
            'structure_type': 'linear' if '\n' not in original_document else 'multi_paragraph',
            'original_document': original_document  # Store full document
        }
    
    def _build_graph_structure(self, nodes, edges):
        """Build the basic graph data structure"""
        graph = {}
        for node in nodes:
            graph[node['id']] = {
                'content': node['content'],
                'dependencies': [],
                'importance': node.get('importance', 0),
                'attributes': node.get('attributes', {}),
                'segment_type': node.get('segment_type', 'unknown'),
                'content_length': len(node.get('content', ''))
            }
        
        for edge in edges:
            if edge['source'] in graph and edge['target'] in graph:
                graph[edge['source']]['dependencies'].append({
                    'target': edge['target'],
                    'weight': edge.get('weight', 1.0)
                })
        
        return graph