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
        
        # Step 4: Generate final reorganized content with original structure as template
        reorganized_content = self._generate_reorganized_content(reconstructed_structure, scaffold)
        print(f"   📝 Generated {len(reorganized_content)} characters of reorganized content")
        
        return {
            'graph': self._build_graph_structure(reconstructed_structure['nodes'], reconstructed_structure['edges']),
            'nodes': reconstructed_structure['nodes'],
            'edges': reconstructed_structure['edges'],
            'reassembled_text': reorganized_content,
            'reconstruction_metadata': {
                'segment_analysis': segment_analysis,
                'reconstruction_method': 'layered_assembly',
                'content_length': len(reorganized_content),
                'optimization_applied': True
            }
        }
    
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
        """Apply reconstruction rules with scaffold guidance and proper deduplication"""
        
        print(f"   🔍 Step 1: Deduplicating {len(nodes)} nodes...")
        # NEW: Add deduplication as first step
        deduplicated_nodes = self._deduplicate_nodes(nodes)
        print(f"      ✂️  Reduced to {len(deduplicated_nodes)} unique nodes")
        
        # Reconstruction Rule 1: Create conceptual hierarchy guided by scaffold
        if scaffold:
            print(f"   🌱 Using scaffold structure to guide reconstruction")
            hierarchical_groups = self._create_scaffold_guided_hierarchy(deduplicated_nodes, analysis, scaffold)
        else:
            hierarchical_groups = self._create_conceptual_hierarchy(deduplicated_nodes, analysis)
        
        # Reconstruction Rule 2: Optimize information flow within and between groups
        flow_optimized = self._optimize_reconstruction_flow(hierarchical_groups, edges)
        
        # Reconstruction Rule 3: Create layered structure for progressive understanding
        layered_structure = self._create_layered_reconstruction(flow_optimized)
        
        # Reconstruction Rule 4: Establish semantic bridges between concepts
        bridge_edges = self._create_semantic_bridges(layered_structure)
        
        return {
            'nodes': layered_structure['nodes'],
            'edges': bridge_edges,
            'metadata': {
                'total_layers': layered_structure['layer_count'],
                'hierarchy_groups': len(hierarchical_groups),
                'semantic_bridges': len(bridge_edges),
                'reconstruction_strategy': 'progressive_understanding',
                'original_node_count': len(nodes),
                'deduplicated_node_count': len(deduplicated_nodes),
                'deduplication_ratio': f"{len(nodes)-len(deduplicated_nodes)}/{len(nodes)}"
            }
        }
    
    def _create_conceptual_hierarchy(self, nodes, analysis):
        """Create hierarchy: Foundational → Technical → Applications → Examples"""
        
        # Define reconstruction hierarchy (different from disassembly order)
        hierarchy_order = ['foundational', 'technical_core', 'application', 'illustrative', 'problem_solving', 'supporting']
        
        hierarchy_groups = {}
        for level, segment_type in enumerate(hierarchy_order):
            hierarchy_groups[segment_type] = {
                'level': level,
                'nodes': [],
                'priority': len(hierarchy_order) - level  # Higher level = higher priority
            }
        
        # Assign nodes to hierarchy levels
        for node in nodes:
            content = node.get('content', '')
            segment_type = self._classify_segment_type(content)
            
            # Add reconstruction metadata to node
            enhanced_node = {
                'id': node['id'],
                'content': content,
                'segment_type': segment_type,
                'hierarchy_level': hierarchy_groups[segment_type]['level'],
                'reconstruction_priority': hierarchy_groups[segment_type]['priority'],
                'attributes': node.get('attributes', {}),
                'importance': self._calculate_reconstruction_importance(content, segment_type)
            }
            
            hierarchy_groups[segment_type]['nodes'].append(enhanced_node)
        
        return hierarchy_groups
    
    def _calculate_reconstruction_importance(self, content, segment_type):
        """Calculate importance for reconstruction ordering"""
        base_importance = {
            'foundational': 10.0,    # Highest - concepts first
            'technical_core': 8.0,   # High - core technical content
            'application': 6.0,      # Medium-high - practical applications
            'illustrative': 4.0,     # Medium - examples and demos
            'problem_solving': 7.0,  # High - problem-solving content
            'supporting': 2.0        # Low - supporting material
        }.get(segment_type, 1.0)
        
        # Adjust based on content characteristics
        length_factor = min(len(content) / 400, 1.5)  # Prefer moderate length
        keyword_density = self._calculate_keyword_density(content)
        
        return base_importance * length_factor * (1 + keyword_density)
    
    def _calculate_keyword_density(self, content):
        """Calculate density of important domain keywords"""
        important_keywords = [
            'attention', 'graph', 'context', 'layer', 'knowledge', 'semantic',
            'partition', 'segment', 'analysis', 'extraction', 'reconstruction'
        ]
        
        content_lower = content.lower()
        keyword_count = sum(1 for keyword in important_keywords if keyword in content_lower)
        word_count = len(content.split())
        
        return keyword_count / word_count if word_count > 0 else 0
    
    def _optimize_reconstruction_flow(self, hierarchy_groups, original_edges):
        """Optimize information flow for reconstruction (progressive understanding)"""
        
        optimized_groups = {}
        
        for segment_type, group_data in hierarchy_groups.items():
            nodes = group_data['nodes']
            
            # Sort nodes within each group by reconstruction importance
            sorted_nodes = sorted(nodes, key=lambda x: x['importance'], reverse=True)
            
            # Create optimal flow within group
            for i, node in enumerate(sorted_nodes):
                node['group_position'] = i
                node['group_flow_score'] = len(sorted_nodes) - i  # Higher score for earlier nodes
            
            optimized_groups[segment_type] = {
                **group_data,
                'nodes': sorted_nodes,
                'flow_optimized': True
            }
        
        return optimized_groups
    
    def _create_layered_reconstruction(self, optimized_groups):
        """Create layered structure for progressive understanding"""
        
        all_nodes = []
        layer_counter = 0
        
        # Process groups in hierarchy order for layered reconstruction
        hierarchy_order = ['foundational', 'technical_core', 'application', 'illustrative', 'problem_solving', 'supporting']
        
        for segment_type in hierarchy_order:
            if segment_type in optimized_groups:
                group_nodes = optimized_groups[segment_type]['nodes']
                
                # Assign layer to all nodes in this group
                for node in group_nodes:
                    node['reconstruction_layer'] = layer_counter
                    node['layer_name'] = segment_type.replace('_', ' ').title()
                
                all_nodes.extend(group_nodes)
                layer_counter += 1
        
        return {
            'nodes': all_nodes,
            'layer_count': layer_counter,
            'reconstruction_complete': True
        }
    
    def _create_semantic_bridges(self, layered_structure):
        """Create semantic bridges between reconstructed layers"""
        edges = []
        nodes = layered_structure['nodes']
        
        # Create intra-layer connections (within same reconstruction layer)
        current_layer = -1
        layer_nodes = []
        
        for node in nodes:
            if node['reconstruction_layer'] != current_layer:
                # Process previous layer connections
                if layer_nodes:
                    layer_edges = self._create_intra_layer_bridges(layer_nodes)
                    edges.extend(layer_edges)
                
                # Start new layer
                current_layer = node['reconstruction_layer']
                layer_nodes = [node]
            else:
                layer_nodes.append(node)
        
        # Process final layer
        if layer_nodes:
            layer_edges = self._create_intra_layer_bridges(layer_nodes)
            edges.extend(layer_edges)
        
        # Create inter-layer bridges (between reconstruction layers)
        inter_layer_edges = self._create_inter_layer_bridges(nodes)
        edges.extend(inter_layer_edges)
        
        return edges
    
    def _create_intra_layer_bridges(self, layer_nodes):
        """Create semantic bridges within a reconstruction layer"""
        edges = []
        
        # Connect nodes based on group flow score (importance order)
        sorted_nodes = sorted(layer_nodes, key=lambda x: x['group_flow_score'], reverse=True)
        
        for i in range(len(sorted_nodes) - 1):
            edge = {
                'source': sorted_nodes[i]['id'],
                'target': sorted_nodes[i + 1]['id'],
                'weight': 0.8,
                'type': 'reconstruction_flow',
                'bridge_type': 'intra_layer'
            }
            edges.append(edge)
        
        return edges
    
    def _create_inter_layer_bridges(self, nodes):
        """Create semantic bridges between reconstruction layers"""
        edges = []
        
        # Group nodes by reconstruction layer
        layers = {}
        for node in nodes:
            layer = node['reconstruction_layer']
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node)
        
        # Create bridges between adjacent layers
        layer_numbers = sorted(layers.keys())
        for i in range(len(layer_numbers) - 1):
            current_layer_nodes = layers[layer_numbers[i]]
            next_layer_nodes = layers[layer_numbers[i + 1]]
            
            # Bridge highest importance nodes between layers
            if current_layer_nodes and next_layer_nodes:
                source_node = max(current_layer_nodes, key=lambda x: x['importance'])
                target_node = max(next_layer_nodes, key=lambda x: x['importance'])
                
                edge = {
                    'source': source_node['id'],
                    'target': target_node['id'],
                    'weight': 0.9,
                    'type': 'reconstruction_bridge',
                    'bridge_type': 'inter_layer'
                }
                edges.append(edge)
        
        return edges
    
    def _generate_reorganized_content(self, reconstructed_structure, scaffold=None):
        """Generate final reorganized text from reconstruction using scaffold as template"""
        
        content_parts = []
        
        if scaffold:
            content_parts.append("# Reconstructed Knowledge Structure (Scaffold-Guided)\n")
            content_parts.append("*Generated through iterative disassembly and scaffold-guided reconstruction*\n\n")
        else:
            content_parts.append("# Reconstructed Knowledge Structure\n")
            content_parts.append("*Generated through iterative disassembly and layered reconstruction*\n\n")
        
        # Generate content guided by scaffold structure
        if scaffold:
            return self._generate_scaffold_guided_content(reconstructed_structure, scaffold, content_parts)
        
        # Fallback to regular generation
        return self._generate_regular_content(reconstructed_structure, content_parts)
    
    def _generate_scaffold_guided_content(self, reconstructed_structure, scaffold, content_parts):
        """Generate content that follows original document structure"""
        
        # Group nodes by their scaffold matches
        scaffold_matched = []
        unmatched_nodes = []
        
        for node in reconstructed_structure['nodes']:
            if 'scaffold_match' in node:
                scaffold_matched.append(node)
            else:
                unmatched_nodes.append(node)
        
        # Sort by original position
        scaffold_matched.sort(key=lambda n: n.get('original_position', 0))
        
        # Reconstruct following original flow but with enhanced content
        current_section = None
        for node in scaffold_matched:
            scaffold_match = node['scaffold_match']
            
            # Add section headers when we encounter them
            if scaffold_match.get('type') == 'heading':
                content_parts.append(f"\n{'#' * (scaffold_match['level'] + 1)} {scaffold_match['title']}\n")
                current_section = scaffold_match['title']
            
            # Add the enhanced content
            enhanced_content = self._enhance_content_with_graph_insights(node, reconstructed_structure)
            content_parts.append(enhanced_content)
            content_parts.append("\n")
        
        # Add any unmatched nodes at the end
        if unmatched_nodes:
            content_parts.append("\n## Additional Insights\n")
            content_parts.append("*Content discovered through graph analysis not present in original*\n\n")
            for node in unmatched_nodes:
                content_parts.append(f"- **{node.get('segment_type', 'unknown')}**: {node['content'][:200]}...\n")
        
        return ''.join(content_parts)
    
    def _generate_regular_content(self, reconstructed_structure, content_parts):
        """Regular content generation without scaffold"""
        
        # Group nodes by reconstruction layer
        layers = {}
        for node in reconstructed_structure['nodes']:
            layer = node['reconstruction_layer']
            if layer not in layers:
                layers[layer] = []
            layers[layer].append(node)
        
        # Generate content layer by layer
        for layer_num in sorted(layers.keys()):
            content_parts.append(f"\n## Layer {layer_num + 1}\n")
            
            for node in layers[layer_num]:
                content_parts.append(f"### {node.get('segment_type', 'Unknown').title()}\n")
                content_parts.append(f"{node['content']}\n\n")
        
        return ''.join(content_parts)
    
    def _enhance_content_with_graph_insights(self, node, reconstructed_structure):
        """Enhance original content with insights from graph analysis"""
        content = node['content']
        
        # Add metadata about the node's role in the graph
        enhancements = []
        
        if node.get('reconstruction_priority', 0) > 7:
            enhancements.append("**[HIGH PRIORITY]** ")
        
        if node.get('segment_type') == 'foundational':
            enhancements.append("*[Foundation]* ")
        elif node.get('segment_type') == 'technical_core':
            enhancements.append("*[Core Technical]* ")
        
        enhanced_content = ''.join(enhancements) + content
        
        return enhanced_content
    
    def _build_graph_structure(self, nodes, edges):
        """Build the basic graph data structure"""
        graph = {}
        for node in nodes:
            graph[node['id']] = {
                'content': node['content'],
                'dependencies': [],
                'importance': node.get('importance', 0),
                'attributes': node.get('attributes', {})
            }
        
        for edge in edges:
            if edge['source'] in graph and edge['target'] in graph:
                graph[edge['source']]['dependencies'].append({
                    'target': edge['target'],
                    'weight': edge.get('weight', 1.0)
                })
        
        return graph
    
    def _apply_reassembly_rules(self, nodes, edges):
        """Apply intelligent reassembly rules to reorganize content"""
        
        # Rule 1: Importance-based ordering
        importance_ordered = self._order_by_importance(nodes)
        
        # Rule 2: Conceptual clustering  
        clustered_nodes = self._cluster_by_concepts(importance_ordered)
        
        # Rule 3: Flow optimization
        flow_optimized = self._optimize_information_flow(clustered_nodes, edges)
        
        # Rule 4: Layered organization
        layered_structure = self._create_layered_organization(flow_optimized, edges)
        
        return {
            'nodes': layered_structure['nodes'],
            'edges': layered_structure['edges'],
            'metadata': {
                'organization_strategy': 'layered_conceptual',
                'total_layers': layered_structure['layer_count'],
                'cluster_count': len(clustered_nodes),
                'flow_score': layered_structure['flow_score']
            }
        }
    
    def _order_by_importance(self, nodes):
        """Rule 1: Order nodes by importance and centrality"""
        # Calculate importance based on content richness and connections
        for node in nodes:
            content_score = self._calculate_content_importance(node['content'])
            connection_score = self._calculate_connection_importance(node, nodes)
            node['calculated_importance'] = content_score + connection_score
        
        # Sort by calculated importance
        return sorted(nodes, key=lambda x: x['calculated_importance'], reverse=True)
    
    def _cluster_by_concepts(self, nodes):
        """Rule 2: Group nodes by conceptual similarity"""
        clusters = []
        used_nodes = set()
        
        for node in nodes:
            if node['id'] in used_nodes:
                continue
                
            # Find conceptually similar nodes
            cluster = [node]
            used_nodes.add(node['id'])
            
            for other_node in nodes:
                if other_node['id'] in used_nodes:
                    continue
                    
                similarity = self._calculate_conceptual_similarity(node['content'], other_node['content'])
                if similarity > 0.3:  # Threshold for clustering
                    cluster.append(other_node)
                    used_nodes.add(other_node['id'])
            
            clusters.append({
                'id': f'cluster_{len(clusters)}',
                'nodes': cluster,
                'primary_concept': self._extract_primary_concept(cluster),
                'cluster_importance': sum(n['calculated_importance'] for n in cluster)
            })
        
        # Sort clusters by importance
        return sorted(clusters, key=lambda x: x['cluster_importance'], reverse=True)
    
    def _optimize_information_flow(self, clusters, edges):
        """Rule 3: Optimize the flow of information between clusters"""
        # Create flow map between clusters
        flow_map = {}
        
        for cluster in clusters:
            cluster_id = cluster['id']
            flow_map[cluster_id] = {
                'incoming': [],
                'outgoing': [],
                'internal_flow': len(cluster['nodes'])
            }
        
        # Map edges to cluster relationships
        cluster_lookup = {}
        for cluster in clusters:
            for node in cluster['nodes']:
                cluster_lookup[node['id']] = cluster['id']
        
        for edge in edges:
            source_cluster = cluster_lookup.get(edge['source'])
            target_cluster = cluster_lookup.get(edge['target'])
            
            if source_cluster and target_cluster and source_cluster != target_cluster:
                flow_map[source_cluster]['outgoing'].append({
                    'target': target_cluster,
                    'weight': edge.get('weight', 1.0)
                })
                flow_map[target_cluster]['incoming'].append({
                    'source': source_cluster,
                    'weight': edge.get('weight', 1.0)
                })
        
        # Optimize cluster ordering based on flow
        optimized_clusters = self._reorder_by_flow(clusters, flow_map)
        
        return {
            'clusters': optimized_clusters,
            'flow_map': flow_map,
            'flow_score': self._calculate_flow_score(flow_map)
        }
    
    def _create_layered_organization(self, flow_data, original_edges):
        """Rule 4: Organize into hierarchical layers"""
        clusters = flow_data['clusters']
        flow_map = flow_data['flow_map']
        
        # Create layers based on information flow and importance
        layers = []
        current_layer = []
        layer_threshold = 0.7  # Importance threshold for layer separation
        
        for cluster in clusters:
            if cluster['cluster_importance'] > layer_threshold:
                if current_layer:
                    layers.append(current_layer)
                    current_layer = []
                layer_threshold *= 0.8  # Decrease threshold for subsequent layers
            
            current_layer.append(cluster)
        
        if current_layer:
            layers.append(current_layer)
        
        # Create final node and edge structure
        layered_nodes = []
        layered_edges = []
        
        for layer_idx, layer in enumerate(layers):
            for cluster in layer:
                for node in cluster['nodes']:
                    layered_node = {
                        'id': node['id'],
                        'content': node['content'],
                        'layer': layer_idx,
                        'cluster': cluster['id'],
                        'importance': node['calculated_importance'],
                        'original_attributes': node.get('attributes', {})
                    }
                    layered_nodes.append(layered_node)
        
        # Recreate edges with layer information using original edges
        for edge in original_edges:
            source_layer = next((n['layer'] for n in layered_nodes if n['id'] == edge['source']), None)
            target_layer = next((n['layer'] for n in layered_nodes if n['id'] == edge['target']), None)
            
            if source_layer is not None and target_layer is not None:
                layered_edge = {
                    'source': edge['source'],
                    'target': edge['target'],
                    'weight': edge.get('weight', 1.0),
                    'source_layer': source_layer,
                    'target_layer': target_layer,
                    'edge_type': 'cross_layer' if source_layer != target_layer else 'intra_layer'
                }
                layered_edges.append(layered_edge)
        
        return {
            'nodes': layered_nodes,
            'edges': layered_edges,
            'layers': layers,
            'layer_count': len(layers),
            'flow_score': flow_data['flow_score']
        }
    
    def _generate_final_output(self, organized_content):
        """Generate the final reassembled text output"""
        content_sections = []
        
        # Group nodes by layer
        layer_groups = {}
        for node in organized_content['nodes']:
            layer = node['layer']
            if layer not in layer_groups:
                layer_groups[layer] = []
            layer_groups[layer].append(node)
        
        # Generate content for each layer
        for layer_num in sorted(layer_groups.keys()):
            layer_nodes = layer_groups[layer_num]
            
            content_sections.append(f"\n## Layer {layer_num + 1}: Core Concepts\n")
            
            # Sort nodes within layer by importance
            layer_nodes.sort(key=lambda x: x['importance'], reverse=True)
            
            for i, node in enumerate(layer_nodes, 1):
                content = node['content']
                # Extract original text if available
                if isinstance(node.get('original_attributes'), dict):
                    original_text = node['original_attributes'].get('original_text', content)
                    if original_text and original_text != content:
                        content = original_text
                
                content_sections.append(f"### {i}. Concept {node['id']}\n")
                content_sections.append(f"{content}\n\n")
        
        # Add connection analysis
        content_sections.append("\n## Information Flow Analysis\n")
        cross_layer_edges = [e for e in organized_content['edges'] if e.get('edge_type') == 'cross_layer']
        
        if cross_layer_edges:
            content_sections.append("### Cross-Layer Connections:\n")
            for edge in cross_layer_edges[:5]:  # Show top 5 connections
                content_sections.append(f"- Layer {edge['source_layer'] + 1} → Layer {edge['target_layer'] + 1} "
                                      f"(strength: {edge['weight']:.2f})\n")
        
        return "".join(content_sections)
    
    def _calculate_content_importance(self, content):
        """Calculate importance score based on content characteristics"""
        if not content:
            return 0.0
            
        # Factors: length, unique words, special markers
        word_count = len(content.split())
        unique_words = len(set(content.lower().split()))
        has_markers = any(marker in content for marker in ['<', '>', 'def ', 'class ', '```'])
        
        importance = (word_count * 0.1) + (unique_words * 0.3) + (10 if has_markers else 0)
        return min(importance, 10.0)  # Cap at 10
    
    def _calculate_connection_importance(self, node, all_nodes):
        """Calculate importance based on connections to other nodes"""
        # Simple heuristic: nodes that connect to many others are more important
        node_content = node['content'].lower()
        connection_count = 0
        
        for other_node in all_nodes:
            if other_node['id'] == node['id']:
                continue
            other_content = other_node['content'].lower()
            shared_words = set(node_content.split()).intersection(set(other_content.split()))
            if len(shared_words) > 3:  # Has significant word overlap
                connection_count += 1
        
        return connection_count * 0.5
    
    def _calculate_conceptual_similarity(self, content1, content2):
        """Calculate conceptual similarity between two content pieces"""
        if not content1 or not content2:
            return 0.0
            
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _extract_primary_concept(self, cluster_nodes):
        """Extract the primary concept from a cluster of nodes"""
        # Find most common meaningful words across cluster
        all_words = []
        for node in cluster_nodes:
            words = node['content'].lower().split()
            # Filter out common words
            meaningful_words = [w for w in words if len(w) > 3 and w not in 
                              ['that', 'this', 'with', 'from', 'they', 'have', 'been']]
            all_words.extend(meaningful_words)
        
        if not all_words:
            return "general_concept"
            
        # Find most frequent meaningful word
        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        most_common = max(word_freq.items(), key=lambda x: x[1])
        return most_common[0]
    
    def _reorder_by_flow(self, clusters, flow_map):
        """Reorder clusters to optimize information flow"""
        # Simple topological-like ordering
        ordered = []
        remaining = clusters.copy()
        
        while remaining:
            # Find cluster with most incoming connections or highest importance
            best_cluster = None
            best_score = -1
            
            for cluster in remaining:
                cluster_id = cluster['id']
                incoming_score = len(flow_map.get(cluster_id, {}).get('incoming', []))
                importance_score = cluster['cluster_importance']
                total_score = incoming_score + importance_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_cluster = cluster
            
            if best_cluster:
                ordered.append(best_cluster)
                remaining.remove(best_cluster)
            else:
                # Fallback: just take the first remaining
                ordered.append(remaining.pop(0))
        
        return ordered
    
    def _calculate_flow_score(self, flow_map):
        """Calculate overall flow quality score"""
        total_connections = 0
        total_weight = 0
        
        for cluster_data in flow_map.values():
            for outgoing in cluster_data.get('outgoing', []):
                total_connections += 1
                total_weight += outgoing['weight']
        
        return total_weight / total_connections if total_connections > 0 else 0.0
    
    def _deduplicate_nodes(self, nodes):
        """
        Deduplicate nodes by merging those with similar content
        This is the key method to fix the duplication problem
        """
        deduplicated = []
        processed_indices = set()
        
        for i, node in enumerate(nodes):
            if i in processed_indices:
                continue
                
            # Start with current node
            merged_node = {
                'id': node['id'],
                'content': node.get('content', ''),
                'attributes': node.get('attributes', {}),
                'merged_from': [node['id']]
            }
            processed_indices.add(i)
            
            # Find similar nodes to merge
            for j, other_node in enumerate(nodes[i+1:], i+1):
                if j in processed_indices:
                    continue
                    
                similarity = self._calculate_content_similarity_strict(
                    merged_node['content'], 
                    other_node.get('content', '')
                )
                
                # Use much stricter similarity threshold for deduplication
                # Only merge truly duplicate content, not just similar content
                if similarity > 0.95:  # Very high threshold for merging - only near-exact duplicates
                    print(f"      🔗 Merging {other_node['id']} into {merged_node['id']} (similarity: {similarity:.2f})")
                    
                    # Merge the content intelligently
                    merged_node['content'] = self._merge_similar_content(
                        merged_node['content'], 
                        other_node.get('content', '')
                    )
                    merged_node['merged_from'].append(other_node['id'])
                    processed_indices.add(j)
            
            deduplicated.append(merged_node)
        
        return deduplicated
    
    def _calculate_content_similarity_strict(self, content1, content2):
        """Calculate similarity with stricter criteria for deduplication"""
        if not content1 or not content2:
            return 0.0
        
        # Normalize content for comparison
        normalized1 = self._normalize_for_comparison(content1)
        normalized2 = self._normalize_for_comparison(content2)
        
        if not normalized1 or not normalized2:
            return 0.0
        
        words1 = set(normalized1.split())
        words2 = set(normalized2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        # Only treat as duplicate if one is an exact subset and they're very similar
        length_ratio = min(len(normalized1), len(normalized2)) / max(len(normalized1), len(normalized2))
        if (normalized1 in normalized2 or normalized2 in normalized1) and length_ratio > 0.8:
            jaccard_similarity = max(jaccard_similarity, 0.9)  # Still require high base similarity
        
        # Boost similarity if they have identical technical patterns
        if self._have_identical_patterns(content1, content2):
            jaccard_similarity = max(jaccard_similarity, 0.75)
        
        return jaccard_similarity
    
    def _normalize_for_comparison(self, content):
        """Normalize content for similarity comparison"""
        import re
        
        # Remove extra whitespace and normalize
        content = re.sub(r'\s+', ' ', content)
        content = content.strip().lower()
        
        # Remove common prefixes/suffixes that don't affect core meaning
        content = re.sub(r'^(content:|text:|partition \d+:)', '', content)
        content = re.sub(r'(importance: \d+\.\d+)$', '', content)
        
        return content
    
    def _have_identical_patterns(self, content1, content2):
        """Check if two pieces of content have identical technical patterns"""
        import re
        
        # Extract code blocks
        code1 = re.findall(r'```[\s\S]*?```', content1)
        code2 = re.findall(r'```[\s\S]*?```', content2)
        
        # If both have code and the code is similar, they're likely duplicates
        if code1 and code2:
            # Simple check: if any code block from content1 appears in content2
            for c1 in code1:
                for c2 in code2:
                    normalized_c1 = re.sub(r'\s+', ' ', c1.strip())
                    normalized_c2 = re.sub(r'\s+', ' ', c2.strip())
                    if normalized_c1 == normalized_c2:
                        return True
        
        # Check for identical algorithm names or technical terms
        technical_patterns = [
            r'TextTiling Algorithm',
            r'C99 Algorithm', 
            r'Dynamic Programming',
            r'Rule [kg]:', 
            r'def \w+\(',
            r'class \w+:'
        ]
        
        for pattern in technical_patterns:
            matches1 = re.findall(pattern, content1, re.IGNORECASE)
            matches2 = re.findall(pattern, content2, re.IGNORECASE)
            if matches1 and matches2 and set(matches1).intersection(set(matches2)):
                return True
        
        return False
    
    def _merge_similar_content(self, content1, content2):
        """Intelligently merge two similar pieces of content"""
        # Take the longer, more comprehensive version as base
        if len(content1) >= len(content2):
            base_content = content1
            additional_content = content2
        else:
            base_content = content2
            additional_content = content1
        
        # Extract unique information from the additional content
        base_words = set(self._normalize_for_comparison(base_content).split())
        additional_words = set(self._normalize_for_comparison(additional_content).split())
        
        unique_words = additional_words - base_words
        
        # If there's significant unique content (>10 words), append it
        if len(unique_words) > 10:
            # Find sentences in additional content that contain unique information
            import re
            sentences = re.split(r'[.!?]+', additional_content)
            unique_sentences = []
            
            for sentence in sentences:
                sentence_words = set(self._normalize_for_comparison(sentence).split())
                if len(sentence_words.intersection(unique_words)) >= 3:
                    unique_sentences.append(sentence.strip())
            
            if unique_sentences:
                # Append unique information with a separator
                base_content += "\n\n" + ". ".join(unique_sentences[:2])  # Limit to 2 sentences
        
        return base_content
    
    def _create_reconstruction_scaffold(self, original_document):
        """
        Create a structural scaffold from the original document to guide reconstruction
        This preserves the original flow while allowing for improved organization
        """
        if not original_document:
            return None
            
        print(f"   🌱 Creating reconstruction scaffold from original document")
        
        # Extract structural elements from original
        scaffold = {
            'original_length': len(original_document),
            'section_markers': [],
            'paragraph_breaks': [],
            'code_blocks': [],
            'narrative_flow': []
        }
        
        lines = original_document.split('\n')
        current_position = 0
        
        for i, line in enumerate(lines):
            line_start = current_position
            current_position += len(line) + 1  # +1 for newline
            
            # Detect structural elements
            if line.startswith('#'):
                scaffold['section_markers'].append({
                    'line': i,
                    'position': line_start,
                    'level': len(line) - len(line.lstrip('#')),
                    'title': line.strip('#').strip(),
                    'type': 'heading'
                })
            elif line.strip() == '':
                scaffold['paragraph_breaks'].append(line_start)
            elif line.strip().startswith('```') or line.strip().startswith('import ') or line.strip().startswith('def '):
                scaffold['code_blocks'].append({
                    'line': i,
                    'position': line_start,
                    'content': line.strip()
                })
            elif len(line.strip()) > 0:
                scaffold['narrative_flow'].append({
                    'line': i,
                    'position': line_start,
                    'content': line.strip()[:100],  # First 100 chars for matching
                    'type': 'narrative'
                })
        
        print(f"     📋 Scaffold: {len(scaffold['section_markers'])} sections, {len(scaffold['code_blocks'])} code blocks")
        return scaffold
    
    def _create_scaffold_guided_hierarchy(self, nodes, analysis, scaffold):
        """Create hierarchy guided by original document structure"""
        print(f"     🏗️  Creating scaffold-guided hierarchy")
        
        # Start with regular hierarchy
        hierarchy_groups = self._create_conceptual_hierarchy(nodes, analysis)
        
        # Enhance with scaffold information
        for group_name, group_data in hierarchy_groups.items():
            for node in group_data['nodes']:
                content = node.get('content', '')
                
                # Try to match node content with scaffold elements
                best_match = self._find_scaffold_match(content, scaffold)
                if best_match:
                    node['scaffold_match'] = best_match
                    node['original_position'] = best_match.get('position', 0)
                    node['original_line'] = best_match.get('line', 0)
                    
                    # Adjust hierarchy level based on scaffold structure
                    if best_match.get('type') == 'heading':
                        # Headings should be higher in hierarchy
                        node['hierarchy_level'] = max(0, node['hierarchy_level'] - 1)
                        node['reconstruction_priority'] += 2
                
            # Sort nodes within group by original position when available
            group_data['nodes'].sort(key=lambda n: n.get('original_position', float('inf')))
        
        return hierarchy_groups
    
    def _find_scaffold_match(self, content, scaffold):
        """Find the best matching scaffold element for this content"""
        content_preview = content.strip()[:100].lower()
        
        # Check narrative flow first (most likely match)
        for flow_item in scaffold['narrative_flow']:
            if self._content_similarity(content_preview, flow_item['content'].lower()) > 0.7:
                return flow_item
        
        # Check section markers
        for section in scaffold['section_markers']:
            if self._content_similarity(content_preview, section['title'].lower()) > 0.6:
                return section
        
        # Check code blocks
        for code_block in scaffold['code_blocks']:
            if self._content_similarity(content_preview, code_block['content'].lower()) > 0.8:
                return code_block
        
        return None
    
    def _content_similarity(self, text1, text2):
        """Simple similarity based on word overlap"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0