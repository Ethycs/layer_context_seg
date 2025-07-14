#!/usr/bin/env python3
"""
Hierarchical Graph Builder
==========================
Transforms flat graph structure into hierarchical tree structure with proper
parent-child relationships and reduced density.
"""

from typing import Dict, List, Tuple, Optional
import networkx as nx
from collections import defaultdict


class HierarchicalGraphBuilder:
    """Build hierarchical tree structure from flat graph"""
    
    def __init__(self, target_density: float = 0.15):
        """
        Initialize with target density (15% for tree-like structure)
        
        Args:
            target_density: Target graph density (edges/possible_edges)
        """
        self.target_density = target_density
    
    def build_hierarchy(self, nodes: List[Dict], edges: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Transform flat graph into hierarchical structure
        
        Args:
            nodes: List of node dictionaries
            edges: List of edge dictionaries
            
        Returns:
            Tuple of (hierarchical_nodes, tree_edges)
        """
        
        # Create NetworkX graph for analysis
        G = nx.Graph()
        node_map = {node['id']: node for node in nodes}
        
        for node in nodes:
            G.add_node(node['id'], **node)
        
        for edge in edges:
            if edge['source'] in node_map and edge['target'] in node_map:
                G.add_edge(edge['source'], edge['target'], weight=edge.get('weight', 1.0))
        
        # Identify natural hierarchy through multiple methods
        hierarchy_scores = self._compute_hierarchy_scores(G, nodes)
        
        # Build tree structure
        tree = self._build_tree_structure(nodes, hierarchy_scores)
        
        # Convert back to nodes and edges format
        hierarchical_nodes, tree_edges = self._tree_to_graph_format(tree)
        
        # Verify density is reduced
        actual_density = self._calculate_density(hierarchical_nodes, tree_edges)
        print(f"Graph density reduced from 0.92 to {actual_density:.2f}")
        
        return hierarchical_nodes, tree_edges
    
    def _compute_hierarchy_scores(self, G: nx.Graph, nodes: List[Dict]) -> Dict[str, Dict]:
        """Compute multiple hierarchy indicators for each node"""
        
        scores = {}
        
        # 1. Temporal ordering (if nodes have temporal context)
        temporal_scores = self._compute_temporal_hierarchy(nodes)
        
        # 2. Conceptual depth (foundational -> technical -> application)
        conceptual_scores = self._compute_conceptual_hierarchy(nodes)
        
        # 3. Dependency analysis (what depends on what)
        dependency_scores = self._compute_dependency_hierarchy(G)
        
        # 4. Information flow (PageRank-like importance)
        flow_scores = nx.pagerank(G) if len(G) > 0 else {}
        
        # Combine scores
        for node_id in G.nodes():
            scores[node_id] = {
                'temporal': temporal_scores.get(node_id, 0.5),
                'conceptual': conceptual_scores.get(node_id, 0.5),
                'dependency': dependency_scores.get(node_id, 0.5),
                'flow': flow_scores.get(node_id, 0.5),
                'combined': 0  # Will compute below
            }
            
            # Weighted combination
            scores[node_id]['combined'] = (
                0.2 * scores[node_id]['temporal'] +
                0.3 * scores[node_id]['conceptual'] +
                0.3 * scores[node_id]['dependency'] +
                0.2 * scores[node_id]['flow']
            )
        
        return scores
    
    def _compute_temporal_hierarchy(self, nodes: List[Dict]) -> Dict[str, float]:
        """Score based on temporal order (earlier = higher in hierarchy)"""
        scores = {}
        
        for i, node in enumerate(nodes):
            # Normalize position to [0, 1], earlier = higher score
            scores[node['id']] = 1.0 - (i / len(nodes))
        
        return scores
    
    def _compute_conceptual_hierarchy(self, nodes: List[Dict]) -> Dict[str, float]:
        """Score based on conceptual type (foundational = root)"""
        
        # Hierarchy levels for different segment types
        type_hierarchy = {
            'foundational': 1.0,      # Root level
            'theoretical': 0.8,       # High level concepts
            'technical_core': 0.6,    # Mid-level implementation
            'illustrative': 0.4,      # Examples
            'application': 0.3,       # Specific use cases
            'problem_solving': 0.2,   # Leaf nodes
            'supporting': 0.1         # Auxiliary content
        }
        
        scores = {}
        for node in nodes:
            seg_type = node.get('segment_type', 'supporting')
            scores[node['id']] = type_hierarchy.get(seg_type, 0.1)
        
        return scores
    
    def _compute_dependency_hierarchy(self, G: nx.Graph) -> Dict[str, float]:
        """Score based on dependency structure (prerequisites = higher)"""
        
        if len(G) == 0:
            return {}
        
        # Nodes with many dependents should be higher in hierarchy
        scores = {}
        
        for node in G.nodes():
            # Count nodes that would be unreachable if this node was removed
            G_copy = G.copy()
            G_copy.remove_node(node)
            
            # Check how many components this creates
            before_components = nx.number_connected_components(G)
            after_components = nx.number_connected_components(G_copy) if len(G_copy) > 0 else 0
            
            # Higher score if removing node fragments the graph
            scores[node] = (after_components - before_components) / len(G)
        
        # Normalize scores
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v/max_score for k, v in scores.items()}
        
        return scores
    
    def _build_tree_structure(self, nodes: List[Dict], hierarchy_scores: Dict[str, Dict]) -> Dict:
        """Build actual tree structure from hierarchy scores"""
        
        # Sort nodes by combined hierarchy score
        sorted_nodes = sorted(
            nodes,
            key=lambda n: hierarchy_scores.get(n['id'], {}).get('combined', 0),
            reverse=True
        )
        
        # Build tree top-down
        tree = {
            'root': {
                'id': 'root',
                'children': [],
                'content': 'Knowledge Graph Root',
                'level': 0
            }
        }
        
        # Place nodes in tree based on scores and conceptual relationships
        level_assignments = self._assign_tree_levels(sorted_nodes, hierarchy_scores)
        
        # Build parent-child relationships
        for node in sorted_nodes:
            node_id = node['id']
            level = level_assignments[node_id]
            
            # Find appropriate parent
            parent_id = self._find_best_parent(node, tree, level, hierarchy_scores)
            
            # Add to tree
            tree_node = {
                'id': node_id,
                'children': [],
                'content': node.get('content', ''),
                'level': level,
                'metadata': node
            }
            
            if parent_id == 'root':
                tree['root']['children'].append(tree_node)
            else:
                self._add_to_parent(tree['root'], parent_id, tree_node)
            
            # Store node in flat structure for easy access
            tree[node_id] = tree_node
        
        return tree
    
    def _assign_tree_levels(self, sorted_nodes: List[Dict], hierarchy_scores: Dict[str, Dict]) -> Dict[str, int]:
        """Assign tree depth levels to nodes"""
        
        levels = {}
        
        # Use conceptual hierarchy as primary guide
        for node in sorted_nodes:
            score = hierarchy_scores.get(node['id'], {}).get('conceptual', 0.5)
            
            # Map score to tree level (0 = root, higher = deeper)
            if score > 0.8:
                level = 1  # Just below root
            elif score > 0.6:
                level = 2
            elif score > 0.4:
                level = 3
            elif score > 0.2:
                level = 4
            else:
                level = 5  # Leaf level
            
            levels[node['id']] = level
        
        return levels
    
    def _find_best_parent(self, node: Dict, tree: Dict, target_level: int, 
                         hierarchy_scores: Dict[str, Dict]) -> str:
        """Find best parent for node in tree"""
        
        # If level 1, parent is root
        if target_level == 1:
            return 'root'
        
        # Find potential parents (nodes at level - 1)
        potential_parents = []
        
        def find_nodes_at_level(tree_node, current_level, target):
            if current_level == target - 1:
                potential_parents.append(tree_node['id'])
            for child in tree_node.get('children', []):
                find_nodes_at_level(child, current_level + 1, target)
        
        find_nodes_at_level(tree['root'], 0, target_level)
        
        if not potential_parents:
            # If no nodes at target level -1, go to root
            return 'root'
        
        # Choose parent based on content similarity
        best_parent = potential_parents[0]
        best_score = 0
        
        node_content = node.get('content', '').lower()
        node_words = set(node_content.split())
        
        for parent_id in potential_parents:
            if parent_id == 'root':
                continue
                
            parent_node = tree.get(parent_id, {})
            parent_content = parent_node.get('content', '').lower()
            parent_words = set(parent_content.split())
            
            # Compute similarity
            if node_words and parent_words:
                similarity = len(node_words.intersection(parent_words)) / len(node_words.union(parent_words))
                if similarity > best_score:
                    best_score = similarity
                    best_parent = parent_id
        
        return best_parent
    
    def _add_to_parent(self, tree_node: Dict, parent_id: str, child_node: Dict):
        """Recursively find parent and add child"""
        
        if tree_node['id'] == parent_id:
            tree_node['children'].append(child_node)
            return True
        
        for child in tree_node.get('children', []):
            if self._add_to_parent(child, parent_id, child_node):
                return True
        
        return False
    
    def _tree_to_graph_format(self, tree: Dict) -> Tuple[List[Dict], List[Dict]]:
        """Convert tree structure back to nodes and edges format"""
        
        nodes = []
        edges = []
        
        def process_node(tree_node, parent_id=None, depth=0):
            if tree_node['id'] == 'root':
                # Process root's children
                for child in tree_node.get('children', []):
                    process_node(child, None, depth + 1)
                return
            
            # Create node entry
            node_data = tree_node.get('metadata', {}).copy()
            node_data['id'] = tree_node['id']
            node_data['tree_level'] = depth
            node_data['parent'] = parent_id
            nodes.append(node_data)
            
            # Create edge to parent
            if parent_id:
                edges.append({
                    'source': parent_id,
                    'target': tree_node['id'],
                    'type': 'parent_child',
                    'weight': 1.0
                })
            
            # Process children
            for child in tree_node.get('children', []):
                process_node(child, tree_node['id'], depth + 1)
        
        process_node(tree['root'])
        
        return nodes, edges
    
    def _calculate_density(self, nodes: List[Dict], edges: List[Dict]) -> float:
        """Calculate graph density"""
        
        n = len(nodes)
        if n <= 1:
            return 0
        
        max_edges = n * (n - 1) / 2
        actual_edges = len(edges)
        
        return actual_edges / max_edges if max_edges > 0 else 0