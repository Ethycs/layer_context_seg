"""
Percolation Context Window Module
--------------------------------
Implements context windows with percolation-optimized overlap based on 
graph theory principles. This ensures information flows optimally between
partitions while minimizing redundancy.
"""

import re
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

from .context_window import ContextWindow

class PercolationContextWindow(ContextWindow):
    """
    Context window implementation with percolation-optimized overlap
    based on the condensed architecture
    """
    
    def __init__(self, size: int = 8192, overlap_ratio: float = 0.25):
        """
        Initialize percolation context window
        
        Args:
            size: Maximum size of each window in words/tokens
            overlap_ratio: Overlap between windows (0.15-0.30 recommended
                          based on percolation theory)
        """
        super().__init__(size=size)
        self.overlap_ratio = max(0.15, min(0.30, overlap_ratio))  # Constrain to optimal range
        self.graph = None  # Will store the window connectivity graph
    
    def create_window(self, input_text: Union[str, List[str]]) -> List[str]:
        """
        Create context windows with percolation-optimized overlap
        
        Args:
            input_text: Text to process (string or list of strings)
            
        Returns:
            List of overlapping context windows
        """
        windows = super().create_window(input_text)
        
        # For single window case, no need for percolation
        if len(windows) <= 1:
            return windows
            
        # Apply percolation optimization to ensure proper overlap
        optimized_windows = self._apply_percolation_overlap(windows)
        
        # Build connectivity graph between windows
        self.graph = self._build_window_graph(optimized_windows)
        
        return optimized_windows
    
    def _apply_percolation_overlap(self, windows: List[str]) -> List[str]:
        """
        Ensure windows have optimal overlap according to percolation theory
        
        Args:
            windows: List of initial context windows
            
        Returns:
            List of windows with optimized overlap
        """
        if not windows or len(windows) <= 1:
            return windows
            
        optimized = []
        prev_window = windows[0]
        optimized.append(prev_window)
        
        for i in range(1, len(windows)):
            current = windows[i]
            
            # Calculate current overlap between windows
            overlap = self._calculate_overlap(prev_window, current)
            current_ratio = overlap / len(prev_window.split())
            
            # If overlap is below optimal threshold, create intermediate window
            if current_ratio < self.overlap_ratio:
                # Create intermediate window with proper overlap
                intermediate = self._create_intermediate_window(prev_window, current)
                if intermediate:
                    optimized.append(intermediate)
            
            optimized.append(current)
            prev_window = current
        
        return optimized
    
    def _calculate_overlap(self, window1: str, window2: str) -> int:
        """
        Calculate word overlap between two windows
        
        Args:
            window1: First window text
            window2: Second window text
            
        Returns:
            Number of overlapping words
        """
        # Simple word-based overlap calculation
        words1 = set(window1.split()[-self.size//4:])  # Take last quarter of first window
        words2 = set(window2.split()[:self.size//4])   # Take first quarter of second window
        
        return len(words1.intersection(words2))
    
    def _create_intermediate_window(self, window1: str, window2: str) -> Optional[str]:
        """
        Create an intermediate window to ensure proper overlap
        
        Args:
            window1: First window text
            window2: Second window text
            
        Returns:
            New intermediate window or None if windows already properly overlap
        """
        # Calculate how many words we need from each window
        words1 = window1.split()
        words2 = window2.split()
        
        # Check if windows are too small for meaningful overlap
        if len(words1) < self.size // 4 or len(words2) < self.size // 4:
            return None
            
        # Take the latter part of window1 and early part of window2
        overlap_size = int(self.size * self.overlap_ratio)
        end_of_w1 = words1[-overlap_size:]
        start_of_w2 = words2[:overlap_size]
        
        # Create intermediate window with proper overlap
        intermediate_words = end_of_w1 + start_of_w2
        
        # Ensure we don't exceed size limit
        if len(intermediate_words) > self.size:
            # Trim to size while preserving overlap ratio
            keep_from_each = self.size // 2
            intermediate_words = end_of_w1[-keep_from_each:] + start_of_w2[:keep_from_each]
            
        return ' '.join(intermediate_words)
    
    def _build_window_graph(self, windows: List[str]) -> Dict:
        """
        Build graph representing connections between windows
        
        Args:
            windows: List of context windows
            
        Returns:
            Dictionary representing the window connectivity graph
        """
        graph = {
            'nodes': [],
            'edges': []
        }
        
        # Add nodes (windows)
        for i, window in enumerate(windows):
            graph['nodes'].append({
                'id': i,
                'content': window,
                'size': len(window.split())
            })
        
        # Add edges based on overlap
        for i in range(len(windows) - 1):
            overlap = self._calculate_overlap(windows[i], windows[i+1])
            overlap_ratio = overlap / len(windows[i].split())
            
            # Calculate edge weight based on overlap ratio
            # Higher overlap = stronger connection
            edge_weight = min(1.0, overlap_ratio / self.overlap_ratio)
            
            graph['edges'].append({
                'source': i,
                'target': i + 1,
                'weight': edge_weight,
                'overlap': overlap
            })
            
            # Add additional edges for windows with semantic similarity
            # This creates small-world network properties
            if i + 2 < len(windows):
                similarity = self._calculate_semantic_similarity(windows[i], windows[i+2])
                if similarity > 0.3:  # Only add if reasonably similar
                    graph['edges'].append({
                        'source': i,
                        'target': i + 2,
                        'weight': similarity,
                        'type': 'semantic'
                    })
        
        return graph
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between text segments
        
        Args:
            text1: First text segment
            text2: Second text segment
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Simple Jaccard similarity for now
        # In a production system, would use embeddings
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_graph(self) -> Dict:
        """Get the window connectivity graph"""
        return self.graph
    
    def get_optimal_overlap(self) -> float:
        """Get the current optimal overlap ratio based on percolation theory"""
        return self.overlap_ratio
