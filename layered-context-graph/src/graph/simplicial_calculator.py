import numpy as np
import itertools

class SimplicialComplexCalculator:
    """
    A stateful calculator for processing attention windows one by one
    to build up simplicial complex information without holding all data
    in memory.
    """
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.all_simplices_per_head = []

    def _calculate_simplices_for_matrix(self, attention_matrix):
        """Calculates simplicial complexes for a single attention matrix."""
        n_tokens = attention_matrix.shape[0]
        similarity = (attention_matrix + attention_matrix.T) / 2
        strong_connections = similarity > self.threshold
        
        simplices = []
        for i in range(n_tokens):
            simplices.append({i})
            
        for i in range(n_tokens):
            for j in range(i + 1, n_tokens):
                if strong_connections[i, j]:
                    simplices.append({i, j})
        
        for k in range(3, min(n_tokens, 5)):
            k_simplices = []
            for subset in itertools.combinations(range(n_tokens), k):
                is_simplex = all(strong_connections[i, j] for i, j in itertools.combinations(subset, 2))
                if is_simplex:
                    k_simplices.append(set(subset))
            if not k_simplices:
                break
            simplices.extend(k_simplices)
            
        return simplices

    def process_window(self, window_data):
        """
        Processes a single window of attention data, calculates simplices
        for each head, and stores the results.
        """
        if not window_data or 'layers' not in window_data:
            return

        # Average attention across layers for each head
        head_attentions = np.mean([layer['attention'] for layer in window_data['layers']], axis=0)
        
        window_simplices = []
        for head_attention in head_attentions:
            simplices = self._calculate_simplices_for_matrix(head_attention)
            window_simplices.append(simplices)
            
        self.all_simplices_per_head.append({
            "metadata": window_data.get('metadata', {}),
            "simplices": window_simplices,
            "tokens": window_data.get('tokens', [])
        })

    def get_results(self):
        """Returns the collected simplicial complex data."""
        return self.all_simplices_per_head
