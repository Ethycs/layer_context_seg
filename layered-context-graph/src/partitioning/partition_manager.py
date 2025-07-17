import logging
import numpy as np

logger = logging.getLogger(__name__)

class PartitionManager:
    def __init__(self, attention_extractor, cohesion_threshold=0.4):
        """
        Initializes the manager with an attention extractor.
        
        Args:
            attention_extractor: The initialized QwQModel instance, used for its attention.
            cohesion_threshold: The threshold for detecting a boundary. Lower values find more boundaries.
        """
        self.attention_extractor = attention_extractor
        self.cohesion_threshold = cohesion_threshold
        self.segmentation_history = []

    def create_partitions(self, document: str) -> list[str]:
        """
        Creates partitions by analyzing attention patterns to find semantic boundaries.
        """
        logger.info("Starting attention-based segmentation...")
        if not self.attention_extractor or not hasattr(self.attention_extractor, 'extract_attention'):
            logger.error("Attention extractor not provided or is invalid. Cannot perform segmentation.")
            return document.split('\n\n')

        # 1. Get attention data for the whole document
        try:
            # This assumes the document fits in the model's context window.
            attention_data = self.attention_extractor.extract_attention(document)
        except Exception as e:
            logger.error(f"Failed to extract attention: {e}")
            return document.split('\n\n')

        if not attention_data:
            logger.warning("Could not extract attention. Falling back to basic paragraph split.")
            return document.split('\n\n')

        # 2. Find split points from attention
        tokens = self.attention_extractor.tokenizer.tokenize(document)
        if not tokens:
            return [document]

        # Handle both single and windowed attention formats
        if 'attentions' in attention_data:
            # Single window case
            boundaries = self._find_boundaries_from_attention(attention_data['attentions'])
        elif 'windowed_attentions' in attention_data:
            # Sliding window case
            all_boundaries = set()
            for window in attention_data['windowed_attentions']:
                window_boundaries = self._find_boundaries_from_attention(window['attention'])
                # Offset boundaries by the window's start token
                for b in window_boundaries:
                    all_boundaries.add(b + window['start_token'])
            boundaries = sorted(list(all_boundaries))
        else:
            logger.warning("Unknown attention format. Falling back to basic paragraph split.")
            return document.split('\n\n')
        
        self.segmentation_history.append({
            'method': 'attention_based',
            'found_boundaries': len(boundaries),
            'cohesion_threshold': self.cohesion_threshold
        })

        # 3. Create segments from boundaries
        segments = []
        start_tok_idx = 0
        for boundary_tok_idx in sorted(boundaries):
            # Convert token indices to string
            segment_tokens = tokens[start_tok_idx:boundary_tok_idx]
            segments.append(self.attention_extractor.tokenizer.convert_tokens_to_string(segment_tokens))
            start_tok_idx = boundary_tok_idx
        
        # Add the final segment
        final_segment_tokens = tokens[start_tok_idx:]
        segments.append(self.attention_extractor.tokenizer.convert_tokens_to_string(final_segment_tokens))
        
        logger.info(f"Attention-based segmentation complete. Produced {len(segments)} segments.")
        return [s.strip() for s in segments if s.strip()]

    def _find_boundaries_from_attention(self, attention_matrices: list) -> list[int]:
        """
        Analyzes attention matrices to find optimal segmentation points using a TextTiling-like approach.
        """
        if not attention_matrices:
            return []
            
        # 1. Aggregate attention scores
        try:
            # Convert list of lists to numpy array for robust processing
            np_attention_matrices = [np.array(layer) for layer in attention_matrices]
            # Average across all layers and heads
            aggregated_attention = np.mean([np.mean(layer, axis=0) for layer in np_attention_matrices], axis=0)
            seq_len = aggregated_attention.shape[-1]
        except Exception as e:
            logger.error(f"Could not process attention matrices: {e}")
            return []

        # 2. Calculate cohesion scores
        cohesion_scores = []
        w = max(1, seq_len // 20) # Dynamic window size, e.g., 5% of sequence length

        for i in range(w, seq_len - w):
            # Compare text block before vs. after the gap
            block1 = aggregated_attention[i-w:i, i-w:i]
            block2 = aggregated_attention[i:i+w, i:i+w]
            
            # Calculate internal cohesion
            cohesion1 = block1.mean()
            cohesion2 = block2.mean()
            
            # Calculate cross-block "adhesion"
            adhesion = aggregated_attention[i-w:i, i:i+w].mean()
            
            # A boundary is where internal cohesion is high, but adhesion is low.
            # We'll use a simplified depth score: (cohesion1 + cohesion2) - 2 * adhesion
            depth_score = (cohesion1 + cohesion2) - (2 * adhesion)
            cohesion_scores.append(depth_score)

        # 3. Identify boundaries (peaks in the depth score)
        boundaries = []
        if not cohesion_scores:
            return []
            
        scores = np.array(cohesion_scores)
        # Normalize scores
        scores = (scores - np.mean(scores)) / (np.std(scores) + 1e-6)
        
        for i in range(1, len(scores) - 1):
            # Find peaks that are higher than their neighbors and a threshold
            if scores[i] > scores[i-1] and scores[i] > scores[i+1] and scores[i] > self.cohesion_threshold:
                # Map index back to token position
                boundaries.append(i + w)

        return boundaries

    def get_segmentation_summary(self):
        """Get summary of the segmentation process"""
        return self.segmentation_history
