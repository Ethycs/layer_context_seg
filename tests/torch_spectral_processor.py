#!/usr/bin/env python3
"""
PyTorch Spectral Processor for Conversation Tracking
===================================================
Integrates real attention extraction with GPU-accelerated spectral clustering.
"""

import sys
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# Add src directory to path
project_root = Path(__file__).parent.resolve()
src_path = project_root / "layered-context-graph" / "src"
sys.path.insert(0, str(src_path))

# Import PyTorch components
from graph.torch_spectral_clustering import ConversationAwareSpectralClustering
from graph.torch_attention_graph_builder import TorchAttentionGraphBuilder
from models.instruction_seeder import InstructionSeeder
from models.attention_extractor import EnhancedAttentionExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TorchSpectralProcessor:
    """
    Main processor using PyTorch-based spectral clustering with real attention.
    """
    
    def __init__(self, 
                 qwq_model_path: Optional[str] = None,
                 device: str = None):
        """
        Initialize the spectral processor.
        
        Args:
            qwq_model_path: Path to QwQ GGUF model
            device: Computation device ('cuda', 'cpu', or None for auto)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Initializing TorchSpectralProcessor on {self.device}")
        
        # Initialize components
        self.attention_extractor = EnhancedAttentionExtractor(qwq_model_path)
        self.graph_builder = TorchAttentionGraphBuilder(device=self.device)
        self.spectral_clustering = ConversationAwareSpectralClustering(device=self.device)
        self.instruction_seeder = InstructionSeeder()
        
    def process_conversation(self, 
                           text: str,
                           mode: str = 'timeline',
                           use_linguistic_programming: bool = True) -> Dict[str, Any]:
        """
        Process conversation using spectral clustering with real attention.
        
        Args:
            text: Conversation transcript
            mode: Processing mode ('timeline', 'speaker', 'evolution', 'topics')
            use_linguistic_programming: Whether to seed instructions
            
        Returns:
            Processing results with spectral clustering
        """
        start_time = datetime.now()
        logger.info(f"Processing conversation in '{mode}' mode...")
        
        # Step 1: Seed instructions if enabled
        if use_linguistic_programming:
            logger.info("Seeding conversation-specific instructions...")
            seeded_text = self.instruction_seeder.seed_conversation_instructions(
                text, mode=mode, density=0.15
            )
            # Also add speaker boundaries for better segmentation
            seeded_text = self.instruction_seeder.seed_speaker_boundaries(seeded_text)
        else:
            seeded_text = text
        
        # Step 2: Extract segments and speaker labels
        segments, speaker_labels = self._extract_conversation_segments(seeded_text)
        logger.info(f"Extracted {len(segments)} conversation segments")
        
        # Step 3: Extract real attention using QwQ
        logger.info("Extracting attention patterns with QwQ...")
        attention_data = self.attention_extractor.extract_attention_for_tape_splitting(segments)
        
        # Step 4: Convert attention to tensor format
        attention_tensor = self._prepare_attention_tensor(attention_data)
        
        # Step 5: Create instruction bias if using linguistic programming
        instruction_bias = None
        if use_linguistic_programming:
            # Find instruction positions in the tokenized text
            instruction_positions = self._find_instruction_positions(seeded_text)
            if instruction_positions:
                instruction_bias = self.instruction_seeder.create_attention_bias_tensor(
                    instruction_positions, 
                    len(segments),
                    instruction_type='SPEAKER_BOUNDARY' if mode == 'speaker' else 'BOUNDARY'
                )
                instruction_bias = instruction_bias.to(self.device)
        
        # Step 6: Build conversation graph with spectral clustering
        logger.info("Building attention graph with spectral clustering...")
        graph = self.graph_builder.build_conversation_graph(
            attention_tensor,
            segments,
            speaker_labels=speaker_labels,
            mode=mode
        )
        
        # Step 7: Apply instruction bias if available
        if instruction_bias is not None:
            adjacency = graph['adjacency_matrix']
            biased_adjacency = adjacency * (1 + instruction_bias * 0.5)
            
            # Re-run spectral clustering with biased adjacency
            speaker_tensor = self._convert_speakers_to_tensor(speaker_labels) if speaker_labels else None
            clustering_results = self.spectral_clustering.segment_conversation(
                biased_adjacency,
                speaker_labels=speaker_tensor,
                mode=mode
            )
            graph['clustering'] = clustering_results
        
        # Step 8: Generate reassembled output based on spectral clusters
        reassembled = self._reassemble_by_spectral_clusters(
            segments, 
            graph['clustering'],
            speaker_labels,
            mode
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'mode': mode,
            'input_length': len(text),
            'segments': len(segments),
            'speakers': list(set(speaker_labels)) if speaker_labels else [],
            'graph': graph,
            'reassembled_text': reassembled,
            'processing_time': processing_time,
            'device': str(self.device),
            'linguistic_programming': use_linguistic_programming,
            'spectral_metadata': {
                'eigenvalues': graph['clustering'].get('eigenvalues'),
                'fiedler_vector': graph['clustering'].get('fiedler_vector'),
                'num_clusters': self._count_clusters(graph['clustering'])
            }
        }
    
    def _extract_conversation_segments(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract conversation segments and speaker labels.
        """
        import re
        
        segments = []
        speakers = []
        
        # Pattern for speaker turns
        speaker_pattern = r'(?:^|\n)((?:Speaker\s+)?[A-Za-z0-9]+):\s*'
        
        # Split by speakers
        parts = re.split(f'({speaker_pattern})', text)
        
        current_speaker = None
        current_content = ""
        
        for i, part in enumerate(parts):
            if re.match(speaker_pattern, part):
                # Save previous segment if exists
                if current_content.strip() and current_speaker:
                    segments.append(f"{current_speaker}: {current_content.strip()}")
                    speakers.append(current_speaker)
                
                # Update speaker
                current_speaker = part.strip().rstrip(':')
                current_content = ""
            else:
                current_content += part
        
        # Don't forget the last segment
        if current_content.strip() and current_speaker:
            segments.append(f"{current_speaker}: {current_content.strip()}")
            speakers.append(current_speaker)
        
        return segments, speakers
    
    def _prepare_attention_tensor(self, attention_data: Dict) -> torch.Tensor:
        """
        Convert attention data to PyTorch tensor.
        """
        # Extract attention patterns from each window
        all_attention = []
        
        for window_data in attention_data.get('window_patterns', []):
            if 'qwq_attention_patterns' in window_data:
                patterns = window_data['qwq_attention_patterns']
                if isinstance(patterns, dict) and 'attention_weights' in patterns:
                    weights = patterns['attention_weights']
                    if isinstance(weights, (list, torch.Tensor)):
                        all_attention.append(torch.tensor(weights))
        
        if not all_attention:
            # Fallback: create dummy attention
            num_segments = len(attention_data.get('window_patterns', []))
            dummy_attention = torch.eye(num_segments)
            return dummy_attention.unsqueeze(0).to(self.device)
        
        # Stack attention patterns
        attention_tensor = torch.stack(all_attention)
        
        # Ensure correct shape (heads, seq, seq)
        if attention_tensor.dim() == 2:
            attention_tensor = attention_tensor.unsqueeze(0)
        
        return attention_tensor.to(self.device)
    
    def _find_instruction_positions(self, text: str) -> List[int]:
        """
        Find positions where instructions were inserted.
        """
        import re
        
        positions = []
        instruction_pattern = r'<([A-Z_]+)>'
        
        for match in re.finditer(instruction_pattern, text):
            positions.append(match.start())
        
        return positions
    
    def _convert_speakers_to_tensor(self, speaker_labels: List[str]) -> torch.Tensor:
        """
        Convert speaker labels to numeric tensor.
        """
        unique_speakers = list(set(speaker_labels))
        speaker_map = {speaker: i for i, speaker in enumerate(unique_speakers)}
        
        speaker_tensor = torch.tensor(
            [speaker_map[s] for s in speaker_labels],
            device=self.device
        )
        
        return speaker_tensor
    
    def _count_clusters(self, clustering_results: Dict) -> int:
        """
        Count number of clusters in results.
        """
        if 'assignments' in clustering_results:
            assignments = clustering_results['assignments']
            if isinstance(assignments, torch.Tensor):
                if assignments.dim() == 2:  # Soft assignments
                    return assignments.shape[1]
                else:  # Hard assignments
                    return len(torch.unique(assignments))
        elif 'final_assignments' in clustering_results:
            return len(torch.unique(clustering_results['final_assignments']))
        return 0
    
    def _reassemble_by_spectral_clusters(self, 
                                       segments: List[str],
                                       clustering_results: Dict,
                                       speaker_labels: List[str],
                                       mode: str) -> str:
        """
        Reassemble segments based on spectral clustering results.
        """
        content_parts = []
        content_parts.append(f"# Spectral Clustering Results - {mode.title()} Mode\n")
        content_parts.append(f"*Using real attention from QwQ-32B model*\n\n")
        
        # Add spectral analysis summary
        if 'eigenvalues' in clustering_results:
            eigenvalues = clustering_results['eigenvalues']
            if isinstance(eigenvalues, torch.Tensor):
                eigen_np = eigenvalues.cpu().numpy()
                content_parts.append("## Spectral Analysis\n")
                content_parts.append(f"- Smallest eigenvalues: {eigen_np[:3]}\n")
                content_parts.append(f"- Spectral gap: {eigen_np[1] - eigen_np[0]:.4f}\n\n")
        
        # Get cluster assignments
        if 'assignments' in clustering_results:
            assignments = clustering_results['assignments']
            if isinstance(assignments, torch.Tensor):
                if assignments.dim() == 2:  # Soft assignments
                    hard_assignments = assignments.argmax(dim=1).cpu().numpy()
                else:
                    hard_assignments = assignments.cpu().numpy()
                
                # Group segments by cluster
                clusters = {}
                for i, (segment, cluster_id) in enumerate(zip(segments, hard_assignments)):
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append((i, segment, speaker_labels[i] if speaker_labels else None))
                
                # Display clusters
                for cluster_id in sorted(clusters.keys()):
                    content_parts.append(f"## Cluster {cluster_id + 1}\n")
                    for idx, segment, speaker in clusters[cluster_id]:
                        content_parts.append(f"[{idx}] {segment}\n")
                    content_parts.append("\n")
        
        elif 'final_assignments' in clustering_results:
            # Hierarchical clustering results
            assignments = clustering_results['final_assignments'].cpu().numpy()
            
            # Group by hierarchical clusters
            clusters = {}
            for i, (segment, cluster_id) in enumerate(zip(segments, assignments)):
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(segment)
            
            # Display hierarchical structure
            content_parts.append("## Hierarchical Clustering\n")
            for level, level_data in clustering_results.get('tree', {}).items():
                content_parts.append(f"### Level {level}\n")
                for split in level_data:
                    content_parts.append(f"- Split into {len(split['partition1'])} + {len(split['partition2'])} segments\n")
            content_parts.append("\n")
            
            # Show final clusters
            for cluster_id in sorted(clusters.keys()):
                content_parts.append(f"## Final Cluster {cluster_id + 1}\n")
                for segment in clusters[cluster_id]:
                    content_parts.append(f"{segment}\n")
                content_parts.append("\n")
        
        else:
            # Fallback: just show segments in order
            content_parts.append("## Segments (No clustering available)\n")
            for segment in segments:
                content_parts.append(f"{segment}\n")
        
        return ''.join(content_parts)


def demo_spectral_conversation_processing():
    """
    Demonstrate spectral clustering on a conversation.
    """
    # Sample conversation
    conversation = """
Speaker A: I think we should implement a microservices architecture.
Speaker B: That's interesting, but what about the complexity it adds?
Speaker A: True, but the scalability benefits outweigh the complexity.
Speaker B: Earlier you mentioned monoliths were simpler. This contradicts that.
Speaker A: You're right. Let me clarify - monoliths are simpler initially.
Speaker B: So you're saying start with monolith, then migrate?
Speaker A: Exactly! That's the evolution I'm proposing.
Speaker B: I agree with that approach. We can start simple and evolve.
Speaker A: Great. Let's also consider how this affects our deployment strategy.
Speaker B: What about using containers from the start?
Speaker A: That would help with the eventual migration to microservices.
Speaker B: This builds on your earlier point about evolution perfectly.
    """
    
    # Initialize processor
    processor = TorchSpectralProcessor()
    
    # Process in different modes
    modes = ['timeline', 'speaker', 'evolution', 'topics']
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Processing conversation in '{mode}' mode")
        print('='*60)
        
        results = processor.process_conversation(
            conversation,
            mode=mode,
            use_linguistic_programming=True
        )
        
        print(f"Processing time: {results['processing_time']:.2f}s")
        print(f"Device: {results['device']}")
        print(f"Segments: {results['segments']}")
        print(f"Speakers: {results['speakers']}")
        
        if 'spectral_metadata' in results:
            meta = results['spectral_metadata']
            print(f"Number of clusters: {meta['num_clusters']}")
            
            if meta['eigenvalues'] is not None:
                print(f"Spectral gap: {meta['eigenvalues'][1] - meta['eigenvalues'][0]:.4f}")
        
        print("\n--- Reassembled Output ---")
        print(results['reassembled_text'][:500] + "...\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PyTorch Spectral Processor for Conversation Tracking"
    )
    parser.add_argument('--input', '-i', help='Input conversation file')
    parser.add_argument('--mode', '-m', 
                       choices=['timeline', 'speaker', 'evolution', 'topics'],
                       default='timeline',
                       help='Processing mode')
    parser.add_argument('--no-linguistic', action='store_true',
                       help='Disable linguistic programming')
    parser.add_argument('--device', choices=['cuda', 'cpu'],
                       help='Force specific device')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with sample conversation')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_spectral_conversation_processing()
    else:
        if not args.input:
            print("Error: --input required when not using --demo")
            sys.exit(1)
        
        # Load conversation
        with open(args.input, 'r') as f:
            conversation = f.read()
        
        # Process
        processor = TorchSpectralProcessor(device=args.device)
        results = processor.process_conversation(
            conversation,
            mode=args.mode,
            use_linguistic_programming=not args.no_linguistic
        )
        
        # Display results
        print(f"Processed {results['segments']} segments in {results['processing_time']:.2f}s")
        print(f"Spectral clustering found {results['spectral_metadata']['num_clusters']} clusters")
        print("\n--- Output ---")
        print(results['reassembled_text'])