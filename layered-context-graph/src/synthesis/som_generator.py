#!/usr/bin/env python3
"""
Self-Organizing Document Generator
==================================
This module implements the "Tape -> Map -> Path -> Tape" pipeline using a
Self-Organizing Map (SOM) to create a conceptual landscape of a document.
"""

import numpy as np
import json
from minisom import MiniSom
import logging

logger = logging.getLogger(__name__)

class SOM_DocumentGenerator:
    """
    Transforms a linear document into a 2D map of concepts and then
    intelligently traverses this map to generate a new, coherent document.
    """
    def __init__(self, llm_client, embedding_model, partitioner):
        """
        Initializes the generator.
        
        Args:
            llm_client: The primary LLM for reasoning and generation.
            embedding_model: A model for creating text embeddings.
            partitioner: The PartitionManager to extract initial nodes.
        """
        self.llm = llm_client
        self.embedding_model = embedding_model
        self.partitioner = partitioner
        self.nodes = {}
        self.som = None
        self.som_grid = {} # Maps (x,y) coordinates to list of node_ids

    def _extract_nodes(self, document_text: str):
        """Pass 1: Deconstruct the tape into nodes with summaries."""
        print("Step 1: Extracting semantic nodes...")
        # Use the partitioner to get initial segments
        segments = self.partitioner.create_partitions(document_text)
        
        # For now, we'll use the segments directly as nodes.
        # A more advanced implementation could add an LLM summary step here.
        for i, segment_content in enumerate(segments):
            node_id = f"node_{i}"
            self.nodes[node_id] = {
                'content': segment_content,
                'summary': segment_content[:150] + "..." # Simple summary for now
            }
        print(f"  -> Extracted {len(self.nodes)} nodes.")

    def _train_som(self, map_size=(10, 10)):
        """Pass 2: Train the SOM on the node embeddings."""
        print("Step 2: Training the Self-Organizing Map...")
        if not self.nodes:
            print("  -> No nodes to train on. Skipping.")
            return

        node_ids = list(self.nodes.keys())
        summaries = [self.nodes[nid]['summary'] for nid in node_ids]
        
        # Get semantic "fingerprints" for each node
        data = self.embedding_model.encode(summaries)
        input_len = data.shape[1]

        # Initialize and train the SOM
        self.som = MiniSom(map_size[0], map_size[1], input_len, sigma=1.5, learning_rate=0.5)
        self.som.random_weights_init(data)
        self.som.train_batch(data, 1000, verbose=True)
        print("  -> SOM training complete.")

    def _map_nodes_to_grid(self):
        """Pass 3: Place each node onto its final position on the map."""
        print("Step 3: Mapping nodes to the 2D semantic grid...")
        if self.som is None:
            print("  -> SOM not trained. Skipping.")
            return

        self.som_grid = {}
        node_ids = list(self.nodes.keys())
        summaries = [self.nodes[nid]['summary'] for nid in node_ids]
        embeddings = self.embedding_model.encode(summaries)

        for i, embedding in enumerate(embeddings):
            node_id = node_ids[i]
            # Find the Best Matching Unit (BMU) for this node
            winner_coords = self.som.winner(embedding)
            if winner_coords not in self.som_grid:
                self.som_grid[winner_coords] = []
            self.som_grid[winner_coords].append(node_id)
        print("  -> Node mapping complete.")

    def _find_narrative_path(self) -> list:
        """Pass 4: Use the LLM to chart a coherent path through the map."""
        print("Step 4: Finding a narrative path through the conceptual map...")
        if not self.som_grid:
            print("  -> No grid to find a path in. Skipping.")
            return []
            
        # Create a "map summary" for the LLM
        map_summary = []
        for coords, node_ids in self.som_grid.items():
            concepts = [self.nodes[nid]['summary'] for nid in node_ids]
            map_summary.append(f"Region at {coords}: Contains ideas about '{', '.join(concepts)}'")

        prompt = f"""
        You are a master storyteller and curriculum designer. Below is a summary of a 2D map of concepts.
        Your task is to create a logical reading order—a path that a person should follow to understand the ideas.
        Start with the most foundational topic and move logically to more complex or related topics.
        
        CONCEPTUAL MAP:
        {"\n".join(map_summary)}
        
        Return a JSON list of coordinates representing the optimal path. For example: [[0,1], [0,2], [1,2], ...].
        
        OPTIMAL PATH (JSON):
        """
        
        try:
            # Use the generate_text method from the OllamaClient
            response = self.llm.generate_text(prompt)
            # The response might be a string that needs parsing, or already a JSON object
            # depending on the client. Assuming it's a string for now.
            path_coords = json.loads(response)
        except Exception as e:
            logger.error(f"LLM failed to generate a narrative path: {e}")
            # Fallback to a simple grid traversal
            path_coords = sorted(self.som_grid.keys())

        # Unroll the coordinates into a flat list of node IDs
        generation_plan = []
        for coords in path_coords:
            node_ids_in_region = self.som_grid.get(tuple(coords), [])
            # Here we could even ask the LLM to order the nodes *within* a region
            generation_plan.extend(node_ids_in_region)
        print(f"  -> Narrative path established with {len(generation_plan)} nodes.")
        return generation_plan

    def assemble_document(self, document_text: str) -> str:
        """Runs the full pipeline and generates the final document."""
        self._extract_nodes(document_text)
        self._train_som()
        self._map_nodes_to_grid()
        
        generation_plan = self._find_narrative_path()
        
        # Pass 5: Autoregressive Assembly using the discovered path
        print("Step 5: Assembling final document via Autoregressive Scaffolding...")
        final_tape = ["# Self-Organized Document\n"]
        
        # A simple assembly for now. A more advanced version would use the LLM
        # to generate smooth transitions between nodes.
        for node_id in generation_plan:
            final_tape.append(self.nodes[node_id]['content'])
            
        print("  -> Document assembly complete.")
        return "\n\n---\n\n".join(final_tape)
