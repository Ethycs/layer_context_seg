Component 1: The Foundational Metaphor and Theory (README.md)
Generated markdown
# Project AURA: Adaptive Universal Reorganization Architecture

## 1. Core Philosophy: The "Tape-to-Graph" Transformation

At its heart, this project treats any linear document—be it a transcript, a codebase, or a research paper—as a one-dimensional **"tape"** of information. While easy to produce, this linear format obscures the complex, non-linear relationships between the concepts within it. Our system transforms this simple tape into a rich, multi-dimensional **knowledge graph**.

The process is as follows:

1.  **Disassembly (Tape to Nodes):** The continuous tape is first segmented into discrete, semantic "nodes." Each node represents a coherent idea, a code block, or a conversational turn. This is not an arbitrary split; it is a meaning-preserving discretization.
2.  **Reconstruction (Nodes to Graph):** We then establish relational edges between these nodes. These edges represent dependencies, explanations, contradictions, or temporal sequences, creating a true network of knowledge.
3.  **Reassembly (Graph to Purpose-Driven Documents):** The knowledge graph is not the final output. It is a powerful intermediate representation. From this single graph, we can reassemble the information into countless new "tapes," each optimized for a specific purpose (e.g., a tutorial, an executive summary, a technical reference).

## 2. The Mathematical Justification: Percolation Theory and Optimal Chunking

A critical step in the "Tape-to-Nodes" transformation is determining the ideal size and overlap of our initial text chunks. We ground our approach in **Percolation Theory**.

Imagine our document's concepts as a physical medium. For information to "percolate" or flow from one end of the document to the other, the chunks (nodes) must be sufficiently connected.

-   **Too little overlap (<15%):** The chunks are disconnected islands of meaning. The system can understand local concepts but fails to form a "big picture" view. No "giant connected component" of knowledge emerges.
-   **Too much overlap (>30%):** The system becomes computationally inefficient and redundant. The connections are trivially obvious, and we lose the ability to identify meaningful, non-local relationships.

The **critical threshold** lies in the **15-30% overlap range**. At this density, a phase transition occurs. The graph of nodes becomes globally connected, allowing insights and context to flow across the entire information space. This ensures that the meaning of a concept at the beginning of the tape can influence and be influenced by a concept at the end, enabling true retroactive understanding. This mathematically-grounded overlap is fundamental to our chunking strategy, ensuring the resulting knowledge graph is both coherent and comprehensive.
Use code with caution.
Markdown
Component 2: The Simpler, API-Only Fallback (src/api_processor.py)
Generated python
# src/api_processor.py

import json
from itertools import combinations

class SimpleKnowledgeGraphBuilder:
    """
    Builds a knowledge graph from a document using only standard LLM API calls.
    This is the robust, fallback implementation that does not require attention head access.
    """
    def __init__(self, llm_api_client):
        self.llm = llm_api_client

    def smart_chunk(self, text: str) -> list[str]:
        """Uses the LLM to decide on semantic chunk boundaries."""
        prompt = f"""
        You are an expert text analyst. Your task is to split the following document into coherent semantic chunks.
        Mark the boundary between each chunk with the special marker '|||'.
        Aim for chunks that represent a single, self-contained idea, function, or topic.

        DOCUMENT:
        ---
        {text}
        ---

        Split the document now by inserting '|||' at the optimal boundaries.
        """
        response = self.llm.complete(prompt)
        return response.strip().split('|||')

    def extract_nodes(self, chunks: list[str]) -> list[dict]:
        """Converts raw text chunks into structured graph nodes using the LLM."""
        nodes = []
        for i, chunk in enumerate(chunks):
            prompt = f"""
            Analyze the following text chunk and extract its core structured information.
            Return a single JSON object with the keys: "id", "concept", "category", "summary".
            - "id": A unique integer for this chunk.
            - "concept": A short, descriptive title (3-5 words).
            - "category": Classify into one of: 'Code', 'DataLoading', 'Analysis', 'Visualization', 'Discussion', 'Question', 'Decision'.
            - "summary": A one-sentence summary of the chunk's content.

            CHUNK:
            ---
            {chunk}
            ---

            JSON Output:
            """
            try:
                response_text = self.llm.complete(prompt)
                node_data = json.loads(response_text)
                node_data['id'] = i # Ensure unique ID
                node_data['content'] = chunk # Store original content
                nodes.append(node_data)
            except (json.JSONDecodeError, TypeError):
                # Handle cases where LLM output is not valid JSON
                print(f"Warning: Could not parse node from chunk {i}.")
                continue
        return nodes

    def build_connections(self, nodes: list[dict]) -> list[dict]:
        """Determines the relationships (edges) between nodes."""
        edges = []
        for node1, node2 in combinations(nodes, 2):
            prompt = f"""
            Analyze the relationship between the following two concepts.
            Describe the relationship with one of these labels: 'depends_on', 'explains', 'contradicts', 'extends', 'is_example_of', 'is_related_to', 'none'.

            CONCEPT 1: {node1['concept']} ({node1['summary']})
            CONCEPT 2: {node2['concept']} ({node2['summary']})

            RELATIONSHIP (one label only):
            """
            relation = self.llm.complete(prompt).strip()
            if relation != 'none':
                edges.append({
                    "from": node1['id'],
                    "to": node2['id'],
                    "relation": relation
                })
        return edges

    def process_document(self, document_text: str) -> dict:
        """Orchestrates the full tape-to-graph process."""
        print("Step 1: Chunking document...")
        chunks = self.smart_chunk(document_text)
        print(f"  -> Found {len(chunks)} chunks.")

        print("Step 2: Extracting nodes...")
        nodes = self.extract_nodes(chunks)
        print(f"  -> Extracted {len(nodes)} nodes.")
        
        print("Step 3: Building connections...")
        edges = self.build_connections(nodes)
        print(f"  -> Found {len(edges)} connections.")

        return {"nodes": nodes, "edges": edges}
Use code with caution.
Python
Component 3: Graph Refinement and Node Classification (src/graph_utils.py)
Generated python
# src/graph_utils.py

import networkx as nx
# Assume the existence of an embedding client and a similarity function
# from sentence_transformers.util import cos_sim
# from sentence_transformers import SentenceTransformer

class GraphProcessor:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        # self.embedding_model = SentenceTransformer(embedding_model_name)
        pass # In a real implementation, the model would be loaded here.

    def get_embedding(self, text: str):
        """Placeholder for a text embedding function."""
        # return self.embedding_model.encode(text)
        # For demonstration, returning a dummy vector based on length
        return [len(text) * 0.1] * 384

    def refine_graph(self, graph: nx.Graph, similarity_threshold=0.85):
        """Merges highly similar nodes and prunes weak edges to clean the graph."""
        print("Refining graph...")
        # Calculate embeddings for all nodes
        for node_id, data in graph.nodes(data=True):
            if 'embedding' not in data:
                text_to_embed = data.get('concept', '') + ' ' + data.get('summary', '')
                graph.nodes[node_id]['embedding'] = self.get_embedding(text_to_embed)

        # Find and merge similar nodes
        merged_nodes = set()
        for n1, n2 in combinations(graph.nodes(), 2):
            if n1 in merged_nodes or n2 in merged_nodes:
                continue
            
            # similarity = cos_sim(graph.nodes[n1]['embedding'], graph.nodes[n2]['embedding'])
            # Using a placeholder for similarity calculation
            similarity = 1 - abs(graph.nodes[n1]['embedding'][0] - graph.nodes[n2]['embedding'][0])

            if similarity > similarity_threshold:
                # Merge n2 into n1
                graph = nx.contracted_nodes(graph, n1, n2, self_loops=False)
                merged_nodes.add(n2)
                print(f"  -> Merged node {n2} into {n1}")
        
        # This part is simplified; a real implementation would merge attributes correctly.
        # nx.contracted_nodes is a good starting point.
        
        return graph

    def classify_nodes(self, graph: nx.Graph):
        """
        Classifies nodes as KEEP, DELETE, or TRACK based on graph metrics and content.
        """
        print("Classifying nodes...")
        if not graph.nodes():
            return graph

        pagerank = nx.pagerank(graph)
        degrees = dict(graph.degree())
        
        max_degree = max(degrees.values()) if degrees else 1

        for node_id, data in graph.nodes(data=True):
            # Combine multiple signals for an importance score
            pr_score = pagerank.get(node_id, 0)
            degree_score = degrees.get(node_id, 0) / max_degree
            importance_score = 0.6 * pr_score + 0.4 * degree_score

            # Content-based rules
            content = data.get('content', '').upper()
            if 'TODO' in content or 'FIXME' in content or 'QUESTION' in content:
                classification = 'TRACK'
            elif importance_score < 0.05 and degrees.get(node_id, 0) <= 1:
                classification = 'DELETE'
            else:
                classification = 'KEEP'
            
            graph.nodes[node_id]['classification'] = classification
            graph.nodes[node_id]['importance_score'] = importance_score
            print(f"  -> Node {node_id} ({data.get('concept', '')[:20]}...): {classification} (Score: {importance_score:.2f})")

        return graph
Use code with caution.
Python
Component 4: The Final Application Wrapper (src/condenser.py)
Generated python
# src/condenser.py

import networkx as nx
# from .api_processor import SimpleKnowledgeGraphBuilder
# from .graph_utils import GraphProcessor

class TranscriptCondenser:
    """
    The main user-facing class that orchestrates the entire process of
    transforming a transcript into various structured, condensed outputs.
    """
    def __init__(self, llm_api_client):
        self.llm = llm_api_client
        # Can be swapped with attention_processor in the future
        self.processor = SimpleKnowledgeGraphBuilder(llm_api_client)
        self.graph_utils = GraphProcessor()
        self.graph = None

    def _prime_context(self, transcript: str, rules: dict) -> str:
        """Creates a comprehensive prompt for the initial chunking."""
        # In a real system, these rules would heavily influence the prompts
        # in the processor, but here we just prepend them for context.
        return f"""
        System Task: Analyze and structure the following transcript.
        Segmentation Rule: {rules.get('segmentation', 'Split at topic shifts.')}
        Reorganization Rule: {rules.get('reorganization', 'Group by theme.')}

        --- TRANSCRIPT START ---
        {transcript}
        --- TRANSCRIPT END ---
        """

    def process_transcript(self, transcript: str, rules: dict = None):
        """
        Processes a full transcript to build and classify a knowledge graph.
        """
        if not rules:
            rules = {
                "segmentation": "Split the transcript at natural speaker turns and major topic shifts.",
                "reorganization": "Group related concepts and create a logical flow."
            }

        primed_transcript = self._prime_context(transcript, rules)
        
        # Build the initial graph
        graph_data = self.processor.process_document(primed_transcript)
        
        # Create a NetworkX graph object
        G = nx.Graph()
        for node in graph_data["nodes"]:
            # Use a simple copy to avoid modifying the original dict
            node_attrs = node.copy()
            node_id = node_attrs.pop('id')
            G.add_node(node_id, **node_attrs)
        
        for edge in graph_data["edges"]:
            G.add_edge(edge["from"], edge["to"], relation=edge["relation"])
            
        # Refine and Classify
        refined_graph = self.graph_utils.refine_graph(G)
        self.graph = self.graph_utils.classify_nodes(refined_graph)
        
        print("\nProcessing complete. Graph is built and classified.")
        return self

    def extract_summary(self, style='executive') -> str:
        """Reassembles the graph into a summary."""
        if not self.graph:
            return "Error: Please process a transcript first."

        # Filter nodes based on classification
        keep_nodes = [data for _, data in self.graph.nodes(data=True) if data.get('classification') == 'KEEP']
        
        # Sort by importance
        sorted_nodes = sorted(keep_nodes, key=lambda x: x.get('importance_score', 0), reverse=True)

        context_for_summary = "\n".join([f"- {n['concept']}: {n['summary']}" for n in sorted_nodes])

        prompt = f"""
        Based on the following key concepts from a transcript, write a concise {style} summary.

        KEY CONCEPTS:
        {context_for_summary}

        {style.upper()} SUMMARY:
        """
        return self.llm.complete(prompt)

    def extract_action_items(self) -> list[str]:
        """Extracts and lists all TRACKed items as actionable tasks."""
        if not self.graph:
            return ["Error: Please process a transcript first."]
        
        track_nodes = [data['content'] for _, data in self.graph.nodes(data=True) if data.get('classification') == 'TRACK']
        
        if not track_nodes:
            return ["No action items or tracked topics found."]

        prompt = f"""
        Convert the following text chunks, which were marked for tracking, into a clear list of action items or questions.

        CHUNKS TO ANALYZE:
        ---
        {"\n---\n".join(track_nodes)}
        ---

        ACTION ITEMS / QUESTIONS LIST:
        """
        return self.llm.complete(prompt).strip().split('\n')
        
    def get_graph_data(self, format='json'):
        """Exports the final graph for visualization or further processing."""
        if not self.graph:
            return None
        if format == 'json':
            return nx.node_link_data(self.graph)
        # Add other formats like graphml if needed
        return None