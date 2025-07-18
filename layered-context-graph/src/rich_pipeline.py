import logging
import json
from pathlib import Path
import networkx as nx
from partitioning.partition_manager import PartitionManager

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
RESULTS_DIR = Path('./results')
RESULTS_DIR.mkdir(exist_ok=True)

def run_rich_pipeline(text: str, k_rules: list, reassembly_prompt: str, output_key: str):
    """
    Executes the full, stateful pipeline using the PartitionManager.
    """
    logger.info("--- Starting Rich Pipeline ---")

    # 1. Instantiate the manager. Models will be lazy-loaded inside it.
    manager = PartitionManager()

    # 2. Partition the text into a hierarchical graph.
    logger.info("Phase 1: Partitioning...")
    manager.partition(text, k_rules)

    # 3. Classify the nodes in the graph.
    logger.info("Phase 2: Classifying nodes...")
    manager.classify()

    # 4. Reassemble the graph into a new text format.
    logger.info("Phase 3: Reassembling text...")
    reassembled_text = manager.reassemble(reassembly_prompt, key=output_key)

    # 5. Save the outputs.
    logger.info("Phase 4: Saving results...")
    
    # Save graph to a JSON format (GraphML for structure, JSON for node data)
    graph_path = RESULTS_DIR / f"{output_key}_graph.json"
    graph_data = nx.node_link_data(manager.graph)
    # Convert EnrichedSegment objects to dictionaries for serialization
    for node in graph_data['nodes']:
        node['segment'] = node['segment'].__dict__
    with open(graph_path, 'w') as f:
        json.dump(graph_data, f, indent=2)
    logger.info(f"Graph saved to {graph_path}")

    # Save reassembled text
    text_path = RESULTS_DIR / f"{output_key}_reassembled.txt"
    with open(text_path, 'w') as f:
        f.write(reassembled_text)
    logger.info(f"Reassembled text saved to {text_path}")

    logger.info("--- Rich Pipeline Finished ---")
    return graph_data, reassembled_text

if __name__ == '__main__':
    # This is the main entry point for running the pipeline from the command line.
    
    # Load the source text
    source_text_path = Path('./demo_content/physics_paper.txt')
    if not source_text_path.exists():
        raise FileNotFoundError(f"Source text not found at {source_text_path}")
    with open(source_text_path, 'r') as f:
        source_text = f.read()

    # Define the K-Rules for segmentation
    segmentation_rules = [
        "Split the document into its major sections like Introduction, Main Body, and Conclusion.",
        "Break down each section into paragraphs.",
        "Isolate code blocks and mathematical formulas.",
        "Divide long paragraphs into individual sentences."
    ]

    # Define the prompt for the final reassembly
    synthesis_prompt = "Create a concise, easy-to-read summary of the key findings from the provided text."

    # Run the pipeline
    run_rich_pipeline(
        text=source_text,
        k_rules=segmentation_rules,
        reassembly_prompt=synthesis_prompt,
        output_key='physics_summary'
    )
