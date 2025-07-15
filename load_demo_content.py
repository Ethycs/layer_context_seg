#!/usr/bin/env python3
"""
Demo content loader - loads demo content from files
"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def get_demo_content(demo_type: str) -> str:
    """Get demo content from files"""
    # Special case for loading the full layered context file
    if demo_type == 'layered_context_file':
        try:
            with open('Layered_Context_Window_Graphs_beefa8c4_2025-07-13T03-26-26-136Z.txt', 'r') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Could not load layered context file: {e}")
            demo_type = 'layered_context'
    
    # Load demo content from files
    demo_file_path = Path(__file__).parent / 'demo_content' / f'{demo_type}.txt'
    
    try:
        with open(demo_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            logger.info(f"Loaded demo content from {demo_file_path}")
            return content
    except FileNotFoundError:
        logger.error(f"Demo file not found: {demo_file_path}")
        return f"Demo content file not found: {demo_type}.txt"
    except Exception as e:
        logger.error(f"Error loading demo content: {e}")
        return f"Error loading demo content: {str(e)}"