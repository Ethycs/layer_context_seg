#!/usr/bin/env python3
"""
Enhanced Tape Synthesizer - Coherent Content Generation
======================================================
This module performs Graph → Tape₂ transformation by rewriting
content into coherent, structured documents using seed prompts.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class EnhancedTapeSynthesizer:
    """Synthesizes coherent documents from knowledge graphs"""
    
    def __init__(self):
        self.synthesis_strategies = {
            'executive_summary': self._synthesize_executive_summary,
            'tutorial': self._synthesize_tutorial,
            'reference': self._synthesize_reference,
            'readme': self._synthesize_readme
        }
    
    def synthesize(self, graph_data: Dict, strategy: str = 'executive_summary') -> Dict[str, Any]:
        """
        Generate coherent synthesized content from the knowledge graph
        
        This creates properly structured documents by:
        1. Analyzing the graph structure and content
        2. Creating a coherent narrative structure
        3. Rewriting node content into flowing prose
        4. Maintaining semantic relationships while improving readability
        """
        
        if strategy not in self.synthesis_strategies:
            available = list(self.synthesis_strategies.keys())
            raise ValueError(f"Unknown strategy: {strategy}. Available: {available}")
        
        # Extract components
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])
        original_text = graph_data.get('original_text', '')
        
        # Apply synthesis strategy with proper rewriting
        synthesized_content = self.synthesis_strategies[strategy](nodes, edges, original_text)
        
        return {
            'tape2': synthesized_content,
            'strategy': strategy,
            'synthesis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'nodes_used': len(nodes),
                'edges_used': len(edges),
                'original_length': len(original_text),
                'synthesized_length': len(synthesized_content),
                'compression_ratio': len(original_text) / len(synthesized_content) if synthesized_content else 0,
                'method': 'coherent_rewriting'
            }
        }
    
    def _synthesize_executive_summary(self, nodes: List[Dict], edges: List[Dict], original: str) -> str:
        """Generate a coherent executive summary"""
        
        # Filter and sort nodes by importance
        important_nodes = sorted(
            nodes, 
            key=lambda n: n.get('importance', 0), 
            reverse=True
        )[:10]
        
        # Group nodes by type for logical flow
        foundational = [n for n in important_nodes if n.get('segment_type') == 'foundational']
        technical = [n for n in important_nodes if n.get('segment_type') == 'technical_core']
        applications = [n for n in important_nodes if n.get('segment_type') == 'application']
        
        # Build coherent summary
        summary = []
        summary.append("# Executive Summary\n")
        
        # Opening paragraph - synthesize main theme
        if foundational:
            main_concepts = self._extract_key_concepts(foundational)
            summary.append(f"This analysis examines {main_concepts}, revealing a sophisticated system of interconnected components and principles. ")
            summary.append("The following summary distills the key insights and relationships discovered through comprehensive knowledge graph analysis.\n\n")
        else:
            summary.append("This document presents a comprehensive analysis of the core concepts and their relationships as revealed through systematic knowledge extraction.\n\n")
        
        # Core findings section
        summary.append("## Core Findings\n\n")
        
        # Synthesize foundational concepts into flowing prose
        if foundational:
            foundation_text = self._synthesize_paragraph(
                foundational,
                "The foundational analysis reveals",
                focus="principles"
            )
            summary.append(foundation_text + "\n\n")
        
        # Technical insights
        if technical:
            summary.append("## Technical Architecture\n\n")
            tech_text = self._synthesize_paragraph(
                technical,
                "The technical implementation demonstrates",
                focus="mechanisms"
            )
            summary.append(tech_text + "\n\n")
        
        # Practical applications
        if applications:
            summary.append("## Applications and Impact\n\n")
            app_text = self._synthesize_paragraph(
                applications,
                "Practical applications include",
                focus="uses"
            )
            summary.append(app_text + "\n\n")
        
        # Key relationships
        summary.append("## Key Relationships\n\n")
        relationship_text = self._analyze_relationships(nodes, edges)
        summary.append(relationship_text + "\n\n")
        
        # Conclusions and recommendations
        summary.append("## Conclusions\n\n")
        conclusions = self._synthesize_conclusions(important_nodes, edges)
        summary.append(conclusions + "\n")
        
        # Metadata footer
        summary.append(f"\n---\n*Synthesized from {len(nodes)} knowledge components with {len(edges)} identified relationships.*")
        
        return ''.join(summary)
    
    def _synthesize_tutorial(self, nodes: List[Dict], edges: List[Dict], original: str) -> str:
        """Generate a coherent step-by-step tutorial"""
        
        # Organize nodes for tutorial flow
        foundational = [n for n in nodes if n.get('segment_type') == 'foundational']
        technical = [n for n in nodes if n.get('segment_type') in ['technical_core', 'implementation']]
        examples = [n for n in nodes if n.get('segment_type') == 'illustrative']
        
        tutorial = []
        tutorial.append("# Comprehensive Tutorial\n\n")
        
        # Introduction
        tutorial.append("## Introduction\n\n")
        intro_text = self._create_tutorial_intro(foundational, technical)
        tutorial.append(intro_text + "\n\n")
        
        # Prerequisites
        if foundational:
            tutorial.append("## Prerequisites\n\n")
            tutorial.append("Before beginning this tutorial, you should understand:\n\n")
            prereqs = self._extract_prerequisites(foundational)
            for prereq in prereqs:
                tutorial.append(f"- {prereq}\n")
            tutorial.append("\n")
        
        # Learning objectives
        tutorial.append("## Learning Objectives\n\n")
        objectives = self._generate_learning_objectives(nodes)
        tutorial.append("By the end of this tutorial, you will:\n\n")
        for obj in objectives:
            tutorial.append(f"- {obj}\n")
        tutorial.append("\n")
        
        # Step-by-step content
        tutorial.append("## Step-by-Step Guide\n\n")
        
        # Generate logical steps from node relationships
        steps = self._organize_tutorial_steps(technical, edges)
        
        for i, step_nodes in enumerate(steps, 1):
            tutorial.append(f"### Step {i}: {self._generate_step_title(step_nodes)}\n\n")
            
            # Synthesize step content
            step_content = self._synthesize_step_content(step_nodes, examples)
            tutorial.append(step_content + "\n\n")
            
            # Add relevant example if available
            relevant_examples = self._find_relevant_examples(step_nodes, examples, edges)
            if relevant_examples:
                tutorial.append("**Example:**\n\n")
                example_text = self._format_example(relevant_examples[0])
                tutorial.append(example_text + "\n\n")
        
        # Summary and next steps
        tutorial.append("## Summary\n\n")
        summary_text = self._create_tutorial_summary(nodes)
        tutorial.append(summary_text + "\n\n")
        
        tutorial.append("## Next Steps\n\n")
        next_steps = self._suggest_next_steps(nodes)
        tutorial.append(next_steps + "\n")
        
        return ''.join(tutorial)
    
    def _synthesize_reference(self, nodes: List[Dict], edges: List[Dict], original: str) -> str:
        """Generate comprehensive reference documentation"""
        
        reference = []
        reference.append("# Reference Documentation\n\n")
        
        # Create comprehensive overview
        reference.append("## Overview\n\n")
        overview = self._create_reference_overview(nodes, edges)
        reference.append(overview + "\n\n")
        
        # Table of contents based on actual content
        reference.append("## Table of Contents\n\n")
        toc = self._generate_table_of_contents(nodes)
        for item in toc:
            reference.append(f"- {item}\n")
        reference.append("\n")
        
        # Group nodes by category
        categories = self._categorize_nodes(nodes)
        
        # Generate sections for each category
        for category, cat_nodes in categories.items():
            if not cat_nodes:
                continue
                
            reference.append(f"## {self._format_category_title(category)}\n\n")
            
            # Category overview
            cat_overview = self._synthesize_category_overview(category, cat_nodes)
            reference.append(cat_overview + "\n\n")
            
            # Detailed entries
            for node in sorted(cat_nodes, key=lambda n: n.get('importance', 0), reverse=True):
                entry_title = self._generate_entry_title(node)
                reference.append(f"### {entry_title}\n\n")
                
                # Synthesize entry content
                entry_content = self._synthesize_reference_entry(node, nodes, edges)
                reference.append(entry_content + "\n\n")
        
        # Cross-reference section
        reference.append("## Cross-References\n\n")
        cross_refs = self._generate_cross_references(nodes, edges)
        reference.append(cross_refs + "\n\n")
        
        # Index
        reference.append("## Index\n\n")
        index = self._generate_index(nodes)
        reference.append(index + "\n")
        
        return ''.join(reference)
    
    def _synthesize_readme(self, nodes: List[Dict], edges: List[Dict], original: str) -> str:
        """Generate a project README"""
        
        readme = []
        readme.append("# Project Documentation\n\n")
        
        # Extract project information
        project_info = self._extract_project_info(nodes)
        
        # Project description
        readme.append("## Description\n\n")
        description = self._synthesize_project_description(nodes)
        readme.append(description + "\n\n")
        
        # Features
        features = self._extract_features(nodes)
        if features:
            readme.append("## Features\n\n")
            for feature in features:
                readme.append(f"- {feature}\n")
            readme.append("\n")
        
        # Installation
        readme.append("## Installation\n\n")
        install_steps = self._extract_installation_steps(nodes)
        if install_steps:
            readme.append("```bash\n")
            for step in install_steps:
                readme.append(f"{step}\n")
            readme.append("```\n\n")
        else:
            readme.append("See documentation for installation instructions.\n\n")
        
        # Usage
        readme.append("## Usage\n\n")
        usage_info = self._synthesize_usage_info(nodes)
        readme.append(usage_info + "\n\n")
        
        # Architecture
        arch_nodes = [n for n in nodes if 'architecture' in n.get('content', '').lower()]
        if arch_nodes:
            readme.append("## Architecture\n\n")
            arch_text = self._synthesize_architecture_section(arch_nodes)
            readme.append(arch_text + "\n\n")
        
        # Contributing
        readme.append("## Contributing\n\n")
        readme.append("Contributions are welcome! Please see our contributing guidelines for more information.\n\n")
        
        # License
        readme.append("## License\n\n")
        readme.append("This project is licensed under the MIT License - see the LICENSE file for details.\n\n")
        
        # Acknowledgments
        readme.append("## Acknowledgments\n\n")
        acknowledgments = self._generate_acknowledgments(nodes)
        readme.append(acknowledgments + "\n")
        
        return ''.join(readme)
    
    # Helper methods for coherent text generation
    
    def _extract_key_concepts(self, nodes: List[Dict]) -> str:
        """Extract and format key concepts from nodes"""
        concepts = []
        for node in nodes[:3]:  # Top 3 concepts
            content = node.get('content', '')
            # Extract first significant phrase
            if '. ' in content:
                concept = content.split('. ')[0].strip()
            else:
                concept = content[:100].strip()
            concepts.append(concept.lower())
        
        if len(concepts) > 1:
            return ', '.join(concepts[:-1]) + ' and ' + concepts[-1]
        elif concepts:
            return concepts[0]
        else:
            return "the core concepts"
    
    def _synthesize_paragraph(self, nodes: List[Dict], intro: str, focus: str) -> str:
        """Synthesize nodes into a coherent paragraph"""
        if not nodes:
            return f"{intro} no significant findings in this area."
        
        # Extract key points from each node
        points = []
        for node in nodes[:5]:  # Limit to 5 for readability
            content = node.get('content', '')
            point = self._extract_key_point(content, focus)
            if point:
                points.append(point)
        
        # Combine into flowing prose
        paragraph = f"{intro} "
        
        if len(points) == 1:
            paragraph += points[0] + "."
        elif len(points) == 2:
            paragraph += f"{points[0]}, and {points[1]}."
        else:
            # Use varied connectors for better flow
            connectors = ["", "Additionally, ", "Furthermore, ", "Moreover, ", "This includes "]
            for i, point in enumerate(points[:5]):
                if i < len(connectors):
                    paragraph += connectors[i] + point
                    if i < len(points) - 1:
                        paragraph += ". "
                    else:
                        paragraph += "."
        
        return paragraph
    
    def _extract_key_point(self, content: str, focus: str) -> str:
        """Extract a key point from content based on focus"""
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        # Look for sentences matching focus
        focus_keywords = {
            'principles': ['principle', 'fundamental', 'core', 'basis', 'foundation'],
            'mechanisms': ['implementation', 'process', 'method', 'technique', 'approach'],
            'uses': ['application', 'use', 'enables', 'allows', 'provides']
        }
        
        keywords = focus_keywords.get(focus, [])
        
        # Find best matching sentence
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                return self._clean_sentence(sentence)
        
        # Fallback to first meaningful sentence
        for sentence in sentences:
            if len(sentence) > 20:
                return self._clean_sentence(sentence)
        
        return self._clean_sentence(content[:100])
    
    def _clean_sentence(self, sentence: str) -> str:
        """Clean and format a sentence for inclusion"""
        # Remove special markers
        for marker in ['<QWQ_', '<MATH>', '<DIALOGUE>', '```']:
            sentence = sentence.replace(marker, '')
        
        # Clean up whitespace
        sentence = ' '.join(sentence.split())
        
        # Ensure proper ending
        if sentence and not sentence[-1] in '.!?':
            sentence += '.'
        
        return sentence.strip()
    
    def _analyze_relationships(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """Analyze and describe key relationships"""
        if not edges:
            return "The analysis reveals minimal explicit relationships between components, suggesting either independent concepts or implicit connections requiring further investigation."
        
        # Count relationship types
        relationship_types = {}
        for edge in edges:
            edge_type = edge.get('type', 'connection')
            relationship_types[edge_type] = relationship_types.get(edge_type, 0) + 1
        
        # Build description
        total_relationships = len(edges)
        descriptions = []
        
        descriptions.append(f"The knowledge graph reveals {total_relationships} significant relationships between concepts")
        
        # Describe major relationship types
        major_types = sorted(relationship_types.items(), key=lambda x: x[1], reverse=True)[:3]
        
        if major_types:
            type_descriptions = []
            for rel_type, count in major_types:
                percentage = (count / total_relationships) * 100
                type_descriptions.append(f"{self._humanize_relationship_type(rel_type)} ({percentage:.0f}%)")
            
            descriptions.append(f", primarily consisting of {', '.join(type_descriptions)}")
        
        descriptions.append(". ")
        
        # Add insight about connectivity
        avg_connections = (len(edges) * 2) / len(nodes) if nodes else 0
        if avg_connections > 3:
            descriptions.append("The high degree of interconnectedness suggests a tightly integrated system where components significantly influence each other.")
        elif avg_connections > 1.5:
            descriptions.append("The moderate connectivity indicates a balanced system with both independent and interdependent components.")
        else:
            descriptions.append("The sparse connectivity suggests relatively independent components with focused interactions.")
        
        return ''.join(descriptions)
    
    def _humanize_relationship_type(self, rel_type: str) -> str:
        """Convert relationship type to human-readable form"""
        type_map = {
            'sequential': 'sequential dependencies',
            'reference': 'conceptual references',
            'elaboration': 'detailed elaborations',
            'contrast': 'contrasting viewpoints',
            'example': 'illustrative examples',
            'dependency': 'functional dependencies'
        }
        return type_map.get(rel_type, rel_type.replace('_', ' '))
    
    def _synthesize_conclusions(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """Synthesize conclusions from the analysis"""
        conclusions = []
        
        # Analyze overall patterns
        node_types = {}
        for node in nodes:
            seg_type = node.get('segment_type', 'general')
            node_types[seg_type] = node_types.get(seg_type, 0) + 1
        
        # Generate insights based on distribution
        if 'foundational' in node_types and node_types['foundational'] > len(nodes) * 0.2:
            conclusions.append("The analysis reveals a strong theoretical foundation underlying the subject matter. ")
        
        if 'technical_core' in node_types and node_types['technical_core'] > len(nodes) * 0.3:
            conclusions.append("Technical implementation details dominate the knowledge structure, indicating a practice-oriented domain. ")
        
        if 'application' in node_types and node_types['application'] > len(nodes) * 0.15:
            conclusions.append("The presence of numerous practical applications demonstrates the real-world relevance and utility of these concepts. ")
        
        # Add recommendation
        if len(edges) > len(nodes) * 1.5:
            conclusions.append("\n\nThe dense network of relationships suggests that understanding this domain requires a holistic approach, ")
            conclusions.append("as concepts are highly interdependent and mutually reinforcing.")
        else:
            conclusions.append("\n\nThe modular structure of the knowledge graph indicates that concepts can be understood independently, ")
            conclusions.append("allowing for targeted learning and application.")
        
        return ''.join(conclusions)
    
    # Additional helper methods would continue here...
    # (Truncated for brevity, but would include all the referenced helper methods)
    
    def _create_tutorial_intro(self, foundational: List[Dict], technical: List[Dict]) -> str:
        """Create tutorial introduction"""
        intro = "This tutorial provides a comprehensive guide to understanding and implementing "
        
        if foundational:
            concepts = self._extract_key_concepts(foundational)
            intro += f"the concepts of {concepts}. "
        
        intro += "Through step-by-step instructions and practical examples, "
        intro += "you'll gain both theoretical understanding and hands-on experience."
        
        return intro
    
    def _extract_prerequisites(self, foundational: List[Dict]) -> List[str]:
        """Extract prerequisites from foundational nodes"""
        prereqs = []
        for node in foundational[:3]:
            content = node.get('content', '')
            if 'require' in content.lower() or 'need' in content.lower():
                prereq = self._extract_key_point(content, 'principles')
                if prereq:
                    prereqs.append(prereq)
        
        if not prereqs:
            prereqs = ["Basic understanding of the domain concepts"]
        
        return prereqs
    
    def _generate_learning_objectives(self, nodes: List[Dict]) -> List[str]:
        """Generate learning objectives from nodes"""
        objectives = []
        
        # Analyze node types to create objectives
        has_technical = any(n.get('segment_type') == 'technical_core' for n in nodes)
        has_practical = any(n.get('segment_type') == 'application' for n in nodes)
        has_examples = any(n.get('segment_type') == 'illustrative' for n in nodes)
        
        if has_technical:
            objectives.append("Understand the core technical concepts and their implementations")
        if has_practical:
            objectives.append("Apply the concepts to real-world scenarios")
        if has_examples:
            objectives.append("Work through practical examples and exercises")
        
        objectives.append("Build a comprehensive mental model of the subject matter")
        
        return objectives
    
    def _organize_tutorial_steps(self, technical: List[Dict], edges: List[Dict]) -> List[List[Dict]]:
        """Organize nodes into logical tutorial steps"""
        # Simple grouping - in practice would use graph algorithms
        steps = []
        
        # Group by complexity/dependencies
        step_size = max(1, len(technical) // 5)  # Aim for ~5 steps
        
        for i in range(0, len(technical), step_size):
            step_nodes = technical[i:i + step_size]
            if step_nodes:
                steps.append(step_nodes)
        
        return steps
    
    def _generate_step_title(self, nodes: List[Dict]) -> str:
        """Generate a title for a tutorial step"""
        if not nodes:
            return "Next Steps"
        
        # Use the most important node's content
        primary_node = max(nodes, key=lambda n: n.get('importance', 0))
        content = primary_node.get('content', '')
        
        # Extract action-oriented title
        if 'implement' in content.lower():
            return "Implementation"
        elif 'create' in content.lower():
            return "Creation"
        elif 'configure' in content.lower():
            return "Configuration"
        elif 'understand' in content.lower():
            return "Understanding Core Concepts"
        else:
            # Extract first few words
            words = content.split()[:4]
            return ' '.join(words).title()
    
    def _synthesize_step_content(self, nodes: List[Dict], examples: List[Dict]) -> str:
        """Synthesize content for a tutorial step"""
        content_parts = []
        
        # Main explanation
        main_content = self._synthesize_paragraph(
            nodes,
            "In this step, you'll learn to",
            focus="mechanisms"
        )
        content_parts.append(main_content)
        
        # Add specific instructions if available
        for node in nodes:
            node_content = node.get('content', '')
            if any(action in node_content.lower() for action in ['create', 'implement', 'configure', 'set up']):
                instruction = self._extract_instruction(node_content)
                if instruction:
                    content_parts.append(f"\n\n{instruction}")
                break
        
        return ''.join(content_parts)
    
    def _extract_instruction(self, content: str) -> str:
        """Extract actionable instruction from content"""
        sentences = content.split('.')
        for sentence in sentences:
            if any(verb in sentence.lower() for verb in ['create', 'implement', 'add', 'configure']):
                return "To do this: " + sentence.strip() + "."
        return ""
    
    def _find_relevant_examples(self, step_nodes: List[Dict], examples: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Find examples relevant to the current step"""
        relevant = []
        
        step_ids = {node['id'] for node in step_nodes}
        
        for example in examples:
            # Check if example is connected to any step node
            for edge in edges:
                if edge['source'] in step_ids and edge['target'] == example['id']:
                    relevant.append(example)
                    break
                elif edge['target'] in step_ids and edge['source'] == example['id']:
                    relevant.append(example)
                    break
        
        return relevant
    
    def _format_example(self, example_node: Dict) -> str:
        """Format an example for display"""
        content = example_node.get('content', '')
        
        # Look for code blocks
        if '```' in content:
            return content
        elif any(marker in content for marker in ['def ', 'function ', 'class ', '{']):
            # Likely code - format as code block
            return f"```\n{content}\n```"
        else:
            # Regular example text
            return content
    
    def _create_tutorial_summary(self, nodes: List[Dict]) -> str:
        """Create tutorial summary"""
        summary = "In this tutorial, you've explored "
        
        # Count what was covered
        node_types = {}
        for node in nodes:
            seg_type = node.get('segment_type', 'general')
            node_types[seg_type] = node_types.get(seg_type, 0) + 1
        
        covered = []
        if node_types.get('foundational', 0) > 0:
            covered.append("fundamental concepts")
        if node_types.get('technical_core', 0) > 0:
            covered.append("technical implementations")
        if node_types.get('application', 0) > 0:
            covered.append("practical applications")
        
        if covered:
            summary += ", ".join(covered) + ". "
        
        summary += "You now have the knowledge and tools to apply these concepts in your own work."
        
        return summary
    
    def _suggest_next_steps(self, nodes: List[Dict]) -> str:
        """Suggest next steps after tutorial"""
        suggestions = []
        
        suggestions.append("To continue your learning journey:\n\n")
        suggestions.append("1. **Practice**: Apply these concepts to your own projects\n")
        suggestions.append("2. **Explore**: Dive deeper into areas that interest you most\n")
        suggestions.append("3. **Share**: Contribute your insights back to the community\n")
        suggestions.append("4. **Extend**: Build upon these foundations with advanced techniques\n")
        
        return ''.join(suggestions)
    
    # Additional helper methods for complete synthesis
    
    def _create_reference_overview(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """Create reference documentation overview"""
        overview = "This reference documentation provides comprehensive coverage of "
        
        # Analyze content
        total_concepts = len(nodes)
        total_relationships = len(edges)
        
        node_types = {}
        for node in nodes:
            seg_type = node.get('segment_type', 'general')
            node_types[seg_type] = node_types.get(seg_type, 0) + 1
        
        overview += f"{total_concepts} interconnected concepts organized into logical categories. "
        overview += f"The documentation captures {total_relationships} relationships that define how these concepts interact and depend on each other. "
        
        # Highlight key areas
        if node_types:
            major_type = max(node_types.items(), key=lambda x: x[1])
            overview += f"The primary focus is on {self._format_category_title(major_type[0]).lower()}, "
            overview += f"which comprises {(major_type[1]/total_concepts)*100:.0f}% of the documented content."
        
        return overview
    
    def _generate_table_of_contents(self, nodes: List[Dict]) -> List[str]:
        """Generate table of contents from nodes"""
        toc = []
        
        # Get unique categories
        categories = set()
        for node in nodes:
            category = self._format_category_title(node.get('segment_type', 'general'))
            categories.add(category)
        
        # Sort categories logically
        category_order = [
            'Overview', 'Fundamental Concepts', 'Core Components',
            'Technical Details', 'Implementation', 'Applications',
            'Examples', 'Best Practices', 'Troubleshooting'
        ]
        
        for cat in category_order:
            if cat in categories:
                toc.append(cat)
        
        # Add any remaining categories
        for cat in sorted(categories - set(category_order)):
            toc.append(cat)
        
        return toc
    
    def _categorize_nodes(self, nodes: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize nodes by type"""
        categories = {}
        
        for node in nodes:
            seg_type = node.get('segment_type', 'general')
            if seg_type not in categories:
                categories[seg_type] = []
            categories[seg_type].append(node)
        
        return categories
    
    def _format_category_title(self, category: str) -> str:
        """Format category name for display"""
        title_map = {
            'foundational': 'Fundamental Concepts',
            'technical_core': 'Core Components',
            'implementation': 'Implementation Details',
            'application': 'Applications',
            'illustrative': 'Examples',
            'problem_solving': 'Troubleshooting',
            'general': 'General Information'
        }
        return title_map.get(category, category.replace('_', ' ').title())
    
    def _synthesize_category_overview(self, category: str, nodes: List[Dict]) -> str:
        """Create overview for a category section"""
        overview = f"This section covers {len(nodes)} "
        
        if category == 'foundational':
            overview += "fundamental concepts that form the theoretical foundation of the system."
        elif category == 'technical_core':
            overview += "core technical components that implement the primary functionality."
        elif category == 'implementation':
            overview += "detailed implementation aspects and technical specifications."
        elif category == 'application':
            overview += "practical applications and use cases."
        else:
            overview += f"items related to {self._format_category_title(category).lower()}."
        
        return overview
    
    def _generate_entry_title(self, node: Dict) -> str:
        """Generate title for a reference entry"""
        content = node.get('content', '')
        
        # Try to extract a natural title
        if '\n' in content:
            first_line = content.split('\n')[0].strip()
            if len(first_line) < 80 and not first_line.endswith('.'):
                return first_line
        
        # Extract key phrase
        sentences = content.split('.')
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) < 80:
                return first_sentence
        
        # Fallback
        words = content.split()[:8]
        return ' '.join(words) + '...'
    
    def _synthesize_reference_entry(self, node: Dict, all_nodes: List[Dict], edges: List[Dict]) -> str:
        """Synthesize a reference documentation entry"""
        entry = []
        
        content = node.get('content', '')
        
        # Description
        description = self._extract_key_point(content, 'mechanisms')
        entry.append(f"{description}\n")
        
        # Details if substantial
        if len(content) > 200:
            entry.append("\n**Details:**\n")
            # Extract middle portion for details
            sentences = content.split('.')
            if len(sentences) > 2:
                detail_sentences = sentences[1:-1]  # Skip first and last
                details = '. '.join(s.strip() for s in detail_sentences[:3])
                if details:
                    entry.append(details + ".\n")
        
        # Related concepts
        related = self._find_related_nodes(node, all_nodes, edges)
        if related:
            entry.append("\n**See Also:** ")
            related_titles = [self._generate_entry_title(r) for r in related[:3]]
            entry.append(", ".join(related_titles))
        
        return ''.join(entry)
    
    def _find_related_nodes(self, node: Dict, all_nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Find nodes related to the given node"""
        related = []
        node_id = node.get('id', '')
        
        # Create node lookup
        node_map = {n.get('id', ''): n for n in all_nodes}
        
        # Find connected nodes
        for edge in edges:
            if edge['source'] == node_id and edge['target'] in node_map:
                related.append(node_map[edge['target']])
            elif edge['target'] == node_id and edge['source'] in node_map:
                related.append(node_map[edge['source']])
        
        # Sort by importance
        related.sort(key=lambda n: n.get('importance', 0), reverse=True)
        
        return related
    
    def _generate_cross_references(self, nodes: List[Dict], edges: List[Dict]) -> str:
        """Generate cross-reference section"""
        cross_refs = []
        
        # Analyze edge patterns
        edge_patterns = {}
        for edge in edges:
            edge_type = edge.get('type', 'reference')
            if edge_type not in edge_patterns:
                edge_patterns[edge_type] = []
            edge_patterns[edge_type].append(edge)
        
        cross_refs.append("The following cross-references highlight important relationships:\n")
        
        for edge_type, type_edges in edge_patterns.items():
            if len(type_edges) > 2:  # Only show significant patterns
                cross_refs.append(f"\n**{self._humanize_relationship_type(edge_type).title()}:**\n")
                # Show top 3 examples
                for edge in type_edges[:3]:
                    source = next((n for n in nodes if n.get('id') == edge['source']), None)
                    target = next((n for n in nodes if n.get('id') == edge['target']), None)
                    if source and target:
                        source_title = self._generate_entry_title(source)[:40]
                        target_title = self._generate_entry_title(target)[:40]
                        cross_refs.append(f"- {source_title} → {target_title}\n")
        
        return ''.join(cross_refs)
    
    def _generate_index(self, nodes: List[Dict]) -> str:
        """Generate an index of key terms"""
        index_terms = {}
        
        # Extract key terms from nodes
        for i, node in enumerate(nodes):
            content = node.get('content', '').lower()
            
            # Extract important terms (simplified version)
            important_words = [
                'algorithm', 'method', 'process', 'system', 'component',
                'implementation', 'architecture', 'pattern', 'model',
                'framework', 'approach', 'technique', 'strategy'
            ]
            
            for term in important_words:
                if term in content:
                    if term not in index_terms:
                        index_terms[term] = []
                    index_terms[term].append(i + 1)  # 1-based indexing
        
        # Build index
        index = []
        for term in sorted(index_terms.keys()):
            locations = index_terms[term][:5]  # Limit to 5 references
            index.append(f"- **{term.title()}**: sections {', '.join(map(str, locations))}\n")
        
        return ''.join(index)
    
    def _extract_project_info(self, nodes: List[Dict]) -> Dict[str, Any]:
        """Extract project information from nodes"""
        info = {
            'name': 'Project',
            'description': '',
            'version': '1.0.0',
            'features': [],
            'requirements': []
        }
        
        # Look for project-specific information
        for node in nodes:
            content = node.get('content', '').lower()
            if 'project' in content and 'name' in content:
                # Try to extract project name
                sentences = node.get('content', '').split('.')
                for sent in sentences:
                    if 'project' in sent.lower():
                        info['name'] = self._extract_project_name(sent)
                        break
        
        return info
    
    def _extract_project_name(self, text: str) -> str:
        """Extract project name from text"""
        # Simple heuristic - look for capitalized words near 'project'
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() == 'project' and i > 0:
                if words[i-1][0].isupper():
                    return words[i-1]
            elif word.lower() == 'project' and i < len(words) - 1:
                if words[i+1][0].isupper():
                    return words[i+1]
        return "Project"
    
    def _synthesize_project_description(self, nodes: List[Dict]) -> str:
        """Create project description from nodes"""
        # Find overview or introduction nodes
        overview_nodes = []
        for node in nodes:
            content = node.get('content', '').lower()
            if any(term in content for term in ['overview', 'introduction', 'purpose', 'description']):
                overview_nodes.append(node)
        
        if overview_nodes:
            # Use the most important overview node
            primary = max(overview_nodes, key=lambda n: n.get('importance', 0))
            return self._extract_key_point(primary.get('content', ''), 'principles')
        else:
            # Generate generic description
            return "A comprehensive system implementing the concepts and techniques documented in this repository."
    
    def _extract_features(self, nodes: List[Dict]) -> List[str]:
        """Extract feature list from nodes"""
        features = []
        
        for node in nodes:
            content = node.get('content', '')
            content_lower = content.lower()
            
            # Look for feature indicators
            if any(indicator in content_lower for indicator in ['feature', 'capability', 'provides', 'enables', 'supports']):
                feature = self._extract_feature_description(content)
                if feature and feature not in features:
                    features.append(feature)
        
        # Limit to top 8 features
        return features[:8]
    
    def _extract_feature_description(self, content: str) -> str:
        """Extract a feature description from content"""
        sentences = content.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in ['feature', 'capability', 'provides', 'enables']):
                # Clean and return
                feature = sentence.strip()
                if len(feature) < 100:
                    return feature
        
        # Fallback - use first sentence
        if sentences:
            return sentences[0].strip()
        
        return ""
    
    def _extract_installation_steps(self, nodes: List[Dict]) -> List[str]:
        """Extract installation steps from nodes"""
        steps = []
        
        for node in nodes:
            content = node.get('content', '')
            content_lower = content.lower()
            
            # Look for installation commands
            if any(cmd in content_lower for cmd in ['pip install', 'npm install', 'git clone', 'docker']):
                lines = content.split('\n')
                for line in lines:
                    if any(cmd in line for cmd in ['pip', 'npm', 'git', 'docker', 'python']):
                        steps.append(line.strip())
        
        if not steps:
            # Provide generic steps
            steps = [
                "git clone <repository-url>",
                "cd <project-directory>",
                "pip install -r requirements.txt"
            ]
        
        return steps[:5]  # Limit to 5 steps
    
    def _synthesize_usage_info(self, nodes: List[Dict]) -> str:
        """Synthesize usage information"""
        usage_nodes = []
        
        for node in nodes:
            content = node.get('content', '').lower()
            if any(term in content for term in ['usage', 'how to', 'example', 'run', 'execute']):
                usage_nodes.append(node)
        
        if usage_nodes:
            # Synthesize from usage nodes
            return self._synthesize_paragraph(
                usage_nodes[:3],
                "To use this system,",
                focus="uses"
            )
        else:
            # Generic usage
            return "Please refer to the documentation for detailed usage instructions and examples."
    
    def _synthesize_architecture_section(self, arch_nodes: List[Dict]) -> str:
        """Synthesize architecture description"""
        if not arch_nodes:
            return "The system follows a modular architecture designed for flexibility and extensibility."
        
        return self._synthesize_paragraph(
            arch_nodes,
            "The architecture",
            focus="mechanisms"
        )
    
    def _generate_acknowledgments(self, nodes: List[Dict]) -> str:
        """Generate acknowledgments section"""
        ack = "This project builds upon the foundational work in "
        
        # Identify key areas
        areas = set()
        for node in nodes[:20]:  # Sample nodes
            content = node.get('content', '').lower()
            if 'theory' in content:
                areas.add('theoretical frameworks')
            if 'algorithm' in content:
                areas.add('algorithmic research')
            if 'implementation' in content:
                areas.add('practical implementations')
        
        if areas:
            ack += ", ".join(sorted(areas)) + "."
        else:
            ack += "various domains of computer science and software engineering."
        
        return ack