#!/usr/bin/env python3
"""
Enhanced logging system for research data collection
Captures timing, complexity, domain, and performance metrics
"""

import json
import time
import os
import re
from datetime import datetime
from PIL import Image
import cv2
import numpy as np

class ResearchLogger:
    def __init__(self, output_dir, instance_id, repo_name):
        self.output_dir = output_dir
        self.instance_id = instance_id
        self.repo_name = repo_name
        self.start_time = time.time()
        self.phase_timings = {}
        self.bug_description = ""
        self.image_paths = []
        self.research_data = {
            "instance_id": instance_id,
            "repo_name": repo_name,
            "domain": self._classify_domain(repo_name),
            "timestamp": datetime.now().isoformat(),
            "phases": {},
            "complexity_metrics": {},
            "performance_metrics": {},
            "visual_analysis": {},
            "visual_complexity": "unknown"  # Will be set by analyze_visual_complexity
        }
    
    def _classify_domain(self, repo_name):
        """
        Classify repository into domain categories based on comprehensive analysis
        of all 12 repos in SWE-bench Multimodal test split
        """
        domain_mapping = {
            # Web UI Component Libraries (3 repos, 192 instances)
            "alibaba-fusion": "web_ui_components",        # 39 instances
            "carbon-design-system": "web_ui_components",  # 133 instances  
            "grommet": "web_ui_components",               # 20 instances
            
            # Syntax Highlighting Libraries (2 repos, 77 instances)
            "PrismJS": "syntax_highlighting",             # 38 instances
            "highlightjs": "syntax_highlighting",         # 39 instances
            
            # Code Formatting & Linting Tools (2 repos, 24 instances)
            "prettier": "code_formatting",                # 13 instances
            "eslint": "code_linting",                     # 11 instances
            
            # Diagram & Visualization Tools (2 repos, 133 instances)
            "bpmn-io": "diagram_rendering",               # 54 instances
            "openlayers": "map_visualization",            # 79 instances
            
            # Web Tools & Performance (1 repo, 54 instances)
            "GoogleChrome": "web_performance_audit",      # 54 instances
            
            # Document & Content Rendering (1 repo, 24 instances)
            "quarto-dev": "document_rendering",           # 24 instances
            
            # Visual Programming IDEs (1 repo, 3 instances)
            "scratchfoundation": "visual_programming_ide" # 3 instances
        }
        return domain_mapping.get(repo_name, "unknown")
    
    def set_bug_description(self, description):
        """Store bug description for visual complexity analysis"""
        self.bug_description = description
        self.research_data["bug_description_length"] = len(description) if description else 0
    
    def start_phase(self, phase_name):
        """Start timing a phase"""
        self.phase_timings[phase_name] = time.time()
        self.research_data["phases"][phase_name] = {
            "start_time": time.time(),
            "status": "running"
        }
    
    def end_phase(self, phase_name, success=True, tokens_used=None):
        """End timing a phase and record results"""
        end_time = time.time()
        start_time = self.phase_timings.get(phase_name, end_time)
        duration = end_time - start_time
        
        self.research_data["phases"][phase_name].update({
            "end_time": end_time,
            "duration": duration,
            "success": success,
            "tokens_used": tokens_used,
            "status": "completed" if success else "failed"
        })
    
    def analyze_visual_complexity(self, image_file_list):
        """
        Analyze visual complexity of bug screenshots using multiple factors:
        - Image analysis metrics (edges, colors, texture)
        - Bug description keywords
        - Domain-specific complexity patterns
        """
        if not image_file_list:
            self.research_data["visual_complexity"] = "simple"
            return
        
        self.image_paths = image_file_list
        complexity_scores = []
        ui_elements = []
        image_features = []
        
        for img_item in image_file_list:
            # Handle both string paths and dictionary format
            if isinstance(img_item, dict):
                img_path = img_item.get('path', '')
            else:
                img_path = str(img_item)
            
            if os.path.exists(img_path):
                try:
                    # Load image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    # Convert to grayscale for analysis
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate complexity metrics
                    height, width = gray.shape
                    total_pixels = height * width
                    
                    # Edge density (complexity indicator)
                    edges = cv2.Canny(gray, 50, 150)
                    edge_density = np.sum(edges > 0) / total_pixels
                    
                    # Color variance (UI element diversity)
                    if len(img.shape) == 3:
                        color_variance = np.var(img, axis=(0,1)).mean()
                    else:
                        color_variance = np.var(gray)
                    
                    # Texture complexity (Laplacian variance)
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    
                    # UI element detection (simple contour-based)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    ui_element_count = len([c for c in contours if cv2.contourArea(c) > 100])
                    
                    # Weighted complexity score
                    complexity_score = (edge_density * 0.4 + 
                                      (color_variance / 10000) * 0.3 + 
                                      (laplacian_var / 1000) * 0.3)
                    
                    complexity_scores.append(complexity_score)
                    ui_elements.append(ui_element_count)
                    
                    image_features.append({
                        "edge_density": edge_density,
                        "color_variance": color_variance,
                        "laplacian_var": laplacian_var,
                        "ui_element_count": ui_element_count,
                        "complexity_score": complexity_score
                    })
                    
                except Exception as e:
                    print(f"Error analyzing image {img_path}: {e}")
                    continue
            else:
                print(f"Image path does not exist: {img_path}")
                continue
        
        # Textual complexity analysis
        text_complexity_score = self._analyze_textual_complexity(self.bug_description)
        
        # Calculate final complexity
        if complexity_scores:
            avg_complexity = np.mean(complexity_scores)
            max_complexity = np.max(complexity_scores)
            avg_ui_elements = np.mean(ui_elements)
            
            self.research_data["visual_analysis"] = {
                "avg_complexity_score": float(avg_complexity),
                "max_complexity_score": float(max_complexity),
                "min_complexity_score": float(np.min(complexity_scores)),
                "avg_ui_elements": float(avg_ui_elements),
                "total_ui_elements": int(np.sum(ui_elements)),
                "num_images": len(complexity_scores),
                "text_complexity_score": text_complexity_score,
                "image_features": image_features
            }
            
            # Determine visual complexity level with hybrid approach
            visual_complexity = self._classify_visual_complexity(
                avg_complexity, 
                avg_ui_elements, 
                text_complexity_score,
                self.repo_name
            )
            self.research_data["visual_complexity"] = visual_complexity
        else:
            # No image analysis, use text-based complexity
            self.research_data["visual_analysis"] = {
                "text_complexity_score": text_complexity_score,
                "num_images": 0
            }
            visual_complexity = self._classify_from_text_only(text_complexity_score)
            self.research_data["visual_complexity"] = visual_complexity
    
    def _analyze_textual_complexity(self, description):
        """
        Analyze bug description for complexity indicators
        Returns a score from 0-10
        """
        if not description:
            return 3  # Default medium-low
        
        description_lower = description.lower()
        
        # Complex visual keywords (weight: +2 each)
        complex_keywords = [
            'animation', 'webgl', 'canvas', '3d', 'rendering', 'shader',
            'transform', 'rotation', 'complex layout', 'intricate',
            'drag', 'drop', 'gesture', 'interaction', 'multiple components',
            'nested', 'overlay', 'z-index', 'positioning'
        ]
        
        # Medium complexity keywords (weight: +1 each)
        medium_keywords = [
            'layout', 'alignment', 'spacing', 'margin', 'padding',
            'color', 'style', 'css', 'display', 'flex', 'grid',
            'responsive', 'viewport', 'size', 'dimension'
        ]
        
        # Simple keywords (weight: +0.5 each)
        simple_keywords = [
            'text', 'label', 'button', 'input', 'form',
            'typo', 'spelling', 'wording', 'missing'
        ]
        
        score = 3  # Base score
        
        # Count keyword matches
        for keyword in complex_keywords:
            if keyword in description_lower:
                score += 2
        
        for keyword in medium_keywords:
            if keyword in description_lower:
                score += 1
        
        for keyword in simple_keywords:
            if keyword in description_lower:
                score += 0.5
        
        # Check for multiple visual issues mentioned
        if description_lower.count('screenshot') > 1 or description_lower.count('image') > 1:
            score += 1
        
        # Cap score at 10
        return min(score, 10)
    
    def _classify_visual_complexity(self, image_complexity, ui_elements, text_score, domain):
        """
        Classify complexity into simple/medium/complex using multiple factors
        
        Factors:
        - Image analysis metrics
        - Number of UI elements
        - Text complexity score
        - Domain-specific patterns
        """
        # Domain-specific baseline adjustments for all 12 repositories
        domain_complexity_bias = {
            "web_ui_components": 0,           # alibaba-fusion, carbon-design-system, grommet
            "syntax_highlighting": 0,         # PrismJS, highlightjs
            "code_formatting": -0.1,          # prettier (usually simpler visual issues)
            "code_linting": -0.1,             # eslint (usually simpler visual issues)
            "diagram_rendering": 0.2,         # bpmn-io (often complex visual rendering)
            "map_visualization": 0.2,         # openlayers (geographic rendering complexity)
            "web_performance_audit": 0,       # GoogleChrome/lighthouse
            "document_rendering": 0.1,        # quarto-dev (document layout complexity)
            "visual_programming_ide": 0.2     # scratchfoundation (block-based UI complexity)
        }
        
        domain_bias = domain_complexity_bias.get(domain, 0)
        
        # Weighted scoring system
        # Image complexity: 0.4 weight
        # UI elements: 0.3 weight  
        # Text description: 0.3 weight
        normalized_img = min(image_complexity, 1.0)  # Normalize to 0-1
        normalized_ui = min(ui_elements / 100, 1.0)  # Normalize to 0-1 (100 elements = high)
        normalized_text = text_score / 10  # Normalize to 0-1
        
        final_score = (normalized_img * 0.4 + 
                      normalized_ui * 0.3 + 
                      normalized_text * 0.3 +
                      domain_bias)
        
        # Classification thresholds
        if final_score < 0.35:
            return "simple"
        elif final_score < 0.65:
            return "medium"
        else:
            return "complex"
    
    def _classify_from_text_only(self, text_score):
        """Classify complexity from text analysis only (when no images available)"""
        if text_score < 4:
            return "simple"
        elif text_score < 7:
            return "medium"
        else:
            return "complex"
    
    def record_performance_metrics(self, phase_name, metrics):
        """Record performance metrics for a phase"""
        if "performance_metrics" not in self.research_data:
            self.research_data["performance_metrics"] = {}
        
        self.research_data["performance_metrics"][phase_name] = metrics
    
    def save_research_data(self):
        """Save all research data to JSON file"""
        # Calculate total execution time
        total_time = time.time() - self.start_time
        self.research_data["total_execution_time"] = total_time
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save to file
        output_file = os.path.join(self.output_dir, "research_metrics.json")
        with open(output_file, 'w') as f:
            json.dump(self.research_data, f, indent=2)
        
        print(f"📊 Research data saved to: {output_file}")
        print(f"   Domain: {self.research_data['domain']}")
        print(f"   Visual Complexity: {self.research_data['visual_complexity']}")
        return output_file

# Integration functions for existing workflow
def log_phase_start(logger, phase_name):
    """Helper function to start phase logging"""
    if logger:
        logger.start_phase(phase_name)

def log_phase_end(logger, phase_name, success=True, tokens=None):
    """Helper function to end phase logging"""
    if logger:
        logger.end_phase(phase_name, success, tokens)

def analyze_bug_images(logger, image_paths):
    """Helper function to analyze bug images"""
    if logger:
        logger.analyze_visual_complexity(image_paths)

