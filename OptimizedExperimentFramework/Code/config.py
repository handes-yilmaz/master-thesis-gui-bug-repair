"""
Configuration management for GUIRepair 3-Phase Framework
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class GUIRepairConfig:
    """Configuration for GUIRepair experiment"""
    
    # Model Configuration
    model_name: str = "gpt-4o-2024-08-06"
    api_key: str = ""
    
    # Temperature & Sampling
    file_localization_temp: float = 1.0
    file_localization_samples: int = 2
    element_localization_temp: float = 0.7
    element_localization_samples: int = 2
    patch_generation_temp: float = 1.0
    patch_generation_samples: int = 1
    
    # File Limits
    max_candidate_files: int = 4
    max_lines_per_file: int = 500
    context_window: int = 10
    
    # Dataset Configuration
    dataset: str = "princeton-nlp/SWE-bench_Multimodal"
    dataset_split: str = "test"
    repo_path: str = "../Data/Reproduce_Scenario"
    
    # Output Configuration
    output_dir: str = "results"
    
    # Experiment Settings
    text_only: bool = False  # For text-only vs multimodal experiments
    enable_images: bool = True
    wait_time_after_api: int = 0  # Seconds to wait after API request
    
    # Code Compression
    compress_assignments: bool = True
    compress_total_lines: int = 30
    compress_prefix_lines: int = 10
    compress_suffix_lines: int = 10
    
    def __post_init__(self):
        """Validate configuration"""
        if not self.api_key:
            raise ValueError("API key is required")
        
        # Disable images if text_only mode
        if self.text_only:
            self.enable_images = False
    
    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments"""
        return cls(
            model_name=args.model,
            api_key=args.api_key,
            file_localization_temp=args.file_loc_temp,
            file_localization_samples=args.file_loc_samples,
            element_localization_temp=args.elem_loc_temp,
            element_localization_samples=args.elem_loc_samples,
            patch_generation_temp=args.patch_temp,
            patch_generation_samples=args.patch_samples,
            max_candidate_files=args.max_files,
            max_lines_per_file=args.max_lines,
            context_window=args.context,
            dataset=args.dataset,
            dataset_split=args.split,
            repo_path=args.repo_path,
            output_dir=args.output_dir,
            text_only=args.text_only,
            enable_images=not args.text_only,
            wait_time_after_api=args.wait_time,
        )


