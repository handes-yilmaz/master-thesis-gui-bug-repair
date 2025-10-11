"""
Prompt templates for GUIRepair 3-Phase Framework
"""
from .file_localization import FileLocalizationPrompts
from .element_localization import ElementLocalizationPrompts
from .patch_generation import PatchGenerationPrompts

__all__ = [
    'FileLocalizationPrompts',
    'ElementLocalizationPrompts',
    'PatchGenerationPrompts',
]



