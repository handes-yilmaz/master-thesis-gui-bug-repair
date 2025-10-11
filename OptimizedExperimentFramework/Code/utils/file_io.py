"""
File I/O utilities for GUIRepair framework
Clean and simple file operations
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional
import base64
from PIL import Image


def read_file(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Read file contents safely
    
    Args:
        file_path: Path to file
        encoding: File encoding (default: utf-8)
        
    Returns:
        File contents as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        UnicodeDecodeError: If encoding fails
    """
    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        return f.read()


def read_file_lines(file_path: str, encoding: str = 'utf-8') -> list[str]:
    """Read file as list of lines"""
    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
        return f.readlines()


def save_file(file_path: str, content: str, encoding: str = 'utf-8') -> bool:
    """
    Save content to file safely
    
    Args:
        file_path: Path to save file
        content: Content to write
        encoding: File encoding (default: utf-8)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create parent directories if needed
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error saving file {file_path}: {e}")
        return False


def read_json(file_path: str) -> Dict[str, Any]:
    """
    Read JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Parsed JSON as dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(file_path: str, data: Dict[str, Any], indent: int = 4) -> bool:
    """
    Save data as JSON file
    Handles Chinese characters properly
    
    Args:
        file_path: Path to save JSON
        data: Data to serialize
        indent: JSON indentation (default: 4)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create parent directories if needed
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Check if data contains Chinese characters
        contains_chinese = _contains_chinese(data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=not contains_chinese)
        return True
    except Exception as e:
        print(f"Error saving JSON {file_path}: {e}")
        return False


def _contains_chinese(obj: Any) -> bool:
    """Check if object contains Chinese characters"""
    if isinstance(obj, str):
        return bool(re.search(r'[\u4e00-\u9fff]', obj))
    elif isinstance(obj, dict):
        return any(_contains_chinese(v) for v in obj.values())
    elif isinstance(obj, (list, tuple, set)):
        return any(_contains_chinese(item) for item in obj)
    return False


def load_image_as_base64(image_path: str, format: str = "png") -> str:
    """
    Load image and convert to base64 string for API
    
    Args:
        image_path: Path to image file
        format: Image format (png, jpeg, etc.)
        
    Returns:
        Base64 encoded image string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def load_images_for_multimodal(
    image_dir: str,
    model_type: str = "openai"
) -> list[Dict[str, Any]]:
    """
    Load all images from directory for multimodal API
    
    Args:
        image_dir: Directory containing images
        model_type: 'openai' or 'claude'
        
    Returns:
        List of image objects formatted for API
    """
    image_path = Path(image_dir)
    if not image_path.exists():
        return []
    
    images = []
    
    # Get all image files
    image_files = sorted([
        f for f in image_path.iterdir()
        if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.webp']
    ])
    
    for img_file in image_files:
        # Determine image type
        img_type = img_file.suffix.lower()[1:]  # Remove dot
        if img_type == 'jpg':
            img_type = 'jpeg'
        
        # Encode image
        base64_image = load_image_as_base64(str(img_file), img_type)
        
        # Format for API
        if model_type == "openai":
            images.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{img_type};base64,{base64_image}"
                }
            })
        elif model_type == "claude":
            images.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": f"image/{img_type}",
                    "data": base64_image
                }
            })
    
    return images


def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)



