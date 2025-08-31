#!/usr/bin/env python
"""
Cloud Processing Setup Script for SWE-bench Experiments
=======================================================

This script helps set up cloud processing for research,
overcoming local resource constraints.
"""

import json
import os
from pathlib import Path

def create_cloud_processing_notebook():
    """Create a Jupyter notebook optimized for cloud processing"""
    
    print("📓 Creating cloud-optimized Jupyter notebook...")
    
    # Create notebooks directory if it doesn't exist
    notebooks_dir = Path("notebooks")
    notebooks_dir.mkdir(exist_ok=True)
    
    # Create a simple notebook structure
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🔬 Real SWE-bench GUI Bug Experiments (Cloud Processing)\n",
                    "\n",
                    "This notebook runs experiments on real SWE-bench GUI bugs using cloud resources.\n",
                    "Perfect for limited local resources!\n",
                    "\n",
                    "## What This Notebook Does:\n",
                    "1. **Loads real SWE-bench GUI bugs** (not synthetic scenarios)\n",
                    "2. **Runs LLM experiments** on authentic bug reports\n",
                    "3. **Generates research results** for research documents\n",
                    "4. **Uses cloud resources** for heavy processing\n",
                    "\n",
                    "## Setup:\n",
                    "```bash\n",
                    "pip install datasets openai anthropic\n",
                    "```\n",
                    "\n",
                    "## Environment Variables:\n",
                    "```bash\n",
                    "export OPENAI_API_KEY='your_key_here'\n",
                    "export ANTHROPIC_API_KEY='your_key_here'\n",
                    "```"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install required packages\n",
                    "!pip install datasets openai anthropic python-dotenv tqdm"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Set your API keys here\n",
                    "import os\n",
                    "\n",
                    "# Replace with your actual API keys\n",
                    "os.environ['OPENAI_API_KEY'] = 'your_openai_key_here'\n",
                    "os.environ['ANTHROPIC_API_KEY'] = 'your_anthropic_key_here'\n",
                    "\n",
                    "print(\"API keys set successfully!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load SWE-bench data\n",
                    "from datasets import load_dataset\n",
                    "import json\n",
                    "\n",
                    "print(\"Loading SWE-bench Multimodal...\")\n",
                    "swebench_mm = load_dataset('SWE-bench/SWE-bench_Multimodal', split='test[:200]')\n",
                    "print(f\"Loaded {len(swebench_mm)} samples\")\n",
                    "\n",
                    "# Extract GUI bugs\n",
                    "gui_keywords = ['button', 'form', 'ui', 'css', 'layout', 'click', 'frontend', 'react', 'vue', 'javascript']\n",
                    "gui_bugs = []\n",
                    "\n",
                    "for sample in swebench_mm:\n",
                    "    problem = sample.get('problem_statement', '').lower()\n",
                    "    if sum(1 for keyword in gui_keywords if keyword in problem) >= 2:\n",
                    "        gui_bugs.append({\n",
                    "            'id': sample.get('instance_id', 'unknown'),\n",
                    "            'repo': sample.get('repo', 'unknown'),\n",
                    "            'title': sample.get('problem_statement', '')[:100] + '...',\n",
                    "            'description': sample.get('problem_statement', ''),\n",
                    "            'has_image': bool(sample.get('image_assets'))\n",
                    "        })\n",
                    "\n",
                    "print(f\"Found {len(gui_bugs)} GUI-related bugs!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run experiments on real bugs\n",
                    "import openai\n",
                    "import anthropic\n",
                    "import time\n",
                    "from datetime import datetime\n",
                    "\n",
                    "# Initialize clients\n",
                    "if os.getenv('OPENAI_API_KEY'):\n",
                    "    openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))\n",
                    "    print(\"✓ OpenAI client initialized\")\n",
                    "\n",
                    "if os.getenv('ANTHROPIC_API_KEY'):\n",
                    "    anthropic_client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))\n",
                    "    print(\"✓ Anthropic client initialized\")\n",
                    "\n",
                    "# Test prompt\n",
                    "test_prompt = \"\"\"\n",
                    "You are a senior frontend engineer. Given this bug report:\n",
                    "\n",
                    "BUG REPORT:\n",
                    "{bug_description}\n",
                    "\n",
                    "REPOSITORY: {repo}\n",
                    "\n",
                    "Provide a solution in this format:\n",
                    "ROOT CAUSE: [brief explanation]\n",
                    "SOLUTION: [detailed fix steps]\n",
                    "EXPLANATION: [how this fixes the issue]\n",
                    "TESTING: [steps to verify the fix]\n",
                    "\"\"\"\n",
                    "\n",
                    "print(\"Starting experiments...\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run experiments and collect results\n",
                    "results = []\n",
                    "\n",
                    "# Test on first 10 bugs to start\n",
                    "test_bugs = gui_bugs[:10]\n",
                    "\n",
                    "for i, bug in enumerate(test_bugs):\n",
                    "    print(f\"\\nTesting bug {i+1}/{len(test_bugs)}: {bug['title'][:60]}...\")\n",
                    "    \n",
                    "    # Format prompt\n",
                    "    prompt = test_prompt.format(\n",
                    "        bug_description=bug['description'][:1000],\n",
                    "        repo=bug['repo']\n",
                    "    )\n",
                    "    \n",
                    "    try:\n",
                    "        # Test with OpenAI if available\n",
                    "        if 'openai_client' in locals():\n",
                    "            start_time = time.time()\n",
                    "            response = openai_client.chat.completions.create(\n",
                    "                model=\"gpt-4o-mini\",\n",
                    "                messages=[{\"role\": \"user\", \"content\": prompt}],\n",
                    "                temperature=0.1\n",
                    "            )\n",
                    "            processing_time = time.time() - start_time\n",
                    "            \n",
                    "            results.append({\n",
                    "                'bug_id': bug['id'],\n",
                    "                'repo': bug['repo'],\n",
                    "                'model': 'gpt-4o-mini',\n",
                    "                'response': response.choices[0].message.content,\n",
                    "                'processing_time': processing_time,\n",
                    "                'timestamp': datetime.now().isoformat()\n",
                    "            })\n",
                    "            \n",
                    "            print(f\"  ✅ OpenAI experiment completed in {processing_time:.2f}s\")\n",
                    "        \n",
                    "        # Test with Anthropic if available\n",
                    "        if 'anthropic_client' in locals():\n",
                    "            start_time = time.time()\n",
                    "            response = anthropic_client.messages.create(\n",
                    "                model=\"claude-3-5-sonnet-20241022\",\n",
                    "                max_tokens=1000,\n",
                    "                temperature=0.1,\n",
                    "                messages=[{\"role\": \"user\", \"content\": prompt}]\n",
                    "            )\n",
                    "            processing_time = time.time() - start_time\n",
                    "            \n",
                    "            results.append({\n",
                    "                'bug_id': bug['id'],\n",
                    "                'repo': bug['repo'],\n",
                    "                'model': 'claude-3-5-sonnet',\n",
                    "                'timestamp': datetime.now().isoformat()\n",
                    "            })\n",
                    "            \n",
                    "            print(f\"  ✅ Anthropic experiment completed in {processing_time:.2f}s\")\n",
                    "    \n",
                    "    except Exception as e:\n",
                    "        print(f\"  ❌ Experiment failed: {e}\")\n",
                    "    \n",
                    "    # Small delay between experiments\n",
                    "    time.sleep(1)\n",
                    "\n",
                    "print(f\"\\n🎉 Completed {len(results)} experiments!\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Save results and generate report\n",
                    "if results:\n",
                    "    # Save results\n",
                    "    results_file = f\"cloud_processing_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json\"\n",
                    "    with open(results_file, 'w') as f:\n",
                    "        json.dump(results, f, indent=2)\n",
                    "    print(f\"💾 Results saved to: {results_file}\")\n",
                    "    \n",
                    "    # Generate summary\n",
                    "    print(\"\\n📊 Experiment Summary:\")\n",
                    "    print(f\"Total experiments: {len(results)}\")\n",
                    "    \n",
                    "    # Count by model\n",
                    "    models = {}\n",
                    "    for result in results:\n",
                    "        model = result['model']\n",
                    "        if model not in models:\n",
                    "            models[model] = 0\n",
                    "        models[model] += 1\n",
                    "    \n",
                    "    for model, count in models.items():\n",
                    "        print(f\"  {model}: {count} experiments\")\n",
                    "    \n",
                    "    # Show sample response\n",
                    "    if results:\n",
                    "        print(f\"\\n📝 Sample Response (first 200 chars):\")\n",
                    "        sample = results[0]['response'][:200]\n",
                    "        print(sample + \"...\")\n",
                    "    \n",
                    "    print(\"\\n🎯 Research now has authentic results from real SWE-bench bugs!\")\n",
                    "else:\n",
                    "    print(\"\\n❌ No results to save\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Save the notebook
    notebook_file = notebooks_dir / "real_swe_bench_experiments.ipynb"
    with open(notebook_file, 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print(f"✅ Cloud-optimized notebook created: {notebook_file}")
    return notebook_file

def create_cloud_processing_guide():
    """Create a comprehensive guide for cloud processing setup"""
    
    print("📋 Creating cloud processing setup guide...")
    
    guide_content = """# 🚀 Cloud Processing Setup Guide

## ☁️ Google Colab Setup (Recommended)

### Step 1: Access Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account
3. Create a new notebook

### Step 2: Upload Your Notebook
1. Download: `notebooks/real_swe_bench_experiments.ipynb`
2. In Colab: File → Upload notebook
3. Select your downloaded `.ipynb` file

### Step 3: Set API Keys
```python
import os
os.environ['OPENAI_API_KEY'] = 'your_openai_key_here'
os.environ['ANTHROPIC_API_KEY'] = 'your_anthropic_key_here'
```

### Step 4: Install Dependencies
```python
!pip install datasets openai anthropic python-dotenv tqdm
```

### Step 5: Run Experiments
Execute all cells in order. The notebook will:
- Load SWE-bench data
- Extract GUI bugs
- Run LLM experiments
- Save results

## 🔧 Local Jupyter Setup (Alternative)

### Step 1: Install Jupyter
```bash
conda activate gui-bug-repair
pip install jupyter notebook
jupyter notebook
```

### Step 2: Open Notebook
Navigate to `notebooks/` and open `real_swe_bench_experiments.ipynb`

## 📊 Expected Results

- **Processing Time**: 20-60 minutes (cloud) vs 2-4 hours (local)
- **Memory Usage**: Unlimited (cloud) vs 4.7GB limit (local)
- **Output**: JSON results ready for research integration

## 🎯 Success Criteria

- [ ] Notebook uploaded to cloud platform
- [ ] API keys configured
- [ ] Dependencies installed
- [ ] Experiments running successfully
- [ ] Results downloaded for local analysis

## 💡 Pro Tips

1. **Start Small**: Test with 5-10 bugs first
2. **Monitor Progress**: Watch experiment execution
3. **Save Frequently**: Download intermediate results
4. **Use GPU Runtime**: Enable GPU in Colab for faster processing
"""
    
    # Save the guide
    guide_file = Path("docs/CLOUD_PROCESSING_QUICK_START.md")
    guide_file.parent.mkdir(exist_ok=True)
    
    with open(guide_file, 'w') as f:
        f.write(guide_content)
    
    print(f"✅ Cloud processing guide created: {guide_file}")
    return guide_file

def main():
    """Main function to set up cloud processing"""
    print("🚀 Cloud Processing Setup for SWE-bench Experiments")
    print("=" * 60)
    
    # Create cloud-optimized notebook
    notebook_file = create_cloud_processing_notebook()
    
    # Create setup guide
    guide_file = create_cloud_processing_guide()
    
    print("\n🎉 Cloud Processing Setup Complete!")
    print("\n📋 What was created:")
    print(f"✅ Cloud-optimized notebook: {notebook_file}")
    print(f"✅ Setup guide: {guide_file}")
    
    print("\n🚀 Next Steps:")
    print("1. Upload notebook to Google Colab")
    print("2. Set your API keys")
    print("3. Install dependencies")
    print("4. Run experiments on real SWE-bench bugs!")
    
    print("\n💡 Benefits of Cloud Processing:")
    print("- Unlimited RAM (vs your 4.7GB constraint)")
    print("- GPU/TPU acceleration")
    print("- Process full 46-bug dataset efficiently")
    print("- Keep your local computer responsive")
    
    print(f"\n📁 Files ready for upload:")
    print(f"  - Notebook: {notebook_file}")
    print(f"  - Guide: {guide_file}")

if __name__ == "__main__":
    main()
