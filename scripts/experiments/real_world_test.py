#!/usr/bin/env python
"""
Real-World GUI Bug Repair Experiments
====================================

This script tests the LLM on more realistic, complex GUI bug scenarios
that developers commonly encounter in production environments.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

# Import our LLM client
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm_client import LLMClient, LLMConfig

load_dotenv()

class RealWorldBugTester:
    def __init__(self, config_path: str = "configs/config.json"):
        """Initialize the real-world bug tester"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize LLM client
        llm_config = LLMConfig(
            provider=self.config["llm"]["provider"],
            model=self.config["llm"]["model"],
            api_key_env=self.config["llm"]["api_key_env"],
            temperature=self.config["llm"]["temperature"]
        )
        self.llm_client = LLMClient(llm_config)
        
        # Create results directory
        self.results_dir = Path("runs/real_world_tests")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def get_real_world_scenarios(self) -> List[Dict[str, Any]]:
        """Define realistic, complex GUI bug scenarios"""
        scenarios = [
            {
                "id": "RW_01",
                "title": "Complex Form Validation Race Condition",
                "description": """Users report that when they quickly fill out a multi-step registration form, the validation sometimes fails incorrectly. Specifically, when users type quickly in the password field and immediately click "Next", the form shows "Password too weak" even though the password meets all requirements. This happens intermittently and is more common on slower devices.""",
                "code_context": """
// React component with async validation
const RegistrationForm = () => {
    const [password, setPassword] = useState('');
    const [isValidating, setIsValidating] = useState(false);
    const [validationResult, setValidationResult] = useState(null);
    
    const validatePassword = async (pass) => {
        setIsValidating(true);
        // Simulate API call for password strength check
        const result = await api.checkPasswordStrength(pass);
        setValidationResult(result);
        setIsValidating(false);
    };
    
    const handlePasswordChange = (e) => {
        const newPassword = e.target.value;
        setPassword(newPassword);
        // Debounced validation
        setTimeout(() => validatePassword(newPassword), 300);
    };
    
    return (
        <form onSubmit={handleSubmit}>
            <input 
                type="password"
                value={password}
                onChange={handlePasswordChange}
                onBlur={() => validatePassword(password)}
            />
            {isValidating && <span>Validating...</span>}
            {validationResult && !validationResult.strong && <span>Password too weak</span>}
            <button type="submit" disabled={isValidating || (validationResult && !validationResult.strong)}>
                Next
            </button>
        </form>
    );
};
                """,
                "ui_events": "user types password → validation starts → user clicks next before validation completes → form shows error incorrectly",
                "expected_issues": ["race condition", "debouncing", "async validation", "state management"]
            },
            
            {
                "id": "RW_02", 
                "title": "Dynamic Content Accessibility Issues",
                "description": """A dashboard component loads data dynamically and displays it in a table. However, screen reader users report that they cannot navigate through the table properly, and the content doesn't announce when new data loads. The table appears to be accessible but only when statically rendered.""",
                "code_context": """
// Dynamic table component
const DataTable = ({ data, isLoading }) => {
    const [currentData, setCurrentData] = useState([]);
    
    useEffect(() => {
        if (data) {
            setCurrentData(data);
        }
    }, [data]);
    
    return (
        <div role="region" aria-label="Data table">
            {isLoading && <div aria-live="polite">Loading data...</div>}
            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Status</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {currentData.map((row, index) => (
                        <tr key={row.id}>
                            <td>{row.name}</td>
                            <td>{row.status}</td>
                            <td>
                                <button onClick={() => handleAction(row.id)}>
                                    {row.status === 'active' ? 'Deactivate' : 'Activate'}
                                </button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};
                """,
                "ui_events": "page loads → data fetches → table renders → screen reader can't navigate properly",
                "expected_issues": ["aria-live regions", "focus management", "dynamic content announcement"]
            },
            
            {
                "id": "RW_03",
                "title": "Cross-Browser CSS Grid Layout Issues", 
                "description": """The main navigation layout uses CSS Grid and works perfectly in Chrome and Firefox, but in Safari (especially on iOS), the menu items are misaligned and some are cut off. The grid container has a fixed height that seems to be causing issues in Safari's grid implementation.""",
                "code_context": """
/* Navigation styles */
.nav-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    grid-template-rows: 50px;
    gap: 10px;
    height: 60px;
    align-items: center;
    padding: 0 20px;
    background: #fff;
    border-bottom: 1px solid #eee;
}

.nav-item {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    padding: 8px 12px;
    text-decoration: none;
    color: #333;
    border-radius: 4px;
    transition: background-color 0.2s;
}

.nav-item:hover {
    background-color: #f5f5f5;
}

/* Mobile styles */
@media (max-width: 768px) {
    .nav-container {
        grid-template-columns: repeat(3, 1fr);
        grid-template-rows: 40px;
        height: 50px;
        gap: 5px;
        padding: 0 10px;
    }
}
                """,
                "ui_events": "page loads → navigation renders → Safari shows misaligned grid items",
                "expected_issues": ["browser compatibility", "CSS grid", "mobile responsiveness", "safari-specific bugs"]
            },
            
            {
                "id": "RW_04",
                "title": "Memory Leak in Modal Component",
                "description": """Users report that after opening and closing a modal dialog multiple times, the application becomes slower and eventually crashes. The modal contains a complex form with file uploads and real-time validation. The issue seems to be related to event listeners not being properly cleaned up.""",
                "code_context": """
// Modal component with potential memory leak
const Modal = ({ isOpen, onClose, children }) => {
    const [activeTab, setActiveTab] = useState(0);
    const [uploadProgress, setUploadProgress] = useState(0);
    
    useEffect(() => {
        if (isOpen) {
            // Add global event listeners
            document.addEventListener('keydown', handleEscape);
            document.addEventListener('click', handleBackgroundClick);
            
            // Start file upload progress simulation
            const interval = setInterval(() => {
                setUploadProgress(prev => {
                    if (prev >= 100) {
                        clearInterval(interval);
                        return 100;
                    }
                    return prev + 10;
                });
            }, 500);
            
            // Missing cleanup for interval and event listeners
        }
    }, [isOpen]);
    
    const handleEscape = (e) => {
        if (e.key === 'Escape') {
            onClose();
        }
    };
    
    const handleBackgroundClick = (e) => {
        if (e.target === e.currentTarget) {
            onClose();
        }
    };
    
    return isOpen ? (
        <div className="modal-overlay" onClick={handleBackgroundClick}>
            <div className="modal-content">
                <button className="close-btn" onClick={onClose}>×</button>
                {children}
            </div>
        </div>
    ) : null;
};
                """,
                "ui_events": "modal opens → file upload starts → user closes modal → listeners not cleaned up → memory leak",
                "expected_issues": ["memory leak", "event listener cleanup", "useEffect dependencies", "interval cleanup"]
            },
            
            {
                "id": "RW_05",
                "title": "Infinite Scroll Performance Degradation",
                "description": """An infinite scroll feed component starts working fine but after scrolling through many items (50+), the scrolling becomes choppy and unresponsive. The component loads images and renders complex cards with animations. The issue seems to be that old items are not being removed from the DOM, causing the page to become bloated.""",
                "code_context": """
// Infinite scroll component with performance issues
const InfiniteFeed = () => {
    const [items, setItems] = useState([]);
    const [page, setPage] = useState(1);
    const [loading, setLoading] = useState(false);
    
    const loadMoreItems = async () => {
        setLoading(true);
        const newItems = await api.getItems(page);
        setItems(prev => [...prev, ...newItems]); // Accumulating all items
        setPage(prev => prev + 1);
        setLoading(false);
    };
    
    useEffect(() => {
        const handleScroll = () => {
            if (window.innerHeight + window.scrollY >= document.body.offsetHeight - 1000) {
                loadMoreItems();
            }
        };
        
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);
    
    return (
        <div className="feed-container">
            {items.map(item => (
                <FeedCard 
                    key={item.id}
                    item={item}
                    // Each card has complex animations and image loading
                />
            ))}
            {loading && <LoadingSpinner />}
        </div>
    );
};

const FeedCard = ({ item }) => {
    const [imageLoaded, setImageLoaded] = useState(false);
    
    useEffect(() => {
        // Complex animation logic
        const animateCard = () => {
            // Expensive DOM manipulations
        };
        
        if (imageLoaded) {
            animateCard();
        }
    }, [imageLoaded]);
    
    return (
        <div className={`feed-card ${imageLoaded ? 'loaded' : 'loading'}`}>
            <img 
                src={item.imageUrl}
                onLoad={() => setImageLoaded(true)}
                alt={item.title}
            />
            <h3>{item.title}</h3>
            <p>{item.description}</p>
            {/* More complex content */}
        </div>
    );
};
                """,
                "ui_events": "user scrolls → new items load → old items stay in DOM → performance degrades → scrolling becomes choppy",
                "expected_issues": ["virtualization", "DOM cleanup", "memory management", "scroll performance", "image optimization"]
            }
        ]
        return scenarios
    
    def run_real_world_test(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run a test on a real-world scenario"""
        start_time = time.time()
        
        prompt = f"""You are a senior frontend engineer debugging a complex production issue. Given this real-world bug scenario:

BUG REPORT: {scenario['description']}

CODE CONTEXT:
{scenario['code_context']}

UI EVENTS: {scenario['ui_events']}

TASK:
1. Identify the root cause(s) of this bug
2. Propose a comprehensive fix that addresses all aspects
3. Explain the technical details of why this bug occurs
4. Suggest additional improvements for robustness and performance
5. Consider accessibility and cross-browser compatibility

Provide your response in this format:
ROOT CAUSE ANALYSIS: [detailed technical explanation]
COMPREHENSIVE SOLUTION: [complete code fix with explanations]
WHY THIS OCCURS: [technical details about the underlying issue]
IMPROVEMENTS: [additional robustness and performance improvements]
ACCESSIBILITY: [accessibility considerations]
TESTING: [how to test this fix]"""

        try:
            response = self.llm_client.complete(prompt)
            processing_time = time.time() - start_time
            
            return {
                "scenario_id": scenario["id"],
                "title": scenario["title"],
                "description": scenario["description"],
                "llm_response": response,
                "processing_time": processing_time,
                "model_used": self.config["llm"]["model"],
                "temperature": self.config["llm"]["temperature"],
                "timestamp": datetime.now().isoformat(),
                "expected_issues": scenario["expected_issues"]
            }
            
        except Exception as e:
            print(f"Error testing scenario {scenario['id']}: {e}")
            return None
    
    def run_all_real_world_tests(self) -> List[Dict[str, Any]]:
        """Run all real-world scenario tests"""
        scenarios = self.get_real_world_scenarios()
        results = []
        
        print(f"🧪 Real-World GUI Bug Repair Tests")
        print(f"=" * 50)
        print(f"Testing {len(scenarios)} complex scenarios...")
        
        for i, scenario in enumerate(scenarios):
            print(f"\n--- Scenario {i+1}/{len(scenarios)}: {scenario['title']} ---")
            print(f"Expected issues: {', '.join(scenario['expected_issues'])}")
            
            result = self.run_real_world_test(scenario)
            if result:
                results.append(result)
                print(f"✅ Completed (took {result['processing_time']:.2f}s)")
            else:
                print(f"❌ Failed")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]]) -> str:
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"real_world_test_results_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")
        return str(filepath)
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> None:
        """Analyze and display test results"""
        if not results:
            print("No results to analyze")
            return
        
        print(f"\n📊 REAL-WORLD TEST ANALYSIS")
        print(f"=" * 50)
        
        # Basic stats
        total_time = sum(r['processing_time'] for r in results)
        avg_response_length = sum(len(r['llm_response']) for r in results) / len(results)
        
        print(f"Total tests: {len(results)}")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Average processing time: {total_time/len(results):.2f}s")
        print(f"Average response length: {avg_response_length:.0f} characters")
        
        # Analyze coverage of expected issues
        print(f"\n🎯 EXPECTED ISSUE COVERAGE:")
        all_expected_issues = set()
        for result in results:
            all_expected_issues.update(result['expected_issues'])
        
        for issue in sorted(all_expected_issues):
            coverage = 0
            for result in results:
                if issue.lower() in result['llm_response'].lower():
                    coverage += 1
            coverage_pct = (coverage / len(results)) * 100
            print(f"  {issue}: {coverage_pct:.1f}% coverage")
        
        # Show best and worst responses
        print(f"\n🏆 BEST PERFORMING SCENARIO:")
        best_result = max(results, key=lambda r: len(r['llm_response']))
        print(f"  {best_result['title']}")
        print(f"  Response length: {len(best_result['llm_response'])} characters")
        print(f"  Processing time: {best_result['processing_time']:.2f}s")
        
        print(f"\n📝 SAMPLE RESPONSE PREVIEW:")
        print(f"  {best_result['llm_response'][:300]}...")

def main():
    """Main function to run real-world tests"""
    print("🚀 Starting Real-World GUI Bug Repair Experiments")
    
    tester = RealWorldBugTester()
    
    # Run tests
    results = tester.run_all_real_world_tests()
    
    if results:
        # Save results
        results_file = tester.save_results(results)
        
        # Analyze results
        tester.analyze_results(results)
        
        print(f"\n✅ Real-world testing complete!")
        print(f"📊 Results saved to: {results_file}")
    else:
        print("❌ No tests completed successfully")

if __name__ == "__main__":
    main()
