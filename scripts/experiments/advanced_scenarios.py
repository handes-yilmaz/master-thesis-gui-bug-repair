#!/usr/bin/env python
"""
Advanced Bug Scenarios for GUI Bug Repair Research
=================================================

This module contains sophisticated bug scenarios focusing on:
- Security vulnerabilities (XSS, CSRF, authentication bypass)
- Performance bottlenecks (memory leaks, render blocking, bundle size)
- Cross-browser compatibility issues
- Mobile-specific problems
- Real-time collaboration bugs
- Progressive Web App issues
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path
import json

@dataclass
class AdvancedBugScenario:
    """Advanced bug scenario with comprehensive details"""
    bug_id: str
    title: str
    description: str
    expected_solution: str
    bug_category: str
    severity: str
    difficulty: str
    ui_context: Optional[str] = None
    code_snippet: Optional[str] = None
    security_implications: Optional[List[str]] = None
    performance_impact: Optional[str] = None
    browser_specific: Optional[List[str]] = None
    mobile_affected: bool = False
    accessibility_impact: Optional[str] = None
    testing_scenarios: Optional[List[str]] = None
    fix_priority: str = "medium"

class AdvancedScenariosGenerator:
    """Generates advanced bug scenarios for comprehensive testing"""
    
    def __init__(self):
        self.scenarios = []
    
    def generate_security_scenarios(self) -> List[AdvancedBugScenario]:
        """Generate security-focused bug scenarios"""
        return [
            AdvancedBugScenario(
                bug_id="SEC_XSS_01",
                title="XSS Vulnerability in Dynamic Content Rendering",
                description="User-generated content is rendered using dangerouslySetInnerHTML without proper sanitization, allowing malicious scripts to execute in the browser context.",
                expected_solution="Implement proper input sanitization, use DOMPurify library, add CSP headers",
                bug_category="security",
                severity="critical",
                difficulty="high",
                ui_context="User comment system with rich text display",
                code_snippet="""
function CommentDisplay({ comment }) {
    // VULNERABLE: Direct HTML injection
    return (
        <div 
            dangerouslySetInnerHTML={{ __html: comment.content }}
            className="comment-content"
        />
    );
}

// Vulnerable usage
<CommentDisplay comment={{ content: '<script>alert("XSS")</script>' }} />
""",
                security_implications=["XSS", "data theft", "session hijacking", "malware injection"],
                fix_priority="immediate"
            ),
            
            AdvancedBugScenario(
                bug_id="SEC_CSRF_02",
                title="CSRF Token Missing in State-Changing Operations",
                description="API endpoints that modify user data or application state lack CSRF protection, allowing malicious sites to perform unauthorized actions on behalf of authenticated users.",
                expected_solution="Implement CSRF tokens, validate Origin/Referer headers, use SameSite cookies",
                bug_category="security",
                severity="high",
                difficulty="medium",
                ui_context="User profile update form and payment processing",
                code_snippet="""
// VULNERABLE: No CSRF protection
async function updateProfile(userData) {
    const response = await fetch('/api/profile/update', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${getToken()}` // Missing CSRF token
        },
        body: JSON.stringify(userData)
    });
    return response.json();
}
""",
                security_implications=["CSRF", "unauthorized data modification", "account takeover"],
                fix_priority="high"
            ),
            
            AdvancedBugScenario(
                bug_id="SEC_AUTH_03",
                title="Authentication Bypass via Client-Side Validation",
                description="Critical authentication checks are performed only on the client side, allowing users to bypass security by manipulating browser developer tools or network requests.",
                expected_solution="Implement server-side authentication, validate all requests, use secure session management",
                bug_category="security",
                severity="critical",
                difficulty="medium",
                ui_context="Admin panel access control",
                code_snippet="""
// VULNERABLE: Client-side only validation
function AdminPanel() {
    const [isAdmin, setIsAdmin] = useState(false);
    
    useEffect(() => {
        // VULNERABLE: Can be manipulated in browser
        const userRole = localStorage.getItem('userRole');
        setIsAdmin(userRole === 'admin');
    }, []);
    
    if (!isAdmin) {
        return <div>Access Denied</div>;
    }
    
    return <AdminDashboard />;
}
""",
                security_implications=["authentication bypass", "privilege escalation", "data breach"],
                fix_priority="immediate"
            )
        ]
    
    def generate_performance_scenarios(self) -> List[AdvancedBugScenario]:
        """Generate performance-focused bug scenarios"""
        return [
            AdvancedBugScenario(
                bug_id="PERF_MEMORY_04",
                title="Memory Leak in Event Listeners",
                description="Event listeners are added to DOM elements but never removed, causing memory leaks that degrade performance over time, especially in single-page applications.",
                expected_solution="Properly cleanup event listeners, use AbortController, implement component unmounting",
                bug_category="performance",
                severity="medium",
                difficulty="medium",
                ui_context="Dynamic list with click handlers",
                code_snippet="""
function DynamicList({ items }) {
    useEffect(() => {
        const handleClick = (e) => {
            console.log('Item clicked:', e.target.dataset.id);
        };
        
        // VULNERABLE: Event listeners never cleaned up
        document.addEventListener('click', handleClick);
        
        // Missing cleanup function
        // return () => document.removeEventListener('click', handleClick);
    }, [items]);
    
    return (
        <div>
            {items.map(item => (
                <div key={item.id} data-id={item.id}>
                    {item.name}
                </div>
            ))}
        </div>
    );
}
""",
                performance_impact="Memory usage grows linearly with component instances, eventual browser crash",
                fix_priority="high"
            ),
            
            AdvancedBugScenario(
                bug_id="PERF_RENDER_05",
                title="Render Blocking CSS and JavaScript",
                description="Critical CSS and JavaScript files are loaded synchronously, blocking the initial page render and significantly increasing First Contentful Paint (FCP) time.",
                expected_solution="Implement critical CSS inlining, defer non-critical JS, use resource hints",
                bug_category="performance",
                severity="medium",
                difficulty="low",
                ui_context="Main application bundle loading",
                code_snippet="""
<!-- VULNERABLE: Render blocking resources -->
<head>
    <!-- Blocking CSS -->
    <link rel="stylesheet" href="/styles/main.css" />
    
    <!-- Blocking JavaScript -->
    <script src="/js/app.js"></script>
    
    <!-- No resource hints or optimization -->
</head>
""",
                performance_impact="FCP increased by 2-3 seconds, poor Core Web Vitals scores",
                fix_priority="medium"
            ),
            
            AdvancedBugScenario(
                bug_id="PERF_BUNDLE_06",
                title="Excessive Bundle Size with Unused Dependencies",
                description="JavaScript bundle includes large libraries and dependencies that are not actually used in the application, significantly increasing download and parse time.",
                expected_solution="Implement tree shaking, code splitting, analyze and remove unused dependencies",
                bug_category="performance",
                severity="medium",
                difficulty="low",
                ui_context="Production build with large bundle size",
                code_snippet="""
// VULNERABLE: Importing entire libraries
import * as lodash from 'lodash';  // Imports entire library
import { Button } from '@mui/material';  // Imports entire Material-UI

// Better approach:
// import { debounce } from 'lodash/debounce';
// import Button from '@mui/material/Button';
""",
                performance_impact="Bundle size 2-3x larger than necessary, slower page loads",
                fix_priority="medium"
            )
        ]
    
    def generate_cross_browser_scenarios(self) -> List[AdvancedBugScenario]:
        """Generate cross-browser compatibility bug scenarios"""
        return [
            AdvancedBugScenario(
                bug_id="BROWSER_FLEXBOX_07",
                title="Flexbox Layout Inconsistencies Across Browsers",
                description="Flexbox layouts render differently in Safari, Firefox, and Chrome due to varying implementations of flexbox standards, causing visual misalignment.",
                expected_solution="Add vendor prefixes, use flexbox fallbacks, implement browser-specific CSS",
                bug_category="cross_browser",
                severity="medium",
                difficulty="medium",
                ui_context="Responsive grid layout system",
                code_snippet="""
.grid-container {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
}

.grid-item {
    flex: 1 1 300px;
    /* Missing vendor prefixes and fallbacks */
}

/* Safari-specific issues with flex-basis */
@supports (-webkit-appearance: none) {
    .grid-item {
        flex-basis: 300px;
        min-width: 300px;
    }
}
""",
                browser_specific=["Safari", "Firefox", "Edge"],
                mobile_affected=True,
                fix_priority="medium"
            ),
            
            AdvancedBugScenario(
                bug_id="BROWSER_ES6_08",
                title="ES6+ Features Not Supported in Older Browsers",
                description="Modern JavaScript features like arrow functions, destructuring, and async/await are used without proper transpilation, causing syntax errors in older browsers.",
                expected_solution="Configure Babel transpilation, add polyfills, implement feature detection",
                bug_category="cross_browser",
                severity="high",
                difficulty="low",
                ui_context="Modern React application with ES6+ syntax",
                code_snippet="""
// VULNERABLE: ES6+ features without transpilation
const UserProfile = ({ user }) => {
    const { name, email, preferences = {} } = user;  // Destructuring
    
    const handleSubmit = async (formData) => {  // Async/await
        try {
            const response = await api.updateProfile(formData);
            return response.data;
        } catch (error) {
            console.error('Update failed:', error);
        }
    };
    
    return (
        <form onSubmit={handleSubmit}>
            <input defaultValue={name} />
            <input defaultValue={email} />
        </form>
    );
};
""",
                browser_specific=["IE11", "Safari < 10", "Chrome < 50"],
                fix_priority="high"
            ),
            
            AdvancedBugScenario(
                bug_id="BROWSER_CSS_09",
                title="CSS Grid and Custom Properties Browser Support Issues",
                description="CSS Grid layouts and CSS custom properties (variables) are not supported in older browsers, causing complete layout failures and missing styles.",
                expected_solution="Implement CSS Grid fallbacks, use PostCSS autoprefixer, add feature queries",
                bug_category="cross_browser",
                severity="medium",
                difficulty="medium",
                ui_context="Modern CSS Grid-based dashboard layout",
                code_snippet="""
:root {
    --primary-color: #007bff;
    --secondary-color: #6c757d;
    --grid-gap: 20px;
}

.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: var(--grid-gap);
    /* Missing fallback for older browsers */
}

/* Should include fallback */
.dashboard-grid {
    display: flex;
    flex-wrap: wrap;
}

@supports (display: grid) {
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: var(--grid-gap);
    }
}
""",
                browser_specific=["IE11", "Safari < 10.1", "Chrome < 57"],
                fix_priority="medium"
            )
        ]
    
    def generate_mobile_scenarios(self) -> List[AdvancedBugScenario]:
        """Generate mobile-specific bug scenarios"""
        return [
            AdvancedBugScenario(
                bug_id="MOBILE_TOUCH_10",
                title="Touch Event Handling Issues on Mobile Devices",
                description="Touch events are not properly handled on mobile devices, causing tap delays, double-tap zoom issues, and poor touch responsiveness.",
                expected_solution="Implement proper touch event handling, add viewport meta tags, use touch-action CSS",
                bug_category="mobile",
                severity="medium",
                difficulty="medium",
                ui_context="Interactive buttons and form elements",
                code_snippet="""
// VULNERABLE: Missing touch event handling
function TouchButton({ onClick, children }) {
    return (
        <button 
            onClick={onClick}
            className="touch-button"
            // Missing touch event handlers
        >
            {children}
        </button>
    );
}

/* Missing touch-specific CSS */
.touch-button {
    /* Should include: */
    /* touch-action: manipulation; */
    /* -webkit-tap-highlight-color: transparent; */
}
""",
                mobile_affected=True,
                accessibility_impact="Poor touch experience for mobile users",
                fix_priority="medium"
            ),
            
            AdvancedBugScenario(
                bug_id="MOBILE_VIEWPORT_11",
                title="Viewport and Scaling Issues on Mobile Devices",
                description="Mobile viewport is not properly configured, causing content to be too small, horizontal scrolling, or improper scaling across different device sizes.",
                expected_solution="Add proper viewport meta tags, implement responsive design, test on various devices",
                bug_category="mobile",
                severity="high",
                difficulty="low",
                ui_context="Responsive web application",
                code_snippet="""
<!-- VULNERABLE: Missing or incorrect viewport configuration -->
<head>
    <!-- Missing viewport meta tag -->
    <!-- <meta name="viewport" content="width=device-width, initial-scale=1.0"> -->
    
    <!-- Or incorrect configuration -->
    <!-- <meta name="viewport" content="width=1024"> -->
</head>

/* Missing mobile-first CSS approach */
.container {
    width: 1200px;  /* Fixed width causes horizontal scroll on mobile */
    margin: 0 auto;
}
""",
                mobile_affected=True,
                browser_specific=["Mobile Safari", "Chrome Mobile", "Samsung Internet"],
                fix_priority="high"
            )
        ]
    
    def generate_real_time_scenarios(self) -> List[AdvancedBugScenario]:
        """Generate real-time collaboration bug scenarios"""
        return [
            AdvancedBugScenario(
                bug_id="REALTIME_WEBSOCKET_12",
                title="WebSocket Connection State Management Issues",
                description="WebSocket connections are not properly managed, causing connection drops, reconnection failures, and data synchronization issues in real-time applications.",
                expected_solution="Implement connection state management, automatic reconnection, heartbeat monitoring",
                bug_category="real_time",
                severity="high",
                difficulty="high",
                ui_context="Real-time chat or collaboration application",
                code_snippet="""
// VULNERABLE: Poor WebSocket connection management
class WebSocketManager {
    constructor(url) {
        this.ws = new WebSocket(url);
        this.ws.onopen = () => console.log('Connected');
        this.ws.onclose = () => console.log('Disconnected');
        // Missing reconnection logic, error handling, state management
    }
    
    send(data) {
        if (this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(data);
        }
        // No error handling or retry logic
    }
}
""",
                performance_impact="Frequent disconnections, poor user experience, data loss",
                fix_priority="high"
            ),
            
            AdvancedBugScenario(
                bug_id="REALTIME_STATE_13",
                title="Real-Time State Synchronization Conflicts",
                description="Multiple users editing the same content simultaneously cause state conflicts, data corruption, and inconsistent user experiences.",
                expected_solution="Implement operational transformation, conflict resolution, optimistic updates with rollback",
                bug_category="real_time",
                severity="high",
                difficulty="high",
                ui_context="Collaborative document editor",
                code_snippet="""
// VULNERABLE: No conflict resolution
function DocumentEditor({ documentId }) {
    const [content, setContent] = useState('');
    
    const handleChange = (newContent) => {
        setContent(newContent);
        
        // VULNERABLE: Direct update without conflict checking
        socket.emit('document:update', {
            id: documentId,
            content: newContent,
            timestamp: Date.now()
            // Missing: version, user ID, conflict resolution
        });
    };
    
    // No handling of concurrent updates from other users
}
""",
                performance_impact="Data corruption, poor collaboration experience",
                fix_priority="high"
            )
        ]
    
    def generate_pwa_scenarios(self) -> List[AdvancedBugScenario]:
        """Generate Progressive Web App bug scenarios"""
        return [
            AdvancedBugScenario(
                bug_id="PWA_SERVICE_WORKER_14",
                title="Service Worker Caching Strategy Issues",
                description="Service worker caching strategies are not properly implemented, causing offline functionality failures, stale content, and poor performance.",
                expected_solution="Implement proper caching strategies, version management, cache cleanup",
                bug_category="pwa",
                severity="medium",
                difficulty="high",
                ui_context="Progressive Web Application with offline support",
                code_snippet="""
// VULNERABLE: Poor caching strategy
self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                // VULNERABLE: Always return cached version if available
                return response || fetch(event.request);
                // Missing: cache-first strategy, version management, cache cleanup
            })
    );
});

// Missing: cache versioning, cleanup, offline fallback
""",
                performance_impact="Stale content, poor offline experience, storage bloat",
                fix_priority="medium"
            ),
            
            AdvancedBugScenario(
                bug_id="PWA_INSTALL_15",
                title="PWA Installation and Update Flow Issues",
                description="Progressive Web App installation prompts don't work properly, updates are not handled correctly, and users cannot access the app offline.",
                expected_solution="Implement proper install prompts, update notifications, offline functionality",
                bug_category="pwa",
                severity="medium",
                difficulty="medium",
                ui_context="PWA installation and update system",
                code_snippet="""
// VULNERABLE: Missing PWA installation handling
let deferredPrompt;

window.addEventListener('beforeinstallprompt', (e) => {
    // VULNERABLE: Prompt not stored or handled
    console.log('Install prompt available');
    // Missing: store prompt, show install button, handle user choice
});

// Missing: update notification, offline detection, install button
""",
                mobile_affected=True,
                accessibility_impact="Poor PWA experience for mobile users",
                fix_priority="medium"
            )
        ]
    
    def generate_all_scenarios(self) -> List[AdvancedBugScenario]:
        """Generate all advanced bug scenarios"""
        all_scenarios = []
        
        all_scenarios.extend(self.generate_security_scenarios())
        all_scenarios.extend(self.generate_performance_scenarios())
        all_scenarios.extend(self.generate_cross_browser_scenarios())
        all_scenarios.extend(self.generate_mobile_scenarios())
        all_scenarios.extend(self.generate_real_time_scenarios())
        all_scenarios.extend(self.generate_pwa_scenarios())
        
        return all_scenarios
    
    def save_scenarios_to_file(self, output_path: str = "data/processed/advanced_scenarios.json"):
        """Save all scenarios to a JSON file"""
        scenarios = self.generate_all_scenarios()
        
        # Convert to dictionary format
        scenarios_data = []
        for scenario in scenarios:
            scenario_dict = {
                "bug_id": scenario.bug_id,
                "title": scenario.title,
                "description": scenario.description,
                "expected_solution": scenario.expected_solution,
                "bug_category": scenario.bug_category,
                "severity": scenario.severity,
                "difficulty": scenario.difficulty,
                "ui_context": scenario.ui_context,
                "code_snippet": scenario.code_snippet,
                "security_implications": scenario.security_implications,
                "performance_impact": scenario.performance_impact,
                "browser_specific": scenario.browser_specific,
                "mobile_affected": scenario.mobile_affected,
                "accessibility_impact": scenario.accessibility_impact,
                "testing_scenarios": scenario.testing_scenarios,
                "fix_priority": scenario.fix_priority
            }
            scenarios_data.append(scenario_dict)
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(scenarios_data, f, indent=2)
        
        print(f"Generated {len(scenarios_data)} advanced bug scenarios")
        print(f"Saved to: {output_file}")
        
        return scenarios_data
    
    def print_scenario_summary(self):
        """Print a summary of all generated scenarios"""
        scenarios = self.generate_all_scenarios()
        
        print(f"\n{'='*60}")
        print("ADVANCED BUG SCENARIOS SUMMARY")
        print(f"{'='*60}")
        print(f"Total Scenarios: {len(scenarios)}")
        
        # Group by category
        categories = {}
        for scenario in scenarios:
            cat = scenario.bug_category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(scenario)
        
        for category, cat_scenarios in categories.items():
            print(f"\n{category.upper()} ({len(cat_scenarios)} scenarios):")
            for scenario in cat_scenarios:
                print(f"  - {scenario.bug_id}: {scenario.title} ({scenario.severity})")
        
        print(f"\n{'='*60}")

def main():
    """Main function to generate and save advanced scenarios"""
    generator = AdvancedScenariosGenerator()
    
    # Generate and save scenarios
    scenarios = generator.save_scenarios_to_file()
    
    # Print summary
    generator.print_scenario_summary()
    
    return scenarios

if __name__ == "__main__":
    main()
