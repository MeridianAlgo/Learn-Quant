#!/usr/bin/env python3
"""
Algorithm Learning Platform Launcher
Integrates all Python learning algorithms with the Next.js website for a complete learning experience.
"""

import os
import sys
import subprocess
import webbrowser
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
import json

class LearningPlatformLauncher:
    """Main launcher for the Algorithm Learning Platform."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.nextjs_dir = self.base_dir / "algorithm-learning-platform"
        self.python_utils_dir = self.base_dir
        self.processes = []
        
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed."""
        print("🔍 Checking dependencies...")
        
        # Check Python packages
        required_packages = ['numpy', 'pandas', 'matplotlib', 'scipy']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package} is installed")
            except ImportError:
                missing_packages.append(package)
                print(f"❌ {package} is missing")
        
        if missing_packages:
            print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install"
                ] + missing_packages)
                print("✅ All packages installed successfully")
            except subprocess.CalledProcessError:
                print("❌ Failed to install packages")
                return False
        
        # Check Node.js and npm
        try:
            node_version = subprocess.check_output(["node", "--version"], text=True).strip()
            npm_version = subprocess.check_output(["npm", "--version"], text=True).strip()
            print(f"✅ Node.js: {node_version}")
            print(f"✅ npm: {npm_version}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Node.js and npm are required for the Next.js website")
            print("Please install Node.js from https://nodejs.org/")
            return False
        
        return True
    
    def scan_python_algorithms(self) -> Dict[str, Dict]:
        """Scan and catalog all Python algorithm implementations."""
        print("🔍 Scanning Python algorithms...")
        
        algorithms = {}
        
        # Define algorithm directories
        algorithm_dirs = [
            "UTILS - Algorithms - Sorting",
            "UTILS - Algorithms - Searching", 
            "UTILS - Algorithms - Graph",
            "UTILS - Algorithms - Dynamic Programming",
            "UTILS - Algorithms - Machine Learning"
        ]
        
        for dir_name in algorithm_dirs:
            dir_path = self.python_utils_dir / dir_name
            if dir_path.exists():
                print(f"📁 Scanning {dir_name}...")
                
                for py_file in dir_path.glob("*.py"):
                    if py_file.name != "__init__.py":
                        algorithm_info = self.analyze_python_file(py_file)
                        algorithms[py_file.stem] = algorithm_info
        
        print(f"✅ Found {len(algorithms)} Python algorithms")
        return algorithms
    
    def analyze_python_file(self, file_path: Path) -> Dict:
        """Analyze a Python file to extract algorithm information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract functions
            functions = []
            lines = content.split('\n')
            for line in lines:
                if line.strip().startswith('def '):
                    func_name = line.strip()[4:].split('(')[0]
                    functions.append(func_name)
            
            # Extract docstrings
            docstrings = []
            in_docstring = False
            current_docstring = []
            
            for line in lines:
                if '"""' in line:
                    if not in_docstring:
                        in_docstring = True
                        current_docstring = [line.replace('"""', '').strip()]
                    else:
                        current_docstring.append(line.replace('"""', '').strip())
                        docstrings.append('\n'.join(current_docstring))
                        in_docstring = False
                elif in_docstring:
                    current_docstring.append(line.strip())
            
            return {
                'file': str(file_path),
                'functions': functions,
                'docstrings': docstrings,
                'size': len(content),
                'lines': len(lines)
            }
            
        except Exception as e:
            return {
                'file': str(file_path),
                'error': str(e),
                'functions': [],
                'docstrings': []
            }
    
    def start_nextjs_app(self) -> bool:
        """Start the Next.js learning platform."""
        print("🚀 Starting Next.js learning platform...")
        
        if not self.nextjs_dir.exists():
            print("❌ Next.js directory not found")
            return False
        
        try:
            # Install dependencies if needed
            if not (self.nextjs_dir / "node_modules").exists():
                print("📦 Installing Next.js dependencies...")
                subprocess.run(["npm", "install"], cwd=self.nextjs_dir, check=True)
            
            # Start the development server
            print("🌐 Starting development server...")
            process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=self.nextjs_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=True
            )
            
            self.processes.append(process)
            
            # Wait for server to start
            time.sleep(5)
            
            # Open browser
            webbrowser.open("http://localhost:3000")
            print("✅ Learning platform opened in browser")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to start Next.js app: {e}")
            return False
    
    def run_python_demonstrations(self, algorithms: Dict[str, Dict]):
        """Run demonstrations of Python algorithms."""
        print("🐍 Running Python algorithm demonstrations...")
        
        for name, info in algorithms.items():
            if 'error' not in info:
                print(f"\n🔄 Running {name}...")
                try:
                    # Run the Python file
                    result = subprocess.run([
                        sys.executable, info['file']
                    ], capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        print(f"✅ {name} executed successfully")
                        # Show first few lines of output
                        output_lines = result.stdout.split('\n')[:5]
                        for line in output_lines:
                            if line.strip():
                                print(f"   {line}")
                    else:
                        print(f"❌ {name} failed: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    print(f"⏰ {name} timed out")
                except Exception as e:
                    print(f"❌ {name} error: {e}")
    
    def create_integration_report(self, algorithms: Dict[str, Dict]) -> str:
        """Create a comprehensive integration report."""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_algorithms': len(algorithms),
            'categories': {},
            'functions': [],
            'errors': []
        }
        
        for name, info in algorithms.items():
            if 'error' in info:
                report['errors'].append(f"{name}: {info['error']}")
            else:
                # Categorize by directory
                category = name.split('_')[0] if '_' in name else 'Other'
                if category not in report['categories']:
                    report['categories'][category] = []
                report['categories'][category].append(name)
                
                # Add functions
                for func in info['functions']:
                    report['functions'].append(f"{name}.{func}")
        
        # Save report
        report_path = self.base_dir / "integration_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return str(report_path)
    
    def cleanup(self):
        """Clean up processes and resources."""
        print("🧹 Cleaning up...")
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
    
    def run_interactive_mode(self):
        """Run the platform in interactive mode."""
        print("\n🎯 Algorithm Learning Platform - Interactive Mode")
        print("=" * 50)
        
        while True:
            print("\nOptions:")
            print("1. 🌐 Open Next.js learning platform")
            print("2. 🐍 Run Python algorithm demonstrations")
            print("3. 📊 Show algorithm catalog")
            print("4. 📋 Generate integration report")
            print("5. 🚪 Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                self.start_nextjs_app()
            elif choice == '2':
                algorithms = self.scan_python_algorithms()
                self.run_python_demonstrations(algorithms)
            elif choice == '3':
                algorithms = self.scan_python_algorithms()
                print(f"\n📚 Algorithm Catalog ({len(algorithms)} algorithms):")
                for name, info in algorithms.items():
                    if 'error' not in info:
                        print(f"  ✅ {name}: {len(info['functions'])} functions")
                    else:
                        print(f"  ❌ {name}: Error")
            elif choice == '4':
                algorithms = self.scan_python_algorithms()
                report_path = self.create_integration_report(algorithms)
                print(f"📋 Integration report saved to: {report_path}")
            elif choice == '5':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def run_full_demo(self):
        """Run a complete demonstration of the platform."""
        print("🎪 Running full platform demonstration...")
        
        # Check dependencies
        if not self.check_dependencies():
            return
        
        # Scan algorithms
        algorithms = self.scan_python_algorithms()
        
        # Create report
        report_path = self.create_integration_report(algorithms)
        print(f"📋 Integration report: {report_path}")
        
        # Start Next.js app
        if self.start_nextjs_app():
            print("\n🌐 Next.js platform is running at http://localhost:3000")
            print("🐍 Python algorithms are available in the UTILS folders")
            print("\nPress Ctrl+C to stop the platform")
            
            try:
                # Keep running
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping platform...")
        
        self.cleanup()

def main():
    """Main entry point."""
    launcher = LearningPlatformLauncher()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "interactive":
            launcher.run_interactive_mode()
        elif command == "demo":
            launcher.run_full_demo()
        elif command == "web":
            launcher.start_nextjs_app()
        elif command == "scan":
            algorithms = launcher.scan_python_algorithms()
            print(f"Found {len(algorithms)} algorithms")
        else:
            print("Usage: python run_learning_platform.py [interactive|demo|web|scan]")
    else:
        # Default to interactive mode
        launcher.run_interactive_mode()

if __name__ == "__main__":
    main()
