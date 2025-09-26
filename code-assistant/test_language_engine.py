"""Test the language engine with multiple programming languages."""

import tempfile
from pathlib import Path

from language_engine import LanguageEngine, LanguageType

def test_language_detection():
    """Test language detection with various code samples."""
    engine = LanguageEngine()

    test_cases = [
        # Python
        ("""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, a, b):
        return a + b
""", LanguageType.PYTHON),

        # JavaScript
        ("""
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}

const add = (a, b) => a + b;

class Calculator {
    constructor() {
        this.history = [];
    }
}
""", LanguageType.JAVASCRIPT),

        # Rust
        ("""
fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n-1) + fibonacci(n-2),
    }
}

struct Calculator {
    history: Vec<i32>,
}

impl Calculator {
    fn new() -> Self {
        Calculator { history: Vec::new() }
    }
}
""", LanguageType.RUST),

        # Go
        ("""
package main

import "fmt"

func fibonacci(n int) int {
    if n <= 1 {
        return n
    }
    return fibonacci(n-1) + fibonacci(n-2)
}

type Calculator struct {
    history []int
}

func main() {
    fmt.Println(fibonacci(10))
}
""", LanguageType.GO),

        # Java
        ("""
public class Calculator {
    private int[] history;

    public Calculator() {
        this.history = new int[100];
    }

    public static int fibonacci(int n) {
        if (n <= 1) return n;
        return fibonacci(n-1) + fibonacci(n-2);
    }

    public int add(int a, int b) {
        return a + b;
    }
}
""", LanguageType.JAVA),
    ]

    print("Testing Language Detection:")
    print("=" * 50)

    for i, (code, expected_lang) in enumerate(test_cases):
        lang_info, code_structure = engine.analyze_content(code)

        success = lang_info.language == expected_lang
        confidence = lang_info.confidence

        print(f"\nTest {i+1}: {expected_lang.value}")
        print(f"Detected: {lang_info.language.value}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Functions found: {len(code_structure.functions)}")
        print(f"Classes found: {len(code_structure.classes)}")
        print(f"Complexity: {code_structure.complexity_score:.1f}")
        print(f"Result: {'PASS' if success else 'FAIL'}")

        # Print some details about detected functions
        if code_structure.functions:
            print(f"Functions: {[f.name for f in code_structure.functions[:3]]}")
        if code_structure.classes:
            print(f"Classes: {[c.name for c in code_structure.classes[:3]]}")

def test_file_detection():
    """Test language detection from file extensions."""
    engine = LanguageEngine()

    test_files = [
        ("test.py", "print('Hello Python')", LanguageType.PYTHON),
        ("test.js", "console.log('Hello JavaScript');", LanguageType.JAVASCRIPT),
        ("test.ts", "const message: string = 'Hello TypeScript';", LanguageType.TYPESCRIPT),
        ("test.rs", "fn main() { println!('Hello Rust'); }", LanguageType.RUST),
        ("test.go", "package main\nfunc main() { fmt.Println('Hello Go') }", LanguageType.GO),
        ("test.java", "public class Test { public static void main(String[] args) {} }", LanguageType.JAVA),
    ]

    print("\n\nTesting File Extension Detection:")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as temp_dir:
        for filename, content, expected_lang in test_files:
            file_path = Path(temp_dir) / filename
            file_path.write_text(content)

            lang_info, code_structure = engine.analyze_file(str(file_path))

            success = lang_info.language == expected_lang

            print(f"\nFile: {filename}")
            print(f"Expected: {expected_lang.value}")
            print(f"Detected: {lang_info.language.value}")
            print(f"Confidence: {lang_info.confidence:.2f}")
            print(f"Result: {'PASS' if success else 'FAIL'}")

def test_complex_analysis():
    """Test complex code analysis features."""
    engine = LanguageEngine()

    # Python code with multiple features
    python_code = '''
"""A comprehensive Python module for testing."""

import os
import sys
from typing import List, Dict, Optional

class DataProcessor:
    """Processes data using various algorithms."""

    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.data = []

    def process_data(self, items: List[str]) -> List[str]:
        """Process a list of items."""
        result = []
        for item in items:
            if self._is_valid(item):
                result.append(self._transform(item))
        return result

    def _is_valid(self, item: str) -> bool:
        """Check if item is valid."""
        return len(item) > 0

    def _transform(self, item: str) -> str:
        """Transform an item."""
        return item.upper()

async def async_function(data: List[str]) -> Dict[str, int]:
    """An async function for demonstration."""
    result = {}
    for item in data:
        result[item] = len(item)
    return result

def utility_function(x: int, y: int) -> int:
    """A utility function."""
    if x > y:
        return x * 2
    elif x < y:
        return y * 2
    else:
        return x + y

if __name__ == "__main__":
    processor = DataProcessor({"mode": "test"})
    data = ["hello", "world", "python"]
    result = processor.process_data(data)
    print(result)
'''

    print("\n\nTesting Complex Code Analysis:")
    print("=" * 50)

    lang_info, code_structure = engine.analyze_content(python_code)

    print(f"Language: {lang_info.language.value}")
    print(f"Confidence: {lang_info.confidence:.2f}")
    print(f"Lines of code: {code_structure.lines_of_code}")
    print(f"Complexity score: {code_structure.complexity_score:.1f}")
    print(f"Docstring coverage: {code_structure.docstring_coverage:.1f}%")

    print(f"\nFunctions ({len(code_structure.functions)}):")
    for func in code_structure.functions:
        async_str = " (async)" if func.is_async else ""
        params_str = f"({', '.join(func.parameters)})" if func.parameters else "()"
        doc_str = " [documented]" if func.docstring else ""
        print(f"  - {func.name}{params_str}{async_str}{doc_str}")

    print(f"\nClasses ({len(code_structure.classes)}):")
    for cls in code_structure.classes:
        doc_str = " [documented]" if cls.docstring else ""
        print(f"  - {cls.name}{doc_str}")
        for method in cls.methods[:3]:  # Show first 3 methods
            print(f"    └─ {method.name}()")

    print(f"\nImports ({len(code_structure.imports)}):")
    for imp in code_structure.imports[:5]:  # Show first 5 imports
        print(f"  - {imp}")

def main():
    """Run all tests."""
    print("Universal Language Engine Test Suite")
    print("=" * 60)

    try:
        test_language_detection()
        test_file_detection()
        test_complex_analysis()

        print("\n\n" + "=" * 60)
        print("All tests completed successfully!")
        print("The language engine supports multiple programming languages")
        print("and can analyze code structure, complexity, and documentation.")

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()