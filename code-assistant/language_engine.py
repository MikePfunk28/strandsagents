"""Universal Language Engine for multi-language code analysis."""

import re
import ast
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class LanguageType(Enum):
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    CSHARP = "csharp"
    PHP = "php"
    RUBY = "ruby"
    UNKNOWN = "unknown"


@dataclass
class LanguageInfo:
    language: LanguageType
    confidence: float
    version: Optional[str] = None
    framework: Optional[str] = None
    features: List[str] = None

    def __post_init__(self):
        if self.features is None:
            self.features = []


@dataclass
class Function:
    name: str
    start_line: int
    end_line: int
    parameters: List[str]
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    is_async: bool = False
    is_static: bool = False
    visibility: str = "public"
    complexity: int = 1


@dataclass
class Class:
    name: str
    start_line: int
    end_line: int
    methods: List[Function]
    attributes: List[str]
    base_classes: List[str] = None
    docstring: Optional[str] = None
    is_abstract: bool = False

    def __post_init__(self):
        if self.base_classes is None:
            self.base_classes = []


@dataclass
class CodeStructure:
    language: LanguageType
    functions: List[Function]
    classes: List[Class]
    imports: List[str]
    variables: List[str]
    complexity_score: float
    lines_of_code: int
    docstring_coverage: float


class LanguageDetector:
    """Detects programming language from file content or extension."""

    # File extension mappings
    EXTENSION_MAP = {
        '.py': LanguageType.PYTHON,
        '.js': LanguageType.JAVASCRIPT,
        '.jsx': LanguageType.JAVASCRIPT,
        '.ts': LanguageType.TYPESCRIPT,
        '.tsx': LanguageType.TYPESCRIPT,
        '.rs': LanguageType.RUST,
        '.go': LanguageType.GO,
        '.java': LanguageType.JAVA,
        '.cpp': LanguageType.CPP,
        '.cc': LanguageType.CPP,
        '.cxx': LanguageType.CPP,
        '.c': LanguageType.C,
        '.h': LanguageType.C,
        '.cs': LanguageType.CSHARP,
        '.php': LanguageType.PHP,
        '.rb': LanguageType.RUBY,
    }

    # Language signature patterns
    LANGUAGE_PATTERNS = {
        LanguageType.PYTHON: [
            re.compile(r'^\s*def\s+\w+\s*\(', re.MULTILINE),
            re.compile(r'^\s*class\s+\w+.*:', re.MULTILINE),
            re.compile(r'^\s*import\s+\w+', re.MULTILINE),
            re.compile(r'^\s*from\s+\w+\s+import', re.MULTILINE),
            re.compile(r'if\s+__name__\s*==\s*["\']__main__["\']', re.MULTILINE),
        ],
        LanguageType.JAVASCRIPT: [
            re.compile(r'function\s+\w+\s*\(', re.MULTILINE),
            re.compile(r'const\s+\w+\s*=', re.MULTILINE),
            re.compile(r'let\s+\w+\s*=', re.MULTILINE),
            re.compile(r'var\s+\w+\s*=', re.MULTILINE),
            re.compile(r'=>\s*{', re.MULTILINE),
        ],
        LanguageType.TYPESCRIPT: [
            re.compile(r':\s*\w+\s*[=;]', re.MULTILINE),  # Type annotations
            re.compile(r'interface\s+\w+', re.MULTILINE),
            re.compile(r'type\s+\w+\s*=', re.MULTILINE),
            re.compile(r'export\s+(class|interface|type)', re.MULTILINE),
        ],
        LanguageType.RUST: [
            re.compile(r'fn\s+\w+\s*\(', re.MULTILINE),
            re.compile(r'struct\s+\w+', re.MULTILINE),
            re.compile(r'impl\s+\w+', re.MULTILINE),
            re.compile(r'use\s+\w+::', re.MULTILINE),
            re.compile(r'let\s+\w+\s*=', re.MULTILINE),
        ],
        LanguageType.GO: [
            re.compile(r'func\s+\w+\s*\(', re.MULTILINE),
            re.compile(r'package\s+\w+', re.MULTILINE),
            re.compile(r'import\s+\(', re.MULTILINE),
            re.compile(r'type\s+\w+\s+struct', re.MULTILINE),
        ],
        LanguageType.JAVA: [
            re.compile(r'public\s+class\s+\w+', re.MULTILINE),
            re.compile(r'public\s+static\s+void\s+main', re.MULTILINE),
            re.compile(r'import\s+\w+(\.\w+)*;', re.MULTILINE),
            re.compile(r'@\w+', re.MULTILINE),  # Annotations
        ],
        LanguageType.CPP: [
            re.compile(r'#include\s+<\w+>', re.MULTILINE),
            re.compile(r'class\s+\w+', re.MULTILINE),
            re.compile(r'namespace\s+\w+', re.MULTILINE),
            re.compile(r'::\w+', re.MULTILINE),  # Scope resolution
        ],
    }

    def detect_from_file(self, file_path: str) -> LanguageInfo:
        """Detect language from file path and content."""
        path = Path(file_path)

        # First try extension-based detection
        extension_lang = self.EXTENSION_MAP.get(path.suffix.lower())

        if not path.exists():
            return LanguageInfo(
                language=extension_lang or LanguageType.UNKNOWN,
                confidence=0.5 if extension_lang else 0.0
            )

        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
            content_detection = self.detect_from_content(content)

            # Combine extension and content detection
            if extension_lang and extension_lang == content_detection.language:
                # Both agree - high confidence
                return LanguageInfo(
                    language=extension_lang,
                    confidence=min(1.0, content_detection.confidence + 0.3)
                )
            elif extension_lang and content_detection.confidence < 0.5:
                # Trust extension when content detection is uncertain
                return LanguageInfo(
                    language=extension_lang,
                    confidence=0.7
                )
            else:
                # Trust content detection
                return content_detection

        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            return LanguageInfo(
                language=extension_lang or LanguageType.UNKNOWN,
                confidence=0.3 if extension_lang else 0.0
            )

    def detect_from_content(self, content: str) -> LanguageInfo:
        """Detect language from code content using pattern matching."""
        if not content.strip():
            return LanguageInfo(language=LanguageType.UNKNOWN, confidence=0.0)

        scores = {}
        total_lines = len(content.splitlines())

        # Score each language based on pattern matches
        for language, patterns in self.LANGUAGE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(pattern.findall(content))
                # Weight by pattern importance and normalize by content length
                score += matches * (10 / max(total_lines, 1))
            scores[language] = score

        if not scores:
            return LanguageInfo(language=LanguageType.UNKNOWN, confidence=0.0)

        # Find the best match
        best_language = max(scores.items(), key=lambda x: x[1])
        best_score = best_language[1]

        # Normalize confidence to 0-1 range
        confidence = min(1.0, best_score / 5.0)  # Arbitrary scaling factor

        return LanguageInfo(
            language=best_language[0],
            confidence=confidence
        )


class CodeAnalyzer:
    """Analyzes code structure for different programming languages."""

    def analyze_code(self, content: str, language: LanguageType) -> CodeStructure:
        """Analyze code structure based on language."""
        if language == LanguageType.PYTHON:
            return self._analyze_python(content)
        elif language in [LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT]:
            return self._analyze_javascript(content)
        elif language == LanguageType.RUST:
            return self._analyze_rust(content)
        elif language == LanguageType.GO:
            return self._analyze_go(content)
        elif language == LanguageType.JAVA:
            return self._analyze_java(content)
        else:
            return self._analyze_generic(content, language)

    def _analyze_python(self, content: str) -> CodeStructure:
        """Analyze Python code using AST."""
        try:
            tree = ast.parse(content)

            functions = []
            classes = []
            imports = []
            variables = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func = Function(
                        name=node.name,
                        start_line=node.lineno,
                        end_line=getattr(node, 'end_lineno', node.lineno),
                        parameters=[arg.arg for arg in node.args.args],
                        docstring=ast.get_docstring(node),
                        is_async=isinstance(node, ast.AsyncFunctionDef)
                    )
                    functions.append(func)

                elif isinstance(node, ast.ClassDef):
                    class_methods = [
                        Function(
                            name=n.name,
                            start_line=n.lineno,
                            end_line=getattr(n, 'end_lineno', n.lineno),
                            parameters=[arg.arg for arg in n.args.args],
                            docstring=ast.get_docstring(n),
                            is_async=isinstance(n, ast.AsyncFunctionDef)
                        )
                        for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]

                    cls = Class(
                        name=node.name,
                        start_line=node.lineno,
                        end_line=getattr(node, 'end_lineno', node.lineno),
                        methods=class_methods,
                        attributes=[],  # Would need more complex analysis
                        base_classes=[base.id for base in node.bases if isinstance(base, ast.Name)],
                        docstring=ast.get_docstring(node)
                    )
                    classes.append(cls)

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        imports.extend([alias.name for alias in node.names])
                    else:
                        imports.append(node.module or "")

            lines_of_code = len([line for line in content.splitlines() if line.strip()])
            complexity_score = self._calculate_complexity(content)
            docstring_coverage = self._calculate_docstring_coverage(functions, classes)

            return CodeStructure(
                language=LanguageType.PYTHON,
                functions=functions,
                classes=classes,
                imports=imports,
                variables=variables,
                complexity_score=complexity_score,
                lines_of_code=lines_of_code,
                docstring_coverage=docstring_coverage
            )

        except SyntaxError as e:
            logger.warning(f"Python syntax error: {e}")
            return self._analyze_generic(content, LanguageType.PYTHON)

    def _analyze_javascript(self, content: str) -> CodeStructure:
        """Analyze JavaScript/TypeScript code using regex patterns."""
        # Function detection
        func_pattern = re.compile(r'(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)\s*=>|\([^)]*\)\s*\{|function))', re.MULTILINE)
        class_pattern = re.compile(r'class\s+(\w+)(?:\s+extends\s+(\w+))?', re.MULTILINE)
        import_pattern = re.compile(r'(?:import\s+.*?from\s+["\']([^"\']+)["\']|import\s+["\']([^"\']+)["\']|const\s+.*?=\s*require\(["\']([^"\']+)["\']\))', re.MULTILINE)

        functions = []
        classes = []
        imports = []

        # Find functions
        for match in func_pattern.finditer(content):
            func_name = match.group(1) or match.group(2)
            if func_name:
                line_num = content[:match.start()].count('\n') + 1
                functions.append(Function(
                    name=func_name,
                    start_line=line_num,
                    end_line=line_num,  # Simplified
                    parameters=[],  # Would need more complex parsing
                ))

        # Find classes
        for match in class_pattern.finditer(content):
            class_name = match.group(1)
            base_class = match.group(2) if match.group(2) else None
            line_num = content[:match.start()].count('\n') + 1
            classes.append(Class(
                name=class_name,
                start_line=line_num,
                end_line=line_num,  # Simplified
                methods=[],
                attributes=[],
                base_classes=[base_class] if base_class else []
            ))

        # Find imports
        for match in import_pattern.finditer(content):
            import_name = match.group(1) or match.group(2) or match.group(3)
            if import_name:
                imports.append(import_name)

        lines_of_code = len([line for line in content.splitlines() if line.strip()])
        complexity_score = self._calculate_complexity(content)

        return CodeStructure(
            language=LanguageType.JAVASCRIPT,
            functions=functions,
            classes=classes,
            imports=imports,
            variables=[],
            complexity_score=complexity_score,
            lines_of_code=lines_of_code,
            docstring_coverage=0.0  # JS doesn't have docstrings like Python
        )

    def _analyze_rust(self, content: str) -> CodeStructure:
        """Analyze Rust code using regex patterns."""
        func_pattern = re.compile(r'fn\s+(\w+)\s*\(', re.MULTILINE)
        struct_pattern = re.compile(r'struct\s+(\w+)', re.MULTILINE)
        use_pattern = re.compile(r'use\s+([^;]+);', re.MULTILINE)

        functions = []
        classes = []  # Structs in Rust
        imports = []

        # Find functions
        for match in func_pattern.finditer(content):
            func_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            functions.append(Function(
                name=func_name,
                start_line=line_num,
                end_line=line_num,
                parameters=[]
            ))

        # Find structs (similar to classes)
        for match in struct_pattern.finditer(content):
            struct_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            classes.append(Class(
                name=struct_name,
                start_line=line_num,
                end_line=line_num,
                methods=[],
                attributes=[]
            ))

        # Find use statements
        for match in use_pattern.finditer(content):
            import_name = match.group(1).strip()
            imports.append(import_name)

        lines_of_code = len([line for line in content.splitlines() if line.strip()])
        complexity_score = self._calculate_complexity(content)

        return CodeStructure(
            language=LanguageType.RUST,
            functions=functions,
            classes=classes,
            imports=imports,
            variables=[],
            complexity_score=complexity_score,
            lines_of_code=lines_of_code,
            docstring_coverage=0.0
        )

    def _analyze_go(self, content: str) -> CodeStructure:
        """Analyze Go code using regex patterns."""
        func_pattern = re.compile(r'func\s+(?:\([^)]*\)\s+)?(\w+)\s*\(', re.MULTILINE)
        struct_pattern = re.compile(r'type\s+(\w+)\s+struct', re.MULTILINE)
        import_pattern = re.compile(r'import\s+(?:\(\s*([^)]+)\s*\)|"([^"]+)")', re.MULTILINE | re.DOTALL)

        functions = []
        classes = []
        imports = []

        # Find functions
        for match in func_pattern.finditer(content):
            func_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            functions.append(Function(
                name=func_name,
                start_line=line_num,
                end_line=line_num,
                parameters=[]
            ))

        # Find structs
        for match in struct_pattern.finditer(content):
            struct_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            classes.append(Class(
                name=struct_name,
                start_line=line_num,
                end_line=line_num,
                methods=[],
                attributes=[]
            ))

        # Find imports
        for match in import_pattern.finditer(content):
            if match.group(1):  # Multi-line import
                import_block = match.group(1)
                for line in import_block.split('\n'):
                    line = line.strip().strip('"')
                    if line:
                        imports.append(line)
            elif match.group(2):  # Single import
                imports.append(match.group(2))

        lines_of_code = len([line for line in content.splitlines() if line.strip()])
        complexity_score = self._calculate_complexity(content)

        return CodeStructure(
            language=LanguageType.GO,
            functions=functions,
            classes=classes,
            imports=imports,
            variables=[],
            complexity_score=complexity_score,
            lines_of_code=lines_of_code,
            docstring_coverage=0.0
        )

    def _analyze_java(self, content: str) -> CodeStructure:
        """Analyze Java code using regex patterns."""
        method_pattern = re.compile(r'(?:public|private|protected)?\s*(?:static)?\s*(?:\w+\s+)?(\w+)\s*\([^)]*\)\s*{', re.MULTILINE)
        class_pattern = re.compile(r'(?:public|private)?\s*class\s+(\w+)(?:\s+extends\s+(\w+))?', re.MULTILINE)
        import_pattern = re.compile(r'import\s+([^;]+);', re.MULTILINE)

        functions = []
        classes = []
        imports = []

        # Find methods (simplified - doesn't distinguish from classes well)
        for match in method_pattern.finditer(content):
            method_name = match.group(1)
            line_num = content[:match.start()].count('\n') + 1
            functions.append(Function(
                name=method_name,
                start_line=line_num,
                end_line=line_num,
                parameters=[]
            ))

        # Find classes
        for match in class_pattern.finditer(content):
            class_name = match.group(1)
            base_class = match.group(2)
            line_num = content[:match.start()].count('\n') + 1
            classes.append(Class(
                name=class_name,
                start_line=line_num,
                end_line=line_num,
                methods=[],
                attributes=[],
                base_classes=[base_class] if base_class else []
            ))

        # Find imports
        for match in import_pattern.finditer(content):
            imports.append(match.group(1).strip())

        lines_of_code = len([line for line in content.splitlines() if line.strip()])
        complexity_score = self._calculate_complexity(content)

        return CodeStructure(
            language=LanguageType.JAVA,
            functions=functions,
            classes=classes,
            imports=imports,
            variables=[],
            complexity_score=complexity_score,
            lines_of_code=lines_of_code,
            docstring_coverage=0.0
        )

    def _analyze_generic(self, content: str, language: LanguageType) -> CodeStructure:
        """Generic analysis for unsupported languages."""
        lines_of_code = len([line for line in content.splitlines() if line.strip()])
        complexity_score = self._calculate_complexity(content)

        return CodeStructure(
            language=language,
            functions=[],
            classes=[],
            imports=[],
            variables=[],
            complexity_score=complexity_score,
            lines_of_code=lines_of_code,
            docstring_coverage=0.0
        )

    def _calculate_complexity(self, content: str) -> float:
        """Calculate basic complexity score based on control flow keywords."""
        complexity_keywords = [
            'if', 'else', 'elif', 'for', 'while', 'switch', 'case',
            'try', 'catch', 'except', 'finally', 'match', 'when'
        ]

        lines = content.lower().split('\n')
        complexity = 1  # Base complexity

        for line in lines:
            for keyword in complexity_keywords:
                if keyword in line:
                    complexity += 1

        # Normalize by lines of code
        loc = len([line for line in content.splitlines() if line.strip()])
        return complexity / max(loc, 1) * 100

    def _calculate_docstring_coverage(self, functions: List[Function], classes: List[Class]) -> float:
        """Calculate percentage of functions/classes with docstrings."""
        total_items = len(functions) + len(classes)
        if total_items == 0:
            return 0.0

        documented_items = sum(1 for f in functions if f.docstring) + \
                          sum(1 for c in classes if c.docstring)

        return documented_items / total_items * 100


class LanguageEngine:
    """Main interface for language detection and code analysis."""

    def __init__(self):
        self.detector = LanguageDetector()
        self.analyzer = CodeAnalyzer()

    def analyze_file(self, file_path: str) -> Tuple[LanguageInfo, CodeStructure]:
        """Analyze a file and return language info and code structure."""
        language_info = self.detector.detect_from_file(file_path)

        try:
            content = Path(file_path).read_text(encoding='utf-8', errors='ignore')
            code_structure = self.analyzer.analyze_code(content, language_info.language)
            return language_info, code_structure
        except Exception as e:
            logger.error(f"Error analyzing file {file_path}: {e}")
            return language_info, CodeStructure(
                language=language_info.language,
                functions=[],
                classes=[],
                imports=[],
                variables=[],
                complexity_score=0.0,
                lines_of_code=0,
                docstring_coverage=0.0
            )

    def analyze_content(self, content: str, file_path: Optional[str] = None) -> Tuple[LanguageInfo, CodeStructure]:
        """Analyze content string and return language info and code structure."""
        if file_path:
            # Use file extension hint if available
            extension_lang = self.detector.EXTENSION_MAP.get(Path(file_path).suffix.lower())
            if extension_lang:
                language_info = LanguageInfo(language=extension_lang, confidence=0.8)
            else:
                language_info = self.detector.detect_from_content(content)
        else:
            language_info = self.detector.detect_from_content(content)

        code_structure = self.analyzer.analyze_code(content, language_info.language)
        return language_info, code_structure

    def get_supported_languages(self) -> List[LanguageType]:
        """Get list of supported programming languages."""
        return list(LanguageType)

    def get_language_extensions(self, language: LanguageType) -> List[str]:
        """Get file extensions associated with a language."""
        return [ext for ext, lang in self.detector.EXTENSION_MAP.items() if lang == language]