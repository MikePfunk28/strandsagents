# Development Guidelines & Standards

## Code Quality Standards

### Documentation Patterns
- **Comprehensive Module Docstrings**: Every module starts with detailed docstrings explaining purpose, features, and usage
- **Class and Function Documentation**: All classes and functions include detailed docstrings with Args, Returns, and Examples
- **Inline Comments**: Complex logic includes explanatory comments for maintainability
- **Type Hints**: Extensive use of type hints for parameters, return values, and class attributes

### Error Handling & Logging
- **Structured Logging**: Consistent use of Python logging module with appropriate log levels
- **Graceful Error Handling**: Try-catch blocks with meaningful error messages and fallback behavior
- **Error Context**: Error messages include context about what operation failed and why
- **Logging Configuration**: Centralized logging setup with configurable levels and formats

### Code Structure & Organization
- **Dataclass Usage**: Extensive use of `@dataclass` for structured data with automatic `__init__` and validation
- **Factory Pattern**: Factory functions for creating complex objects (e.g., `create_graph_storage()`)
- **Async/Await**: Proper async programming patterns for I/O operations and concurrent processing
- **Path Handling**: Consistent use of `pathlib.Path` for cross-platform file operations

## Architectural Patterns

### Configuration Management
- **Environment Variables**: Configuration through environment variables with sensible defaults
- **Settings Classes**: Centralized configuration classes with validation
- **Feature Flags**: Boolean flags for enabling/disabling features (e.g., `ALLOW_SHELL=False`)
- **Default Values**: Always provide default values for optional configuration

### Data Storage Patterns
- **Multiple Backend Support**: Abstract storage interfaces with multiple implementations (Parquet, JSON)
- **Serialization Standards**: Consistent JSON serialization with proper datetime handling
- **File History Tracking**: Audit trails for all file operations with metadata
- **Backup Strategies**: Automatic backups before risky operations

### Agent Architecture
- **Base Classes**: Abstract base classes defining common interfaces for agents
- **Decorator Pattern**: Use of decorators for agent creation and configuration
- **Tool Integration**: Standardized tool loading and execution patterns
- **Model Selection**: Intelligent model selection based on task requirements

## Development Standards

### Import Organization
```python
# Standard library imports first
import os
import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Third-party imports
import numpy as np
import pandas as pd
from strands import Agent, tool

# Local imports last
from .base_assistant import BaseAssistant
from ..communication.mcp_client import SwarmMCPClient
```

### Class Design Patterns
- **Dataclass for Data Structures**: Use `@dataclass` for data containers with automatic methods
- **Abstract Base Classes**: Define interfaces using `ABC` and `@abstractmethod`
- **Factory Methods**: Provide factory functions for complex object creation
- **Builder Pattern**: Use builder classes for complex configuration scenarios

### Error Handling Standards
```python
try:
    # Operation that might fail
    result = risky_operation()
    logger.info(f"Operation completed successfully: {result}")
    return result
except SpecificException as e:
    logger.error(f"Specific error in operation: {e}")
    return fallback_value
except Exception as e:
    logger.error(f"Unexpected error in operation: {e}")
    raise
```

### Async Programming Patterns
- **Async Context Managers**: Use `async with` for resource management
- **Task Creation**: Use `asyncio.create_task()` for concurrent operations
- **Timeout Handling**: Always include timeouts for network operations
- **Queue Processing**: Use `asyncio.Queue` for producer-consumer patterns

## Tool Integration Standards

### Strands Tools Usage
- **Tool Decoration**: Use `@tool` decorator for creating reusable tools
- **Parameter Validation**: Validate tool parameters with proper type hints
- **Error Propagation**: Handle tool errors gracefully with meaningful messages
- **Tool Composition**: Combine multiple tools for complex operations

### Model Integration
- **Provider Abstraction**: Support multiple model providers (Ollama, Bedrock, OpenAI)
- **Model Selection**: Intelligent model selection based on task requirements
- **Timeout Configuration**: Configurable timeouts for model operations
- **Fallback Strategies**: Graceful degradation when preferred models unavailable

## Testing & Validation

### Test Structure
- **Comprehensive Test Suites**: Multiple test classes covering different aspects
- **Setup and Teardown**: Proper test setup with cleanup
- **Mock Usage**: Mock external dependencies for isolated testing
- **Integration Tests**: End-to-end testing of complete workflows

### Validation Patterns
- **Input Validation**: Validate all inputs at system boundaries
- **Type Checking**: Use type hints and runtime type checking
- **Schema Validation**: Validate data structures against expected schemas
- **Business Logic Validation**: Validate business rules and constraints

## Performance Optimization

### Caching Strategies
- **Result Caching**: Cache expensive computation results
- **LRU Cache**: Use LRU eviction for memory management
- **Cache Invalidation**: Proper cache invalidation strategies
- **Cache Size Limits**: Configurable cache size limits

### Memory Management
- **Resource Cleanup**: Proper cleanup of resources and connections
- **Memory Monitoring**: Track memory usage in long-running processes
- **Batch Processing**: Process large datasets in batches
- **Lazy Loading**: Load data only when needed

## Security & Safety

### Input Sanitization
- **Parameter Validation**: Validate all input parameters
- **Path Traversal Prevention**: Secure file path handling
- **Command Injection Prevention**: Sanitize shell commands
- **Access Control**: Implement proper access controls

### Safe Operations
- **Backup Before Modification**: Create backups before risky operations
- **Risk Assessment**: Assess and score operation risks
- **Confirmation Prompts**: Require confirmation for destructive operations
- **Audit Logging**: Log all security-relevant operations

## Code Style Conventions

### Naming Conventions
- **snake_case**: Functions, variables, and module names
- **PascalCase**: Class names and type definitions
- **UPPER_CASE**: Constants and configuration values
- **Descriptive Names**: Use clear, descriptive names for all identifiers

### File Organization
- **Module Structure**: Organize code into logical modules
- **Import Order**: Standard library, third-party, local imports
- **File Naming**: Use descriptive, consistent file names
- **Directory Structure**: Organize files into logical directory hierarchies

### Documentation Standards
- **Docstring Format**: Use consistent docstring format with Args, Returns, Examples
- **README Files**: Comprehensive README files for each major component
- **API Documentation**: Document all public APIs with examples
- **Architecture Documentation**: High-level architecture documentation

## Integration Patterns

### MCP Communication
- **Message Handling**: Structured message handling with type safety
- **Connection Management**: Proper connection lifecycle management
- **Error Recovery**: Automatic reconnection and error recovery
- **Protocol Compliance**: Strict adherence to MCP protocol standards

### Database Integration
- **Connection Pooling**: Use connection pooling for database operations
- **Transaction Management**: Proper transaction handling with rollback
- **Migration Support**: Database schema migration support
- **Query Optimization**: Optimize database queries for performance

### External Service Integration
- **Retry Logic**: Implement retry logic with exponential backoff
- **Circuit Breaker**: Use circuit breaker pattern for external services
- **Health Checks**: Implement health checks for external dependencies
- **Service Discovery**: Support for service discovery mechanisms