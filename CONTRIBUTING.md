# Contributing to Accident FIR Automation

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and beginners
- Focus on constructive feedback
- Maintain professionalism

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in Issues
2. Use the bug report template
3. Include:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information
   - Error logs/screenshots

### Suggesting Features

1. Check existing feature requests
2. Clearly describe the feature and its benefits
3. Provide use cases
4. Suggest implementation approach if possible

### Contributing Code

#### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/accident-fir-automation.git
cd accident-fir-automation

# Add upstream remote
git remote add upstream https://github.com/original/accident-fir-automation.git
```

#### 2. Create a Branch

```bash
# Update your fork
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

#### 3. Make Changes

- Follow code style guidelines (see below)
- Write clear commit messages
- Add tests for new features
- Update documentation

#### 4. Test Your Changes

```bash
# Run tests
pytest tests/ -v

# Run linters
black src/
flake8 src/
mypy src/

# Check formatting
isort src/
```

#### 5. Commit and Push

```bash
# Stage changes
git add .

# Commit with clear message
git commit -m "Add feature: description of feature"

# Push to your fork
git push origin feature/your-feature-name
```

#### 6. Create Pull Request

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Fill out the PR template
4. Link related issues
5. Wait for review

## Code Style Guidelines

### Python

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Type hints for function signatures
- Docstrings for all public functions/classes

```python
def analyze_image(
    image: np.ndarray,
    confidence_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Analyze accident image for vehicles and damage.
    
    Args:
        image: Input image in BGR format
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        Dictionary containing detection results
        
    Raises:
        ValueError: If image is invalid
    """
    pass
```

### File Organization

- One class per file (exceptions allowed)
- Group related functions together
- Use descriptive names
- Keep files under 500 lines

### Comments

- Explain WHY, not WHAT
- Keep comments up-to-date
- Use TODO for known issues
- Document complex algorithms

## Testing Guidelines

### Unit Tests

- Test one thing per test
- Use descriptive test names
- Cover edge cases
- Aim for >80% coverage

```python
def test_detector_handles_empty_image():
    """Test that detector handles empty images gracefully"""
    detector = YOLODetector()
    empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
    
    with pytest.raises(ValueError):
        detector.detect(empty_image)
```

### Integration Tests

- Test complete workflows
- Use realistic test data
- Clean up test artifacts

## Documentation

### Updating Docs

- Update README for major features
- Add API documentation for new endpoints
- Include code examples
- Update PROJECT_GUIDE for new workflows

### Docstring Format

Use Google style:

```python
def function(arg1: str, arg2: int = 0) -> bool:
    """
    Summary line.
    
    Extended description.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2. Defaults to 0.
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When arg1 is empty
    """
    pass
```

## Review Process

1. **Automated checks**: CI runs tests and linters
2. **Code review**: Maintainer reviews code
3. **Feedback**: Address requested changes
4. **Approval**: Once approved, PR will be merged

### What Reviewers Look For

- Code quality and style
- Test coverage
- Documentation
- Performance impact
- Breaking changes
- Security implications

## Release Process

1. Update version in `setup.py` and `src/api/main.py`
2. Update CHANGELOG.md
3. Create release tag
4. Build and publish

## Getting Help

- **Questions**: Open a Discussion
- **Bugs**: Open an Issue
- **Chat**: Join our Discord/Slack
- **Email**: developers@example.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in documentation

Thank you for contributing! 🎉
