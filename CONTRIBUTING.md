# Contributing to Cultural Heritage AI Platform

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the platform.

## 🤝 How to Contribute

### Reporting Bugs

1. **Check existing issues** to see if the bug has already been reported
2. **Create a new issue** with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs. actual behavior
   - Environment details (OS, Python version, GPU)
   - Error messages or logs

### Suggesting Enhancements

1. **Open a feature request** issue
2. Describe the enhancement clearly
3. Explain the use case and benefits
4. Consider implementation approach (optional)

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes**
4. **Test thoroughly**
5. **Commit with clear messages**
6. **Push to your fork**
7. **Open a Pull Request**

## 📝 Code Style Guidelines

### Python Style

- Follow **PEP 8** style guide
- Use **Black** for code formatting: `black .`
- Maximum line length: 100 characters
- Use type hints where appropriate

### Documentation

- Add docstrings to all functions and classes
- Use Google-style docstrings
- Update README.md if adding new features
- Include examples in docstrings

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add new artist style to image generation
fix: Resolve CUDA memory leak in restoration module
docs: Update architecture documentation
refactor: Optimize FAISS search performance
test: Add unit tests for authentication module
```

## 🧪 Testing

### Before Submitting

1. **Run existing tests**: `pytest tests/`
2. **Test your changes** in the relevant module
3. **Check for linting errors**: `flake8 .`
4. **Format code**: `black .`

### Writing Tests

- Add tests for new functionality
- Test edge cases and error handling
- Aim for good test coverage
- Use descriptive test names

## 📦 Module-Specific Guidelines

### Art Authentication Module

- Test with various image formats
- Ensure model compatibility
- Document accuracy improvements

### Image Generation Module

- Test with different artist styles
- Verify prompt handling
- Check output quality

### Heritage Restoration Module

- Test with various damage types
- Verify depth map generation
- Ensure restoration quality

### 2D to 3D Module

- Test with different statue types
- Verify mesh quality
- Check GLB export

### RAG Q&A Module

- Test query understanding
- Verify retrieval accuracy
- Check answer quality

## 🔍 Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new features
3. **Ensure all tests pass**
4. **Update CHANGELOG.md** (if applicable)
5. **Request review** from maintainers

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] No merge conflicts
- [ ] Clear PR description

## 🎯 Areas for Contribution

### High Priority

- **Performance Optimization**: Improve inference speed
- **Model Improvements**: Better accuracy/results
- **Documentation**: More examples and tutorials
- **Testing**: Increase test coverage
- **Error Handling**: Better error messages

### Medium Priority

- **New Features**: Additional artist styles, restoration techniques
- **API Development**: RESTful API for modules
- **Web Interface**: Gradio/Streamlit dashboard
- **Deployment**: Docker containers, cloud deployment guides

### Low Priority

- **Code Refactoring**: Improve code organization
- **UI/UX**: Better notebook interfaces
- **Examples**: More use case examples
- **Blog Posts**: Tutorials and case studies

## 📚 Development Setup

### Setting Up Development Environment

```bash
# Clone your fork
git clone https://github.com/your-username/cultural-heritage-ai-platform.git
cd cultural-heritage-ai-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists

# Install pre-commit hooks (if configured)
pre-commit install
```

### Development Workflow

1. **Sync with upstream**: `git fetch upstream && git merge upstream/main`
2. **Create branch**: `git checkout -b feature/your-feature`
3. **Make changes** and test
4. **Commit**: `git commit -m "feat: your feature"`
5. **Push**: `git push origin feature/your-feature`
6. **Open PR** on GitHub

## 🐛 Debugging Tips

### Common Issues

1. **Import errors**: Check Python path and virtual environment
2. **CUDA errors**: Verify GPU availability and CUDA version
3. **Memory issues**: Reduce batch size or image resolution
4. **Model loading**: Check Hugging Face token and internet connection

### Debugging Tools

- Use `print()` statements for debugging
- Use `pdb` for interactive debugging
- Check logs in `logs/` directory (if exists)
- Use `torch.cuda.memory_summary()` for GPU memory

## 📖 Documentation Contributions

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Keep documentation up-to-date with code

### Documentation Structure

- **README.md**: Overview and quick start
- **docs/ARCHITECTURE.md**: System architecture
- **docs/GETTING_STARTED.md**: Setup guide
- **docs/modules/**: Module-specific guides

## 🎨 Design Principles

When contributing, keep these principles in mind:

1. **Modularity**: Each module should be independent
2. **Reusability**: Share common components
3. **Performance**: Optimize for speed and memory
4. **Usability**: Make it easy to use
5. **Maintainability**: Write clean, documented code

## 📧 Questions?

- **GitHub Discussions**: For questions and discussions
- **GitHub Issues**: For bugs and feature requests
- **Email**: [Your email] for direct contact

## 🙏 Thank You!

Your contributions make this platform better for everyone. We appreciate your time and effort!

---

**Happy Contributing!** 🚀

