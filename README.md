# NLLB-200 Translation API

A high-quality FastAPI server for the NLLB-200 multilingual translation model.

## Features

- **SOLID Architecture**: Clean, maintainable code following SOLID principles
- **Type Safety**: Full type hints with Pydantic validation
- **Error Handling**: Comprehensive error handling with meaningful responses
- **Security**: Optional API key authentication
- **Logging**: Structured logging with context
- **Testing**: Unit tests with pytest
- **Scalability**: Async support for high performance

## Quick Start

### Installation

```bash
# Clone and setup
git clone <repository>
cd nllb-200-api
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env
```

### Running

```bash
# Start server
python -m src.main

# Or with uvicorn directly
uvicorn src.main:app --reload
```

## API Usage

### Translate Text

```bash
curl -X POST "http://localhost:8000/api/v1/translate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "source_language": "eng_Latn",
    "target_language": "fra_Latn"
  }'
```

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### Get Supported Languages

```bash
curl http://localhost:8000/api/v1/languages
```

## Development

### Running Tests

```bash
pytest
```

### Type Checking

```bash
mypy src/
```

### Code Formatting

```bash
black src/
isort src/
```

## Architecture

The project follows clean architecture principles:

- **src/**: Source code
  - **config/**: Configuration management
  - **models/**: Pydantic schemas
  - **services/**: Business logic (translation service)
  - **api/**: API routes and middleware
- **tests/**: Unit tests
- **requirements/**: Dependency management

## Supported Languages

The NLLB-200 model supports 200 languages. Check the `/languages` endpoint for the complete list.

## License

MIT License - see LICENSE file for details.

All facebook/NLLB-200 Models on License CC-BY-NC 
