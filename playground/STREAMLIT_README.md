# Traigent Playground - Streamlit UI

A comprehensive web interface for managing LangChain optimization problems and running Traigent optimizations with visual feedback.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# From the TraigentSDK root directory (use the existing venv if present)
python -m venv .venv
source .venv/bin/activate
pip install -e .[playground]  # or: pip install -r playground/requirements_streamlit.txt
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY='your-api-key-here'
```

### 3. Launch the Application

```bash
PYTHONPATH=. streamlit run playground/traigent_control_center.py
```

The app will open in your browser at `http://localhost:8501`

## 📱 Features

### Problem Manager Tab

- **Create New Problems**: Generate optimization problems from natural language descriptions
- **Smart Problem Analysis**: Automatic classification into 9 standardized problem types
- **Constrained Generation**: Type-specific example generation with realistic content
- **View & Edit Problems**: Browse existing problems and their examples
- **Analyze Problems**: Get quality metrics and improvement suggestions

#### 🧠 New Smart Features:

- **Intelligent Classification**: Automatically detects problem type from description
  - "text to sql" → Code Generation with SQL examples
  - "how to lose weight" → Generation with advice/guide examples
  - "classify customer emails" → Classification with category examples
- **Context-Aware Examples**: Generated examples match your specific use case
- **Confidence Scoring**: See how confident the system is about its analysis

### Run Optimizations Tab

- **Configure Runs**: Select problems, models, and optimization strategies
- **Multiple Modes**: Normal, dry run, and mock modes for testing
- **Real-time Progress**: Visual feedback during optimization

### Results Dashboard Tab

- **Historical Results**: View all past optimization runs
- **Performance Metrics**: Accuracy, cost, and timing analysis
- **Visualizations**: Charts showing trends and trade-offs

### Settings & Help Tab

- **API Configuration**: Manage OpenAI API keys
- **System Settings**: Configure default parameters
- **Documentation**: Built-in guides and tips

## 🎯 Usage Examples

### Creating a New Problem (Traditional)

1. Go to Problem Manager → Create New Problem
2. Enter description: "Classify customer emails by urgency and department"
3. Select domain: "customer_service"
4. Set difficulty: "Advanced"
5. Click "Create Problem"

### Creating a Smart Problem (AI-Powered)

1. Go to Problem Manager → Create New Problem
2. Enter description in natural language:
   - "I need a text to SQL converter for e-commerce queries"
   - "Help users understand how photosynthesis works"
   - "Extract company names and funding amounts from news articles"
3. Click "Smart Analysis" ✨
4. Review AI analysis:
   - Problem Type: Automatically detected
   - Domain: Intelligently inferred
   - Confidence: Shows analysis certainty
5. Click "Create Problem" - examples are generated automatically!

### Running an Optimization

1. Go to Run Optimizations
2. Select your problem from the dropdown
3. Choose models to test (e.g., gpt-3.5-turbo, gpt-4o-mini)
4. Select strategy: "adaptive_batch"
5. Click "Start Optimization"

### Analyzing Results

1. Go to Results Dashboard
2. View performance metrics across runs
3. Compare accuracy vs. cost trade-offs
4. Export optimal configurations

## 🔧 Advanced Configuration

### Problem Types (Automatically Detected)

The system intelligently classifies problems into 9 types:

1. **Classification**: Categorizing inputs (sentiment, intent, topic)
2. **Generation**: Creating content (guides, advice, stories)
3. **Code Generation**: Writing code (SQL, Python, JavaScript)
4. **Question Answering**: Q&A systems (FAQ bots, support)
5. **Information Extraction**: Entity extraction (NER, slot filling)
6. **Summarization**: Condensing text (abstracts, briefs)
7. **Ranking/Retrieval**: Search and recommendations
8. **Translation/Transformation**: Style transfer, format conversion
9. **Reasoning**: Logical and mathematical problem solving

### Custom Metrics

When creating problems, you can add custom evaluation metrics:

- coherence
- relevance
- creativity
- domain_accuracy
- For code generation: exact_match, execution_match, syntax_validity

### Optimization Strategies

- **grid**: Exhaustive search through all combinations
- **random**: Random sampling of parameter space
- **adaptive_batch**: Smart batching with early stopping
- **parallel_batch**: Concurrent evaluation for speed

## 📊 Mock Mode

For testing and demonstrations, use Mock Mode to:

- Test the UI without consuming API credits
- Generate sample results
- Understand the workflow

## 🐛 Troubleshooting

### "No module named 'streamlit'"

```bash
pip install streamlit
```

### "API key not found"

Set your OpenAI API key in the Settings tab or via environment variable.

### "No problems found"

Create a problem first using the Problem Manager tab.

### "Generic examples generated"

This was a known issue that has been fixed. The system now generates context-specific examples for all problem types.

## 🧪 Testing

The Traigent Playground includes comprehensive testing:

```bash
# Run tests for the Playground components
python scripts/test/run_tests.py

# Check test coverage (currently 93.8%)
python scripts/test/check_coverage.py
```

### Test Coverage Highlights:

- Script validation and syntax checking
- Mock subprocess execution
- Environment setup verification
- Error handling scenarios

## 🤝 Contributing

To add new features:

1. Edit `traigent_control_center.py`
2. Add new tabs or sections as needed
3. Update tests in `scripts/test/`
4. Ensure >85% test coverage
5. Update this README

## 📝 License

Part of the Traigent SDK - MIT License
