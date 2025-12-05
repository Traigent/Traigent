#!/bin/bash
# Launch script for TraiGent Playground

echo "🎯 Launching TraiGent Playground..."
echo "=================================="

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit plotly pandas
fi

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set"
    echo "   You can set it in the app's Settings tab"
    echo ""
fi

# Launch the app
echo "🚀 Starting Streamlit server..."
echo "   Opening in browser: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run examples/traigent_control_center.py
