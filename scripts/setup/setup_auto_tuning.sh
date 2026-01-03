#!/bin/bash
# Setup script for CI/CD auto-tuning pipeline

set -e

echo "🚀 Setting up Traigent CI/CD Auto-Tuning Pipeline"
echo "================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo -e "${GREEN}✓ Python $python_version is compatible${NC}"
else
    echo -e "${RED}✗ Python $python_version is too old. Required: $required_version or higher${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "\n${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip --quiet

# Install Traigent with dependencies
echo -e "\n${YELLOW}Installing Traigent SDK with dependencies...${NC}"
pip install -e ".[dev,integrations,bayesian]" --quiet
echo -e "${GREEN}✓ Traigent SDK installed${NC}"

# Install DVC and CML
echo -e "\n${YELLOW}Installing DVC and CML...${NC}"
pip install dvc[s3] cml dvclive --quiet
echo -e "${GREEN}✓ DVC and CML installed${NC}"

# Install pre-commit
echo -e "\n${YELLOW}Installing pre-commit hooks...${NC}"
pip install pre-commit --quiet
pre-commit install --install-hooks
echo -e "${GREEN}✓ Pre-commit hooks installed${NC}"

# Initialize DVC if needed
if [ ! -d ".dvc" ]; then
    echo -e "\n${YELLOW}Initializing DVC...${NC}"
    dvc init
    echo -e "${GREEN}✓ DVC initialized${NC}"
else
    echo -e "${GREEN}✓ DVC already initialized${NC}"
fi

# Create necessary directories
echo -e "\n${YELLOW}Creating directory structure...${NC}"
mkdir -p data/raw data/prepared
mkdir -p optimization_results optimization_logs
mkdir -p baselines reports performance_plots
mkdir -p scripts/hooks scripts/auto_tune
echo -e "${GREEN}✓ Directory structure created${NC}"

# Create initial baseline if it doesn't exist
if [ ! -f "baselines/performance.json" ]; then
    echo -e "\n${YELLOW}Creating initial performance baseline...${NC}"
    cat > baselines/performance.json << EOF
{
  "accuracy": 0.8,
  "avg_latency": 1.0,
  "total_cost": 0.0,
  "tokens_used": 0,
  "trials_completed": 0,
  "best_score": 0.8
}
EOF
    echo -e "${GREEN}✓ Initial baseline created${NC}"
else
    echo -e "${GREEN}✓ Baseline already exists${NC}"
fi

# Check for environment variables
echo -e "\n${YELLOW}Checking environment variables...${NC}"
missing_vars=()

if [ -z "$ANTHROPIC_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
    missing_vars+=("ANTHROPIC_API_KEY or OPENAI_API_KEY")
fi

if [ ${#missing_vars[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠ Missing environment variables:${NC}"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    echo -e "${YELLOW}Set them in your environment or GitHub secrets${NC}"
else
    echo -e "${GREEN}✓ Environment variables configured${NC}"
fi

# Test the setup
echo -e "\n${YELLOW}Testing setup with mock mode...${NC}"
TRAIGENT_MOCK_LLM=true python -c "
import traigent
from traigent import traigent as traigent_decorator
print('Traigent version:', traigent.__version__)
print('✓ Import successful')
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Setup test passed${NC}"
else
    echo -e "${RED}✗ Setup test failed${NC}"
    exit 1
fi

# Display next steps
echo -e "\n${GREEN}✅ Setup Complete!${NC}"
echo -e "\n${YELLOW}Next Steps:${NC}"
echo "1. Configure GitHub secrets (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)"
echo "2. Run a test optimization: ${GREEN}TRAIGENT_MOCK_LLM=true dvc repro${NC}"
echo "3. Push to staging branch to trigger auto-tuning"
echo "4. Check the documentation: ${GREEN}docs/CI_CD_AUTO_TUNING.md${NC}"

echo -e "\n${YELLOW}Quick Commands:${NC}"
echo "  Run pipeline:        ${GREEN}dvc repro${NC}"
echo "  Check status:        ${GREEN}dvc status${NC}"
echo "  View metrics:        ${GREEN}dvc metrics show${NC}"
echo "  Manual optimization: ${GREEN}python -m traigent.cli optimize --config configs/auto_tune_config.yaml${NC}"

echo -e "\n${GREEN}Happy Optimizing! 🚀${NC}"
