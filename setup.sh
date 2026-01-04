#!/bin/bash
set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}   LLM Twin: Portfolio Edition - Setup Script   ${NC}"
echo -e "${GREEN}================================================================${NC}"

# 1. Check Pre-requisites
echo -e "\n${YELLOW}[1/4] Checking prerequisites...${NC}"
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add poetry to path for this session
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Setup Environment
echo -e "\n${YELLOW}[2/4] Setting up environment...${NC}"
if [ ! -f .env ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo "⚠️  Please update .env with your API keys later! Using defaults for now."
    # Use mock LLM by default for easy start
    echo "MOCK_LLM=true" >> .env
fi

# PATCH: Update ports to avoid conflicts (27017->27018, 6333->6335)
if grep -q ":27017" .env; then
    echo "🔧 Patching .env ports to avoid conflicts..."
    sed -i 's/:27017/:27018/g' .env
    if ! grep -q "QDRANT_DATABASE_PORT" .env; then
        echo "QDRANT_DATABASE_PORT=6335" >> .env
    fi
fi

# 3. Configure Python Version
echo -e "\n${YELLOW}[3/4] Checking Python version...${NC}"
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11 is required but not found."
    echo "👉 Please run the following commands to install it:"
    echo "   sudo add-apt-repository ppa:deadsnakes/ppa"
    echo "   sudo apt update"
    echo "   sudo apt install -y python3.11 python3.11-venv python3.11-dev"
    echo "Then run 'make setup' again."
    exit 1
else
    echo "✅ Python 3.11 found. Configuring Poetry to use it..."
    poetry env use python3.11
fi

# 3.5 Check for Chrome (Required for Selenium)
echo -e "\n${YELLOW}[3.5/4] Checking for Chrome...${NC}"
if ! command -v google-chrome &> /dev/null && ! command -v chromium-browser &> /dev/null; then
    echo "❌ Chrome/Chromium is not installed (required for ETL)."
    echo "👉 Please run the following commands to install Google Chrome:"
    echo "   wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb"
    echo "   sudo apt install -y ./google-chrome-stable_current_amd64.deb"
    echo "   rm google-chrome-stable_current_amd64.deb"
    echo "Then run 'make setup' again."
    exit 1
else
    echo "✅ Chrome/Chromium found."
fi

# 4. Install Dependencies
echo -e "\n${YELLOW}[4/4] Installing dependencies...${NC}"
poetry install --no-root

# 5. Start Infrastructure (Modified from 4/4)
echo -e "\n${YELLOW}[5/5] Starting local infrastructure (MongoDB + Qdrant)...${NC}"
timeout=10 # Wait up to 10 seconds for user to cancel if needed

# Deep Clean ZenML State to prevent version mismatch (0.93.0 vs 0.74.0)
echo -e "\n${YELLOW}[+] Cleaning ZenML state...${NC}"
rm -rf .zen
rm -rf ~/.config/zenml
find . -maxdepth 2 -name "*.db" -type f -delete
poetry run zenml init

# Use 'poetry run poe' instead of 'poetry poe' to ensure it uses the installed dependency
poetry run poe local-infrastructure-up

echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}   ✅ Setup Completed Successfully!   ${NC}"
echo -e "${GREEN}================================================================${NC}"
echo -e "You can now run the RAG inference:"
echo -e "   ${YELLOW}poetry poe call-inference-ml-service${NC}"
echo -e "\nTo stop infrastructure run:"
echo -e "   ${YELLOW}poetry poe local-infrastructure-down${NC}"
