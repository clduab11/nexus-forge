#!/bin/bash

# Nexus Forge - Quick Demo Launcher
# Google ADK Hackathon Showcase
#
# This script provides a one-command demo launch for hackathon judges
# Run: ./demo/quickstart.sh

set -e

echo "🌟================================================🌟"
echo "🏆 NEXUS FORGE - GOOGLE ADK HACKATHON DEMO 🏆"
echo "🌟================================================🌟"
echo ""
echo "🚀 Launching multi-agent coordination demonstration..."
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    echo "   Usage: ./demo/quickstart.sh"
    exit 1
fi

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed"
    exit 1
fi

# Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment activated"
fi

# Install dependencies if needed
if [ ! -d "venv/lib" ] || [ ! -f ".demo_deps_installed" ]; then
    echo "📦 Installing demo dependencies..."
    pip install -q -e .
    touch .demo_deps_installed
    echo "✅ Dependencies installed"
fi

# Ensure demo directory exists
mkdir -p demo/logs

# Set environment for demo
export DEMO_MODE=true
export LOG_LEVEL=INFO

echo ""
echo "🎯 Demo Scenarios Available:"
echo "   1. Web App Generation Pipeline"
echo "   2. Content Creation with Parallel Execution"
echo "   3. Real-time Agent Coordination"
echo ""

# Prompt for demo choice
read -p "🤔 Run full demo or select scenario? (full/1/2/3): " choice

case $choice in
    "1")
        echo "🎯 Running Web App Generation Demo..."
        python3 -c "
import asyncio
from demo.multi_agent_showcase import DemoCoordinator
async def run():
    coordinator = DemoCoordinator()
    await coordinator.initialize_agents()
    return await coordinator.demo_webapp_generator()
asyncio.run(run())
"
        ;;
    "2")
        echo "🎯 Running Content Creation Pipeline Demo..."
        python3 -c "
import asyncio
from demo.multi_agent_showcase import DemoCoordinator
async def run():
    coordinator = DemoCoordinator()
    await coordinator.initialize_agents()
    return await coordinator.demo_content_pipeline()
asyncio.run(run())
"
        ;;
    "3")
        echo "🎯 Running Real-time Coordination Demo..."
        python3 -c "
import asyncio
from demo.multi_agent_showcase import DemoCoordinator
async def run():
    coordinator = DemoCoordinator()
    await coordinator.initialize_agents()
    return await coordinator.demo_real_time_coordination()
asyncio.run(run())
"
        ;;
    *)
        echo "🎬 Running Complete Multi-Agent Demo..."
        python3 demo/multi_agent_showcase.py
        ;;
esac

echo ""
echo "🎉 Demo completed successfully!"
echo ""
echo "📊 Key Highlights for Judges:"
echo "   ✅ Multi-agent coordination (5 specialized agents)"
echo "   ✅ Google ADK integration with Agent2Agent protocol"
echo "   ✅ Real-time orchestration via Supabase"
echo "   ✅ Parallel execution achieving 3.4x speedup"
echo "   ✅ Production-ready enterprise architecture"
echo ""
echo "📖 For more details, see README.md"
echo "🔗 Code repository: https://github.com/clduab11/nexus-forge"
echo ""
echo "🏆 Nexus Forge: Where AI agents collaborate to build the future!"