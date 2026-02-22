# TTT-Discover Agent for MLE-Bench

A machine learning agent implementing the TTT-Discover algorithm (arXiv:2601.16175) adapted for MLE-Bench evaluation.

## Overview

This agent implements the core TTT-Discover algorithm from the paper "Learning to Discover at Test Time" but uses **prompt-based evolution with a frozen LLM** instead of the paper's RL weight-update approach. The agent maintains a buffer of ML pipeline variants, uses PUCT-based seed selection, and iteratively improves solutions through LLM mutations.

## Features

- **TTT-Discover Algorithm**: Implements buffer-based solution discovery with PUCT selection
- **Prompt-Based Evolution**: Uses OpenAI API (GPT-4o) for code generation instead of RL fine-tuning
- **Cross-Validation Scoring**: Each pipeline variant performs K-fold CV and reports `CV_SCORE`
- **Time Budget Management**: Respects competition time limits with intelligent iteration control
- **Robust Error Handling**: Fallback mechanisms for failed generations and submissions

## Quick Start

### Prerequisites

- Docker
- OpenAI API Key
- MLE-Bench environment

### Installation

1. Clone this repository:
```bash
git clone https://github.com/aadil-sengupta/ttt-discovery-mle-agent.git
cd ttt-discovery-mle-agent
```

2. Build the Docker image:
```bash
export SUBMISSION_DIR=/home/submission
export LOGS_DIR=/home/logs
export CODE_DIR=/home/code
export AGENT_DIR=/home/agent

docker build --platform=linux/amd64 -t tttdiscovery . \
  --build-arg SUBMISSION_DIR=$SUBMISSION_DIR \
  --build-arg LOGS_DIR=$LOGS_DIR \
  --build-arg CODE_DIR=$CODE_DIR \
  --build-arg AGENT_DIR=$AGENT_DIR
```

### Usage with MLE-Bench

```bash
# Run on Spaceship Titanic competition
python run_agent.py --agent-id tttdiscovery/dev --competition-set experiments/splits/spaceship-titanic.txt

# Run on full Lite evaluation set
python run_agent.py --agent-id tttdiscovery --competition-set experiments/splits/low.txt
```

## Agent Variants

- `tttdiscovery`: Full version (50 iterations, 600s timeout)
- `tttdiscovery/dev`: Development version (5 iterations, 300s timeout) for testing
- `tttdiscovery/gpt-4o-mini`: GPT-4o-mini model variant

## Performance

On the Spaceship Titanic competition:
- **Score**: 0.80115
- **Above Median**: ✅ Yes (median threshold: 0.79565)
- **Close to Bronze**: Only 0.00852 points from Bronze medal

## Architecture

### Core Components

1. **`buffer.py`**: Solution buffer storing pipeline variants with scores and lineage
2. **`sampler.py`**: PUCT-based seed selection for intelligent exploration
3. **`generator.py`**: OpenAI API integration for pipeline generation and mutation
4. **`runner.py`**: Pipeline execution with timeout and validation
5. **`main.py`**: Orchestrator implementing the full TTT-Discover loop

### Algorithm Flow

1. **Seed Phase**: Generate initial pipeline variants from scratch
2. **Mutation Loop**: While time remains:
   - Select seed via PUCT scoring
   - Generate mutation via LLM
   - Execute and score variant
   - Add to buffer with lineage
3. **Finalization**: Re-run best variant for final submission

## Configuration

See `config.yaml` for agent variants and hyperparameters:
- `max-iterations`: Maximum number of mutation iterations
- `initial-variants`: Number of initial pipeline seeds
- `puct-c`: PUCT exploration coefficient
- `per-variant-timeout`: Timeout for each pipeline execution

## Citation

If you use this agent in your research, please cite the TTT-Discover paper:

```bibtex
@article{tttdiscover2026,
  title={Learning to Discover at Test Time},
  author={Anonymous},
  journal={arXiv preprint arXiv:2601.16175},
  year={2026}
}
```

## Repository

**GitHub**: https://github.com/aadil-sengupta/ttt-discovery-mle-agent

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.