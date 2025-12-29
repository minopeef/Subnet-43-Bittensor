# SN43 - Graphite Subnet

A Bittensor subnet focused on solving Traveling Salesman Problem (TSP) variants using distributed agents. Miners submit solving algorithms that are evaluated by validators on various TSP problem types and sizes.

## Overview

Subnet 43 is a decentralized network where miners compete to solve TSP problems efficiently. Validators evaluate miner solutions across multiple problem types and sizes, rewarding the best performers based on solution quality and consistency.

## Project Structure

- `sn43/` - Main package containing the core protocol and solver implementations
- `neurons/` - Neuron implementations for miners and validators
- `envs/` - Environment configurations including TSP agent templates
- `docs/` - Documentation for miners and validators
- `Dockerfile` - Validator container image
- `Dockerfile.agent` - Base image for running miner agents in isolation
- `docker-compose.yml` - Validator service orchestration

## Requirements

- Python 3.11
- Docker and Docker Compose
- Bittensor wallet with registered hotkey on subnet 43
- GitHub Personal Access Token (for miners, with gist scope)

## Installation

1. Clone the repository
2. Install dependencies using uv or pip
3. Configure environment variables in `.env` file
4. Register your wallet on subnet 43

## Configuration

Create a `.env` file in the project root with the following variables:

```
BT_WALLET_COLD=your_coldkey_name
BT_WALLET_HOT=your_hotkey_name
GITHUB_TOKEN=your_github_token
SUBTENSOR_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
```

## Usage

### For Miners

1. Develop your TSP solving agent implementing the `solve_problem` function
2. Push your agent to the network using the neurons CLI
3. Monitor your performance and rankings
4. Update your agent as needed

See `docs/miner_readme.md` for detailed miner setup instructions.

### For Validators

1. Build Docker images for validator and agent containers
2. Start the validator service using docker-compose
3. Monitor evaluation logs and performance metrics
4. Maintain the validator to ensure consistent evaluation

See `docs/validator_readme.md` for detailed validator setup instructions.

## Problem Types

The subnet evaluates miners on various TSP problem variants:

- Metric TSP: Standard traveling salesman problem
- mTSP: Multiple salesmen
- mDmTSP: Multiple depots, multiple salesmen
- cmDmTSP: Constrained version with capacity limits
- rcmDmTSP: Randomized constraints
- rcmDmTSPTW: With time windows

Problems are evaluated across different size bins:
- Small: 100-500 nodes
- Medium: 500-1000 nodes
- Large: 1000-2000 nodes

## Solution Requirements

Miner agents must:

- Return solutions within the timeout period (approximately 300 seconds)
- Provide valid tours that start and end at node 0
- Handle all problem types gracefully
- Beat historical best solutions by a threshold to receive rewards

## Rewards

Miners are rewarded based on:

- Solution quality: Lower tour distance results in higher scores
- Improvement over baseline: Must exceed historical best by threshold
- Consistency: Stable performance across multiple evaluations

Only the best miner per problem type and size bin receives rewards each epoch.

## Security

- Miner code runs in isolated Docker containers
- Containers have limited CPU and memory resources
- No network access from agent containers
- Agents cannot access host filesystem
- Token-based authentication for tool calls

## Development

The project uses:

- Bittensor for decentralized network infrastructure
- Docker for containerized agent execution
- NetworkX for graph operations
- PySCIPOpt and Highs for optimization solvers
- FastAPI for API services

## Commands

- `sn43` - Main CLI entry point
- `neurons` - Neuron management CLI
- `python run_validator.py` - Run validator directly

## Troubleshooting

### Missing Environment Variables

Ensure your `.env` file contains all required variables.

### Docker Issues

Verify Docker and Docker Compose are installed and running. Check container logs for errors.

### Registration Issues

Confirm your wallet is registered on subnet 43 using the bittensor CLI.

### Agent Submission Failures

Verify your GitHub token is valid and has the gist scope enabled.

## License

See LICENSE file for details.

