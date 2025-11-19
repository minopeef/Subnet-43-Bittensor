
# Validator Setup Guide

## Prerequisites

### Hardware Requirements

- **CPU**: 32+ cores recommended
- **RAM**: 16GB minimum, 32GB+ recommended
  - Each container needs ~250MB
  - Large problems (3000 nodes) use ~100MB per distance matrix
  - With concurrent evaluation, plan for peak usage
- **Storage**: 1TB+ available
- **Network**: Stable internet connection

### Software Requirements

- Docker and Docker Compose
- Python 3.10 or 3.11
- Bittensor wallet with registered validator hotkey

## Initial Setup

### 1. Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd sn43
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# Bittensor Wallet Configuration
BT_WALLET_COLD=your_coldkey_name
BT_WALLET_HOT=your_validator_hotkey_name

# Optional: Custom Configuration
SUBTENSOR_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
```

### 3. Register as Validator

```bash
# Register on subnet 43
btcli subnet register --netuid 43 --wallet.name your_coldkey_name --hotkey your_validator_hotkey_name

# Verify registration
btcli subnet list --netuid 43 | grep your_validator_hotkey_name
```

## Building Docker Images

### 1. Build the Validator Image

```bash
docker compose build
```

This builds the main validator service that orchestrates evaluations.

### 2. Build the Agent Container Image

```bash
docker build -f Dockerfile.agent -t sn43-agent:local .
```

This creates the base image used to run miner agents in isolated containers.

## Preparing for Launch

### Create Persistent Directories

```bash
# Create directories for persistent storage
mkdir -p best_agents agents

# Set permissions (validator needs write access)
chmod 777 best_agents agents
```

These directories store:
- `best_agents/`: Best performing agents and metadata (persists across restarts)
- `agents/`: Downloaded miner agent files from the network

## Running the Validator

### Start Validation

```bash
docker compose up -d
```

### Monitor Logs

```bash
# Follow logs in real-time
docker compose logs -f validator

# View last 100 lines
docker compose logs --tail 100 validator

# Check for errors
docker compose logs validator | grep ERROR
```

## Understanding the Evaluation Process

### Evaluation Flow

1. **Initialization**: Validator starts, loads TSP environment and tools
2. **Miner Discovery**: Pulls agent code from all registered miners
3. **Container Creation**: Spins up isolated Docker containers for each valid agent
4. **Problem Generation**: Creates TSP problems of varying sizes and types
5. **Concurrent Evaluation**: Runs agent containers against problems 
6. **Scoring**: Evaluates solution quality and tracks performance
7. **Ranking Stability**: Continues evaluating until rankings converge
8. **Reward Calculation**: Awards miners who beat historical best by threshold
9. **Weight Setting**: Commits weights to blockchain

### Problem Types & Bins

Miners are evaluated across multiple problem configurations:

```
TSP (small): 100-500 nodes
TSP (medium): 500-1000 nodes  
TSP (large): 1000-2000 nodes

+ Multiple variants (mTSP, mDmTSP, cmDmTSP, rcmDmTSP, rcmDmTSPTW)
```

### Resource Management

The validator automatically:
- Only evaluates miners with actual submissions
- Scores failed/missing miners as infinity
- Cleans up containers after each epoch


## Maintenance

### Restart Validator

```bash
docker compose restart validator
```

### Clean Up Old Containers

```bash
# Prune unused images
docker image prune -f
```

### Update Code

```bash
# Pull latest changes
git pull

# Rebuild images
docker compose build
docker build -f Dockerfile.agent -t sn43-agent:local .

# Restart
docker compose down
docker compose up -d
```

### View Best Agents

```bash
# List all best agents
ls -la best_agents/

# View metadata
cat best_agents/metadata.json

# Check a specific agent
cat best_agents/ProblemType.TSP_small_uid4_epoch1.py
```

## Security Considerations

- Miner code runs in isolated Docker containers
- Containers have limited resources (CPU/memory limits)
- No network access from agent containers
- Agents cannot access host filesystem
- Token-based authentication for tool calls

## Performance Optimization

1. **SSD Storage**: Use SSD for faster container I/O
2. **Network Bandwidth**: Ensure stable connection for pulling agents
3. **CPU**: More cores = faster concurrent evaluations
4. **RAM**: 32GB+ recommended for validation

---