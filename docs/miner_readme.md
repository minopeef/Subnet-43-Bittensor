# Miner Setup Guide

## Prerequisites

- Python 3.10 or 3.11
- Bittensor wallet with registered hotkey on subnet 43
- GitHub account with Personal Access Token

## Initial Setup

### 1. Install Dependencies

```bash
# Clone the repository
git clone https://github.com/GraphiteAI/sn43.git
cd sn43
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
# Bittensor Wallet Configuration
BT_WALLET_COLD=your_coldkey_name
BT_WALLET_HOT=your_hotkey_name

# GitHub Token (with 'gist' scope)
# Create at: https://github.com/settings/tokens/new
GITHUB_TOKEN=ghp_your_github_personal_access_token

# Optional: Custom Configuration
SUBTENSOR_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
```

### 3. Register Your Wallet

If not already registered:

```bash
btcli subnet register --netuid 43 --wallet.name your_coldkey_name --hotkey your_hotkey_name
```

## Developing Your Agent

### Agent Requirements

Your agent must implement a `solve_problem` function:

```python
def solve_problem(problem_data: dict) -> dict:
    """
    Solve the TSP problem.
    
    Args:
        problem_data: Dictionary containing:
            - 'problem_type': str (e.g., "Metric TSP")
            - 'n_nodes': int
            - 'distance_matrix': List[List[float]]
            - Additional problem-specific fields
            
    Returns:
        Dictionary with 'solution' key containing a valid tour:
        {
            'solution': [0, 1, 2, ..., n-1, 0]  # Must start and end at node 0
        }
    """
    # Your solving logic here
    solution = your_solver(problem_data)
    
    return {"solution": solution}
```

### Solution Validation

- Must return within the timeout (~300 seconds)
- Generally, lower total distance = better score, meet all constraints as close as you can

## Submitting Your Agent

### Create GitHub Token

1. Go to https://github.com/settings/tokens/new
2. Give it a descriptive name (e.g., "SN43 Miner")
3. Set expiration as needed
4. **Enable the `gist` scope** (required!)
5. Generate and copy the token to your `.env` file

### Push Your Agent

```bash
# Push your agent to the network
python3 -m neurons push path/to/your/agent.py

# Example with the provided template:
python3 -m neurons push envs/tsp/agent.py
```

This will:
1. Create a public GitHub Gist with your agent code
2. Commit the Gist URL to the blockchain
3. Make your agent available for validators to evaluate

### Verify Submission

```bash
# Check your UID
btcli subnet list --netuid 43 | grep your_hotkey_name

# Pull your own agent to verify
python3 -m neurons pull <your_uid>
```

## Updating Your Agent

Simply push again with the updated file:

```bash
python3 -m neurons push path/to/your/updated_agent.py
```

The new version will be committed to the blockchain and validators will pull the latest version.

## Tips for Success

1. **Optimize gradually**: Test improvements locally before pushing
2. **Handle all problem types**: The validator tests multiple TSP variants
3. **Stay within timeout**: Solutions must complete in ~5 minutes
4. **Monitor performance**: Track your scores and ranking over time
5. **Join the community**: Get help and share strategies on Discord

## Troubleshooting

### "Missing required environment variable"

Ensure `.env` file has all required variables (see step 2)

### "Failed to create gist"

- Verify GitHub token is valid and has `gist` scope
- Check token hasn't expired
- Ensure internet connectivity

### "Not registered"

Register your wallet first (see step 3)

## Problem Types

The subnet evaluates miners on various TSP problem types:

- **Metric TSP**: Standard traveling salesman problem
- **mTSP**: Multiple salesmen
- **mDmTSP**: Multiple depots, multiple salesmen
- **cmDmTSP**: Constrained version with capacity limits
- **rcmDmTSP**: Randomized constraints
- **rcmDmTSPTW**: With time windows

Your agent should handle all types or gracefully return the best solution it can.

## Rewards

Miners are rewarded based on:
- **Solution quality**: Lower tour distance = higher score
- **Improvement over baseline**: Must beat historical best by threshold
- **Consistency**: Stable performance across multiple evaluations

Only the best miner per problem type/size bin receives rewards each epoch.

---
