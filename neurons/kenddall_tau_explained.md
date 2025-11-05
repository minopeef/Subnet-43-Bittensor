Excellent — this is a key concept for what you’re doing (measuring rank stability).
Let’s break Kendall distance (and Kendall’s tau) down step by step.

⸻

🧩 Intuition

The Kendall distance measures how different two rankings are.

If you have two orderings of the same items (e.g. miners ranked by two different problem instances), Kendall distance counts how many pairwise order disagreements exist between them.

⸻

👇 Example

Suppose we have 4 miners: A, B, C, D.

Two rankings:

Miner	Ranking 1	Ranking 2
A	1	1
B	2	3
C	3	2
D	4	4

So:
	•	Ranking 1: A > B > C > D
	•	Ranking 2: A > C > B > D

Now look at all pairs of miners (there are \binom{4}{2} = 6 total pairs).

Pair	Ranking 1 order	Ranking 2 order	Agreement?
A,B	A > B	A > B	✅
A,C	A > C	A > C	✅
A,D	A > D	A > D	✅
B,C	B > C	C > B	❌
B,D	B > D	B > D	✅
C,D	C > D	C > D	✅

There’s 1 discordant pair (B,C).

⸻

📏 So the Kendall distance = number of discordant pairs = 1.

To normalize it (so it’s between 0 and 1):

K = \frac{\text{discordant pairs}}{\text{total pairs}} = \frac{1}{6} \approx 0.167

⸻

🔁 Relation to Kendall’s Tau (τ)

Kendall’s tau is just a normalized correlation version of this distance:

\tau = 1 - 2K = \frac{n_c - n_d}{\binom{n}{2}}
where:
	•	n_c = number of concordant pairs,
	•	n_d = number of discordant pairs.

So:
	•	τ = 1.0 → rankings identical
	•	τ = 0.0 → rankings random / uncorrelated
	•	τ = −1.0 → rankings completely reversed

In our example:
\tau = 1 - 2 \times 0.167 = 0.667

⸻

🧠 Intuitive Meaning

τ (or Kendall distance)	Interpretation
τ ≈ 1 (distance ≈ 0)	Rankings nearly identical
τ ≈ 0.5	Mostly similar, a few swaps
τ ≈ 0	Unrelated or random rankings
τ ≈ −1 (distance ≈ 1)	Rankings fully inverted


⸻

🧮 Implementation

You can compute Kendall’s tau (and distance) easily in Python:

from scipy.stats import kendalltau
import numpy as np

r1 = [1, 2, 3, 4]  # ranking 1
r2 = [1, 3, 2, 4]  # ranking 2

tau, p_value = kendalltau(r1, r2)
kendall_distance = (1 - tau) / 2

print("tau =", tau)
print("distance =", kendall_distance)

Output:

tau = 0.6666...
distance = 0.1666...


⸻

🔬 Bittensor Miners

Kendall distance or τ is used to measure how stable the miner rankings are from one round (problem) to another, e.g.:
	•	If τ is high (close to 1) between successive rounds, the rankings are stable.
	•	If τ keeps fluctuating or stays low, solvers are still producing inconsistent results.

So when you said earlier:

“I want to stop generating problems when variance in their rankings drops below a threshold,”

you can equivalently say:

“I want to stop when the Kendall tau between successive rankings stays above 0.98 for several rounds.”

⸻

Would you like me to show how to compute Kendall distance between a current ranking and the cumulative historical mean ranking (like what you’d use to detect convergence globally)?