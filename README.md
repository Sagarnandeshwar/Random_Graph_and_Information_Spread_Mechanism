# Social Network Modeling & Information Spread Simulation

> **Final Project – COMP 400: Social Networks and Flow of Information**  
> **Author:** Sagar Nandeshwar  
> **Supervisor:** Prof. Joseph Vybihal  
> McGill University

---

## 📌 Overview

This project introduces a tunable framework to simulate **social network graphs** and a realistic **information spread mechanism**, benchmarked against real Twitter data. The framework captures user behavior through geometric proximity, community similarity, and influence levels. It also proposes a hybrid information propagation model inspired by Bootstrap and First-Passage Percolation.

---

## 🧠 Objectives

- Generate random graphs that mimic Twitter-like properties.
- Analyze structural metrics like centrality, clustering, and diameter.
- Simulate message spread using a custom percolation-based algorithm.
- Detect group membership and measure influence dynamics.

---

## 🕸️ Network Framework

### 👤 Node Attributes

- `position`: 2D location of the user (torus-based)
- `group`: Community or category
- `followers` / `following`: Directed edges
- `postThreshold`: User's likelihood to post shared content
- `view` / `post`: Messages received and posted

### 🔗 Edge Formation

The probability of an edge from node `u → v` is based on:

- **Distance**: Users close in space are more likely to connect
- **Similarity**: Higher chance if users belong to the same group
- **Influence**: Nodes with higher follower-to-following ratio are more attractive

### ⚙️ Tunable Parameters

| Parameter           | Description                                       | Range     |
|--------------------|---------------------------------------------------|-----------|
| `similarity_score` | Strength of group-based attraction                | 0 – 0.5   |
| `distanceBoundary` | Effective radius for proximity                    | 0 – 1     |
| `influenceFactor`  | Significance of influence in edge probability     | 0 – 2     |
| `strictness`       | Overall bias strength                             | 1 – 10    |
| `groupListProb`    | Distribution of community membership              | Custom    |

---

## 📊 Comparison with Twitter Graphs

The proposed model was benchmarked against real Twitter data (~800 nodes). Key similarities include:

| Metric                  | Twitter         | Generated Graph |
|------------------------|-----------------|------------------|
| Degree Centrality       | ~0.41           | ~0.54            |
| Closeness Centrality    | ~0.55           | ~0.57            |
| Betweenness Centrality  | ~0.0014         | ~0.0015          |
| Eigenvector Centrality  | ~0.03           | ~0.05            |
| Clustering Coefficient  | ~0.56           | ~0.43            |
| Graph Diameter          | 4               | 3                |
| Avg. Shortest Path      | 2.1             | 1.7              |

---

## 🔄 Information Spread Model

A custom mechanism was developed by combining:

- **Bootstrap Percolation**: Activation via neighbor threshold
- **First-Passage Percolation**: Spread cost via edge weights

### 📌 Core Ideas

- **Edge weight** decreases with node similarity and sender influence
- **Message budget** decays per hop but accumulates from multiple sources
- **Posting threshold** determines when a node shares a message
- **Inactive nodes** simulate real-world dropout

### 🧪 Observations

- High-influence seeds led to almost full network propagation
- Low-influence seeds failed to break out of local communities
- Application tested on real Twitter data for backtracking message paths

---

## 🧪 Applications

- **Graph generation** that resembles real-world social networks
- **Influence analysis** for targeted marketing
- **Community inference** from message spread and follower data
- **Information flow tracing** to identify viral origin paths

---

## 📁 Project Structure
├── src/ # Python implementation of network & spread model
├── data/ # Twitter datasets (2018–2020 political clusters)
├── report/ # Final COMP 400 report (PDF)
└── README.md # You are here


---

## 🚧 Future Work

- Incorporate ML for automatic graph parameter tuning
- Simulate multi-topic message competition
- Move toward distributed propagation for real-time dynamics
- Model parameter interdependencies to reduce config complexity

---

## 🗓️ Development Timeline

| Task                         | Date           | Developer         |
|-----------------------------|----------------|-------------------|
| Resource Collection          | Sept 23, 2023  | Sagar Nandeshwar  |
| Network Framework            | Oct 26, 2023   | Sagar Nandeshwar  |
| Data Collection & Processing| Nov 15, 2023   | Sagar Nandeshwar  |
| Spread Mechanism             | Dec 05, 2023   | Sagar Nandeshwar  |

---

## 📚 References

Full citation list available in `COMP 400 Report.pdf`. Key references include:

- Erdős–Rényi, Barabási–Albert models
- Geometric & Stochastic Block Models
- Bootstrap & First-Passage Percolation
- Centrality & Clustering Coefficient measures

---

## 🤝 Acknowledgments

Thanks to Prof. Joseph Vybihal for guidance and to McGill University’s COMP 400 course for the opportunity to explore this research.

