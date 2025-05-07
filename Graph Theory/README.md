# Graph Algorithms: Label Propagation and Maximum Clique Detection

This project demonstrates two fundamental graph algorithms:

1. **Label Propagation** – for community detection.
2. **Maximum Clique Detection** – for finding the largest complete subgraph.

---

## Label Propagation Algorithm (LPA)

The Label Propagation Algorithm is a semi-supervised algorithm used for **community detection** in networks. It operates in an **iterative, local, and randomized** fashion.

### **Algorithm Steps:**

Given an undirected graph \( G = (V, E) \):

1. **Initialization**:
   Each node \( v_i \in V \) is assigned a unique label \( l_i = i \).

2. **Propagation**:
   At each iteration, update each node's label to the **most frequent label** among its neighbors:
   \[
   l_i^{(t+1)} = \text{argmax}_l \; \text{count of } l \text{ in neighbors of } v_i
   \]
   Ties are broken **randomly**.

3. **Convergence**:
   Repeat until labels stabilize (no changes in a full iteration) or a maximum number of iterations is reached.

### **Properties**:
- Time Complexity: \( \mathcal{O}(|E|) \)
- Non-deterministic (due to random tie-breaking)
- Works well on large graphs with a clear community structure

---

## Maximum Clique Detection via MILP

A **clique** is a subset of vertices \( C \subseteq V \) such that every pair of nodes in \( C \) is connected:
\[
\forall u, v \in C, \; (u, v) \in E
\]
The **maximum clique** problem seeks the largest such set.

### **Formulation as a MILP (Mixed Integer Linear Program)**:

Let \( x_i \in \{0, 1\} \) indicate whether vertex \( i \) is in the clique.

We maximize:
\[
\max \sum_{i=1}^{n} x_i
\]

Subject to:
- For every non-edge \( (i, j) \notin E \), at most one of \( x_i \) or \( x_j \) can be in the clique:
\[
x_i + x_j \leq 1
\]

This ensures the chosen set of nodes forms a **complete subgraph**.

### **Implementation Highlights**:
- We first compute the **complement adjacency matrix**.
- Then, use `scipy.optimize.milp` to solve the MILP problem.
- Output is a binary vector indicating membership in the clique.

### **Complexity**:
- NP-hard problem
- MILP solvers work well for small/medium graphs

---

## Files

- `Label_Propagation_Demo.ipynb` – Interactive notebook for community detection
- `Max_Clique_Detection.ipynb` – Notebook using MILP to find largest cliques
- `README.md` – This documentation

---

## Dependencies

- `numpy`
- `matplotlib`
- `networkx`
- `scipy` (for MILP)

---

## Example Outputs

- Label Propagation clusters the graph into cohesive communities.
- Max Clique detection highlights the largest fully connected subset of nodes.

---
