# AI-Driven Discovery of Fundamental Physical Laws

This project implements an autonomous scientific framework for discovering fundamental physical laws. Inspired by the paper:

> **AI-Driven Discovery of Fundamental Physical Laws: Toward an Autonomous Scientific Framework**

the project leverages artificial intelligence techniques—symbolic regression, evolutionary optimization, and reinforcement learning—to explore an infinite symbolic space and autonomously derive candidate equations. In addition, it incorporates a rigorous experimental validation module to assess candidate equations under simulated experimental protocols.

## Overview

The framework is divided into two primary modules:

### 1. Mehar’s Algorithm (Symbolic Equation Discovery)
- **Objective:** To autonomously generate candidate equations from an infinite symbolic space while enforcing physical constraints such as dimensional consistency, symmetry, and noise robustness.
- **Key Features:**
  - **Symbolic Representation:** Equations are represented as directed acyclic graphs (DAGs) following a formal grammar.
  - **Iterative Deepening & Monte Carlo Sampling:** The algorithm progressively explores increasingly complex expressions.
  - **Penalty Metrics:** Includes penalties for theoretical consistency error, complexity, dimensional inconsistency, noise sensitivity, and symmetry violations.
  - **Evolutionary Optimization:** Uses mutation, crossover, and tournament selection operators to evolve candidate equations.
  - **Reinforcement Learning (RL):** A Q-learning agent dynamically tunes evolutionary parameters (e.g., mutation and crossover probabilities) based on population performance.

### 2. Gurarpit’s Validation Module (Experimental Validation)
- **Objective:** To rigorously validate candidate equations by simulating experimental protocols, translating candidate equations into observables (via the Babel operator), and computing a composite validation cost.
- **Key Features:**
  - **Experimental Protocols:** Randomly generated sequences of experimental steps (e.g., measurements and injections).
  - **Babel Operator:** Converts a candidate symbolic expression into a simulated observable over a time grid, accounting for experimental actions.
  - **Reverse Engineering:** Audits the candidate’s derivational history to quantify cumulative transformation error.
  - **Error Metrics:** Computes simulation deviation (RMSE), experimental reproducibility, and protocol complexity.
  - **RL Adaptation:** A dedicated Q-learning agent adapts simulation parameters (e.g., noise amplitude) to optimize the validation process.

Both modules are implemented in Python with a focus on modularity, rigor, and reproducibility. Error handling and detailed logging are incorporated throughout the code to ensure robustness.

## Paper Summary

The referenced paper outlines a paradigm shift where AI acts as an **autonomous theorist**—not merely a tool for data processing but as an independent discoverer of new physical laws. Rather than simply rediscovering well-known equations (like Einstein’s field equations or the Schrödinger equation), the framework systematically explores an infinite space of mathematical formulations, constrained only by fundamental principles (symmetry, conservation laws, and dimensional consistency). The culmination of this process is a novel candidate—the **Unified Emergent Field Equation (UEFE)**—which integrates nonlocal terms and fractional derivatives to extend our understanding of gravitational and quantum phenomena.

## Dependencies

The project is written in Python and requires the following packages:

- **Python 3.x**  
- **Sympy:** For symbolic mathematics  
- **NumPy:** For numerical computations  
- **Logging:** (Built-in Python module for detailed logging)

You can install the required Python packages using `pip`:

```bash
pip install sympy numpy
```

## File Structure

- **mehar_algorithm.py:**  
  Contains the complete implementation of Mehar’s Algorithm. This file includes:
  - Expression generation using a formal grammar.
  - Evolutionary operators (mutation, crossover, tournament selection).
  - A composite cost function with penalties for theoretical error, complexity, dimensional inconsistency, noise sensitivity, and symmetry.
  - An RL module for adaptive parameter tuning.
  
- **gurarpits_module.py:**  
  Contains the implementation of Gurarpit’s Experimental Validation Module. This file includes:
  - Sampling and definition of experimental protocols.
  - The Babel operator that translates candidate equations into simulated observables.
  - Reverse engineering of the candidate’s derivation history.
  - Computation of error metrics (RMSE, reproducibility) and an overall validation cost.
  - An RL module for adapting simulation parameters.

## How to Run

### Running Mehar’s Algorithm

To execute the symbolic equation discovery module, run:

```bash
python mehar_algorithm.py
```

The algorithm will initialize a population of candidate equations, evolve them over several generations, and log the progress. At the end, the best candidate equation (including its symbolic representation, complexity, and cost) will be printed.

### Running Gurarpit’s Validation Module

To test the experimental validation of a candidate equation, run:

```bash
python gurarpits_module.py
```

This will simulate experimental protocols, apply the Babel operator to a candidate equation, and print a detailed validation report that includes the overall validation cost, reverse-engineering error, RMSE, reproducibility, and current simulation noise settings.

## Extensibility and Future Work

- **Penalty Refinement:**  
  The current implementations of the dimensional, noise, and symmetry penalties provide a basic approximation. These functions can be further refined to better match the specific scientific requirements and datasets.
  
- **RL Module Enhancements:**  
  Future improvements may include more advanced RL strategies or meta-optimization techniques to dynamically adjust more parameters.
  
- **Additional Operators and Domains:**  
  The expression generation can be extended to include additional mathematical operators (e.g., integral operators, fractional derivatives) and applied to other domains such as quantum chemistry, materials science, or cosmology.

## Conclusion

This project serves as a proof-of-concept for autonomous scientific discovery using AI. By combining symbolic regression, evolutionary optimization, and reinforcement learning, the framework not only reproduces known physical laws but also paves the way for the discovery of novel, unified theories. With further refinement, this approach has the potential to transform how scientific theories are developed and validated.

---

Feel free to contribute, extend, or refine any part of this project. For questions or issues, please open an issue on the repository.

Happy discovering!
