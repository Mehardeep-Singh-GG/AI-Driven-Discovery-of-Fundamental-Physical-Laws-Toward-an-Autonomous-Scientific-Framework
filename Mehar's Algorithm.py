#!/usr/bin/env python3
"""
Mehar's Algorithm – Rigorous, Modular, and Reproducible Implementation

This code implements an AI-driven symbolic equation discovery framework based on
the paper "AI-Driven Discovery of Fundamental Physical Laws: Toward an Autonomous
Scientific Framework". It includes:
  - Candidate generation (using a formal grammar and recursive depth control)
  - Evolutionary operations (mutation, crossover, tournament selection)
  - A composite cost function that includes:
      • Theoretical consistency error (via a test evaluation)
      • Dimensional penalty (by computing physical dimensions)
      • Noise sensitivity penalty (via finite-difference estimates)
      • Symmetry penalty (via a reflection test on a candidate coordinate)
  - A simple reinforcement learning (Q-learning) module for adaptive parameter tuning
  - Detailed error handling and logging for reproducibility and traceability
"""

import logging
import random
import math
import numpy as np
import sympy
from sympy import symbols, sin, cos, exp, Add, Mul, Pow, Function, Float, S
from sympy import preorder_traversal

# --------------------------- Setup and Global Constants ---------------------------

# Setup logging for detailed error and progress tracking.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MeharAlgorithm")

# For reproducibility, set random seeds.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Define symbolic variables.
m, L, T, X = symbols('m L T X')
VARIABLES = [m, L, T, X]

# For dimensional analysis, assign a dictionary of physical dimensions to each variable.
# Dimensions are represented as dictionaries mapping base units to exponents.
# For example, m has dimension {'M':1} (mass), L has {'L':1} (length), T has {'T':1} (time),
# and we assume X is a spatial coordinate (here we treat it as dimensionless for simplicity).
DIM_MAP = {
    m: {'M': 1},
    L: {'L': 1},
    T: {'T': 1},
    X: {}  # Assume X is dimensionless in this example.
}

# Target dimension for each term (e.g., energy: M·L²·T⁻²).
TARGET_DIM = {'M': 1, 'L': 2, 'T': -2}

# Basic operators and functions available in our grammar.
OPERATORS = ['+', '-', '*', '/']
FUNCTIONS = [sin, cos, exp]

# Global weights for cost function components.
LAMBDA_THEORETICAL = 1.0  # Weight for theoretical consistency error (Λ(E))
LAMBDA_COMPLEXITY = 1.0  # Weight for complexity (C(E))
LAMBDA_DIMENSIONAL = 1.0  # Weight for dimensional penalty (D(E))
LAMBDA_NOISE = 1.0  # Weight for noise sensitivity penalty (N(E))
LAMBDA_SYMMETRY = 1.0  # Weight for symmetry penalty (S(E))

# Noise parameter for finite-difference evaluation.
NOISE_BETA = 0.01


# --------------------------- Dimension Helpers ---------------------------

def multiply_dims(dim1, dim2):
    """
    Multiply two dimension dictionaries (i.e., add exponents).
    """
    result = dim1.copy()
    for unit, exp in dim2.items():
        result[unit] = result.get(unit, 0) + exp
        if result[unit] == 0:
            del result[unit]
    return result


def power_dims(dim, exponent):
    """
    Multiply all exponents in dim by exponent.
    """
    return {unit: exp * exponent for unit, exp in dim.items()}


def dims_equal(dim1, dim2):
    """
    Check if two dimension dictionaries are equal.
    """
    return dim1 == dim2


def dimension_difference(dim, target):
    """
    Compute a penalty metric based on the difference between dim and target.
    Returns the sum of absolute differences for each base unit.
    """
    units = set(dim.keys()) | set(target.keys())
    diff = 0
    for u in units:
        diff += abs(dim.get(u, 0) - target.get(u, 0))
    return diff


def get_dimension(expr):
    """
    Recursively compute the physical dimension of a sympy expression.
    Returns a dictionary representing the dimension or None if inconsistent.
    Rules:
      - A symbol: if in DIM_MAP, return its dimension; otherwise, assume dimensionless.
      - A number: dimensionless {}.
      - Add: all addends must have identical dimensions.
      - Mul: multiply dimensions (i.e., add exponents).
      - Pow: if exponent is a number, multiply base dimension by exponent.
      - Function (e.g., sin, cos, exp): argument must be dimensionless; result is dimensionless.
      - If an inconsistency is found, return None.
    """
    try:
        if expr.is_Number:
            return {}
        if expr.is_Symbol:
            return DIM_MAP.get(expr, {})
        if isinstance(expr, Add):
            # All terms must have the same dimension.
            dims = [get_dimension(arg) for arg in expr.args]
            # If any term is inconsistent, propagate None.
            if any(d is None for d in dims):
                return None
            first_dim = dims[0]
            for d in dims[1:]:
                if not dims_equal(first_dim, d):
                    return None
            return first_dim
        if isinstance(expr, Mul):
            dim = {}
            for arg in expr.args:
                arg_dim = get_dimension(arg)
                if arg_dim is None:
                    return None
                dim = multiply_dims(dim, arg_dim)
            return dim
        if isinstance(expr, Pow):
            base, exponent = expr.args
            base_dim = get_dimension(base)
            # Exponent must be a number.
            if not exponent.is_Number:
                return None
            exponent_val = float(exponent)
            # If base has nontrivial dimension and exponent is non-integer, it's usually invalid.
            if base_dim and not float(exponent_val).is_integer():
                return None
            return power_dims(base_dim, exponent_val)
        if expr.is_Function:
            # For sin, cos, exp: argument must be dimensionless.
            for arg in expr.args:
                arg_dim = get_dimension(arg)
                if arg_dim is None or arg_dim != {}:
                    return None
            return {}  # Result is dimensionless.
        # Fallback: if expression has arguments, combine them multiplicatively.
        if expr.args:
            dim = {}
            for arg in expr.args:
                arg_dim = get_dimension(arg)
                if arg_dim is None:
                    return None
                dim = multiply_dims(dim, arg_dim)
            return dim
        return {}  # Default to dimensionless.
    except Exception as e:
        logger.error(f"Error computing dimension for {expr}: {e}")
        return None


# --------------------------- Candidate Equation Class ---------------------------

class CandidateEquation:
    """
    Represents a candidate equation.

    Attributes:
      - expression: a sympy expression representing the candidate equation.
      - derivation_history: list recording mutation/crossover steps.
      - complexity: an integer metric computed from the expression tree.
    """

    def __init__(self, expression, derivation_history=None):
        self.expression = expression
        self.derivation_history = derivation_history if derivation_history is not None else []
        self.complexity = self.compute_complexity()

    def compute_complexity(self):
        """
        Computes a simple complexity metric as the total number of nodes in the expression tree.
        """
        try:
            nodes = list(preorder_traversal(self.expression))
            return len(nodes)
        except Exception as e:
            logger.error(f"Error computing complexity: {e}")
            return float('inf')

    def cost(self):
        """
        Computes the overall cost function:
          J(E) = Λ(E) + λ1*C(E) + λ2*D(E) + λ3*N(E) + λ4*S(E)
        where each component is computed by a dedicated function.
        """
        try:
            theoretical_error = compute_theoretical_error(self.expression)
            complexity_penalty = self.complexity
            dimensional_penalty = compute_dimensional_penalty(self.expression)
            noise_penalty = compute_noise_penalty(self.expression)
            symmetry_penalty = compute_symmetry_penalty(self.expression)

            cost_value = (LAMBDA_THEORETICAL * theoretical_error +
                          LAMBDA_COMPLEXITY * complexity_penalty +
                          LAMBDA_DIMENSIONAL * dimensional_penalty +
                          LAMBDA_NOISE * noise_penalty +
                          LAMBDA_SYMMETRY * symmetry_penalty)
            return cost_value
        except Exception as e:
            logger.error(f"Error computing cost: {e}")
            return float('inf')

    def __str__(self):
        return f"Expression: {str(self.expression)}\nComplexity: {self.complexity}\nCost: {self.cost():.4f}"


# --------------------------- Cost Component Functions ---------------------------

def compute_theoretical_error(expr):
    """
    Computes a theoretical consistency error.
    Here we evaluate the expression at a nominal test point (all variables set to 1) and
    assume the target value is 0. The error is the absolute deviation.
    """
    try:
        test_vals = {m: 1, L: 1, T: 1, X: 1}
        val = expr.evalf(subs=test_vals)
        error = abs(val)  # Our target is zero.
        return float(error)
    except Exception as e:
        logger.error(f"Error computing theoretical error for {expr}: {e}")
        return float('inf')


def compute_dimensional_penalty(expr):
    """
    Computes the dimensional penalty D(E) by decomposing the expression into terms
    and comparing the computed dimension of each term with the TARGET_DIM.
    If a term is dimensionally inconsistent (i.e. get_dimension returns None),
    a heavy penalty is applied.
    """
    try:
        terms = expr.as_ordered_terms()
        penalty = 0.0
        for term in terms:
            term_dim = get_dimension(term)
            if term_dim is None:
                penalty += 10.0  # Heavy penalty for inconsistency.
            else:
                penalty += dimension_difference(term_dim, TARGET_DIM)
        return penalty
    except Exception as e:
        logger.error(f"Error computing dimensional penalty for {expr}: {e}")
        return float('inf')


def compute_noise_penalty(expr):
    """
    Computes the noise sensitivity penalty N(E) by approximating the sensitivity
    of the expression to small perturbations in its inputs.
    For each variable in VARIABLES, we compute a finite-difference estimate.
    The penalty is the average relative change.
    """
    try:
        sample_vals = {var: 1.0 for var in VARIABLES}
        base_val = expr.evalf(subs=sample_vals)
        if base_val == 0:
            base_val = 1e-6  # Avoid division by zero.
        diffs = []
        for var in VARIABLES:
            perturbed_vals = sample_vals.copy()
            noise = NOISE_BETA * sample_vals[var]
            perturbed_vals[var] += noise
            perturbed_val = expr.evalf(subs=perturbed_vals)
            diff = abs(perturbed_val - base_val) / abs(noise)
            diffs.append(diff)
        return float(np.mean(diffs))
    except Exception as e:
        logger.error(f"Error computing noise penalty for {expr}: {e}")
        return float('inf')


def compute_symmetry_penalty(expr):
    """
    Computes the symmetry penalty S(E). For this example we test for reflection
    symmetry with respect to the variable X (i.e., f(X) should equal f(-X)).
    If the expression does not depend on X, the penalty is 0.
    Otherwise, sample a few values and compute the L2 norm of the difference.
    """
    try:
        # If X is not in the free symbols, assume symmetry is not an issue.
        if X not in expr.free_symbols:
            return 0.0

        sample_points = np.linspace(-1, 1, 5)
        differences = []
        for sp in sample_points:
            subs1 = {var: 1.0 for var in VARIABLES}
            subs2 = {var: 1.0 for var in VARIABLES}
            subs1[X] = sp
            subs2[X] = -sp
            val1 = expr.evalf(subs=subs1)
            val2 = expr.evalf(subs=subs2)
            differences.append(float(abs(val1 - val2)))
        # Use the L2 norm of differences as the penalty.
        return float(np.linalg.norm(differences))
    except Exception as e:
        logger.error(f"Error computing symmetry penalty for {expr}: {e}")
        return float('inf')


# --------------------------- Expression Generation ---------------------------

def generate_random_expression(max_depth, current_depth=0):
    """
    Recursively generates a random sympy expression based on a formal grammar.
    At maximum depth, returns a terminal (variable or constant). Otherwise, randomly chooses
    to create a binary operation or apply a function.
    """
    try:
        if current_depth >= max_depth:
            # Terminal: choose a variable or a constant.
            if random.random() < 0.5:
                return random.choice(VARIABLES)
            else:
                return Float(round(random.uniform(0, 10), 2))
        else:
            if random.random() < 0.5:
                # Binary operation
                left = generate_random_expression(max_depth, current_depth + 1)
                right = generate_random_expression(max_depth, current_depth + 1)
                op = random.choice(OPERATORS)
                if op == '+':
                    return left + right
                elif op == '-':
                    return left - right
                elif op == '*':
                    return left * right
                elif op == '/':
                    try:
                        # Protect against division by zero.
                        if right == 0:
                            right += Float(1e-6)
                    except Exception as e:
                        logger.error(f"Error in division check: {e}")
                    return left / right
            else:
                # Function application
                func = random.choice(FUNCTIONS)
                arg = generate_random_expression(max_depth, current_depth + 1)
                return func(arg)
    except Exception as e:
        logger.error(f"Error generating random expression: {e}")
        return Float(1)


# --------------------------- Mutation and Crossover ---------------------------

def mutate_expression(expr):
    """
    Performs a mutation on a sympy expression.
    Randomly selects a subtree (via a preorder traversal) and replaces it with a new random subtree.
    """
    try:
        subexpressions = list(preorder_traversal(expr))
        if not subexpressions:
            return expr
        target = random.choice(subexpressions)
        new_subexpr = generate_random_expression(max_depth=2)
        mutated_expr = expr.xreplace({target: new_subexpr})
        return mutated_expr
    except Exception as e:
        logger.error(f"Error during mutation: {e}")
        return expr


def crossover_expressions(expr1, expr2):
    """
    Performs crossover between two sympy expressions.
    Randomly selects a subtree from each expression and swaps them.
    Returns the new expressions.
    """
    try:
        subexprs1 = list(preorder_traversal(expr1))
        subexprs2 = list(preorder_traversal(expr2))
        if not subexprs1 or not subexprs2:
            return expr1, expr2
        subtree1 = random.choice(subexprs1)
        subtree2 = random.choice(subexprs2)
        new_expr1 = expr1.xreplace({subtree1: subtree2})
        new_expr2 = expr2.xreplace({subtree2: subtree1})
        return new_expr1, new_expr2
    except Exception as e:
        logger.error(f"Error during crossover: {e}")
        return expr1, expr2


# --------------------------- Selection Operator ---------------------------

def tournament_selection(population, tournament_size=3):
    """
    Selects one candidate from the population using tournament selection.
    """
    try:
        competitors = random.sample(population, tournament_size)
        competitors.sort(key=lambda candidate: candidate.cost())
        return competitors[0]
    except Exception as e:
        logger.error(f"Error in tournament selection: {e}")
        return random.choice(population)


# --------------------------- Reinforcement Learning Agent ---------------------------

class RLAgent:
    """
    A simple Q-learning agent to adjust parameters (mutation and crossover probabilities).

    The state is defined as (mean_cost, std_cost) over the current population.
    The agent updates its Q-table based on the reward (improvement in best candidate cost).
    """

    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table = {}  # Maps state -> {action: Q-value}

    def get_state(self, population):
        """
        Defines the state as a tuple (mean_cost, std_cost) computed from the candidate costs.
        """
        costs = [candidate.cost() for candidate in population]
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        return (round(mean_cost, 2), round(std_cost, 2))

    def choose_action(self, state):
        """
        Selects an action using an epsilon-greedy strategy.
        """
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.q_table.get(state, {action: 0 for action in self.actions})
            return max(q_values, key=q_values.get)

    def update_q(self, state, action, reward, next_state):
        """
        Performs the Q-learning update.
        """
        try:
            if state not in self.q_table:
                self.q_table[state] = {a: 0 for a in self.actions}
            if next_state not in self.q_table:
                self.q_table[next_state] = {a: 0 for a in self.actions}
            old_value = self.q_table[state][action]
            next_max = max(self.q_table[next_state].values())
            new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)
            self.q_table[state][action] = new_value
        except Exception as e:
            logger.error(f"Error in Q-learning update: {e}")


# Define possible actions for RL adjustment.
ACTIONS = ['increase_pm', 'decrease_pm', 'increase_pc', 'decrease_pc']


# --------------------------- Mehar's Algorithm Class ---------------------------

class MeharAlgorithm:
    """
    Implements the full evolutionary algorithm with reinforcement learning adaptation.

    Attributes:
      - population_size: number of candidate equations per generation.
      - generations: total number of generations.
      - pm: mutation probability.
      - pc: crossover probability.
      - max_depth: maximum depth for the random expression generator (iterative deepening).
      - population: current list of CandidateEquation objects.
      - rl_agent: instance of RLAgent for parameter tuning.
    """

    def __init__(self, population_size=50, generations=20, initial_pm=0.2, initial_pc=0.7, max_depth=3):
        self.population_size = population_size
        self.generations = generations
        self.pm = initial_pm  # Mutation probability.
        self.pc = initial_pc  # Crossover probability.
        self.max_depth = max_depth
        self.population = []
        self.rl_agent = RLAgent(actions=ACTIONS)

    def initialize_population(self):
        """
        Initializes the candidate population by generating random expressions.
        """
        self.population = []
        for _ in range(self.population_size):
            expr = generate_random_expression(self.max_depth)
            candidate = CandidateEquation(expr)
            self.population.append(candidate)
        logger.info("Population initialized.")

    def evolve_population(self):
        """
        Evolves the population over the specified number of generations.
        Each generation involves evaluation, RL-based parameter update, selection,
        crossover, mutation, and iterative deepening.
        """
        for gen in range(self.generations):
            logger.info(f"\n--- Generation {gen + 1}/{self.generations} ---")
            new_population = []

            # Update candidates: compute complexity and cost.
            for candidate in self.population:
                try:
                    candidate.complexity = candidate.compute_complexity()
                    _ = candidate.cost()
                except Exception as e:
                    logger.error(f"Error evaluating candidate: {e}")

            # Reinforcement Learning: choose an action to adjust parameters.
            state = self.rl_agent.get_state(self.population)
            action = self.rl_agent.choose_action(state)
            if action == 'increase_pm':
                self.pm = min(1.0, self.pm + 0.05)
            elif action == 'decrease_pm':
                self.pm = max(0.0, self.pm - 0.05)
            elif action == 'increase_pc':
                self.pc = min(1.0, self.pc + 0.05)
            elif action == 'decrease_pc':
                self.pc = max(0.0, self.pc - 0.05)
            logger.info(f"RL action: {action} -> pm: {self.pm:.2f}, pc: {self.pc:.2f}")

            # Create new candidates via selection, crossover, and mutation.
            while len(new_population) < self.population_size:
                parent1 = tournament_selection(self.population)
                parent2 = tournament_selection(self.population)
                child_expr = parent1.expression

                # Apply crossover with probability pc.
                if random.random() < self.pc:
                    child_expr, _ = crossover_expressions(parent1.expression, parent2.expression)

                # Apply mutation with probability pm.
                if random.random() < self.pm:
                    child_expr = mutate_expression(child_expr)

                derivation = parent1.derivation_history + parent2.derivation_history
                derivation.append(f"Generated in generation {gen + 1}")
                try:
                    child_candidate = CandidateEquation(child_expr, derivation_history=derivation)
                except Exception as e:
                    logger.error(f"Error creating child candidate: {e}")
                    continue
                new_population.append(child_candidate)

            # Compute RL reward as improvement in best candidate cost.
            best_cost_before = min(candidate.cost() for candidate in self.population)
            best_cost_after = min(candidate.cost() for candidate in new_population)
            reward = best_cost_before - best_cost_after  # Reward is improvement.
            next_state = self.rl_agent.get_state(new_population)
            self.rl_agent.update_q(state, action, reward, next_state)

            # Replace the population with the new one.
            self.population = new_population

            # Iterative deepening: allow deeper expressions as generations progress.
            self.max_depth += 1
            best_candidate = min(self.population, key=lambda c: c.cost())
            logger.info(f"Best candidate in generation {gen + 1}:\n{best_candidate}")

    def run(self):
        """
        Runs the entire Mehar's Algorithm, returning the best candidate found.
        """
        try:
            self.initialize_population()
            self.evolve_population()
            best_candidate = min(self.population, key=lambda candidate: candidate.cost())
            logger.info("\nAlgorithm complete.")
            logger.info(f"Best candidate found:\n{best_candidate}")
            return best_candidate
        except Exception as e:
            logger.error(f"Error during algorithm run: {e}")
            return None


# --------------------------- Main Execution ---------------------------

if __name__ == "__main__":
    # Create an instance of Mehar's Algorithm with desired parameters.
    algorithm = MeharAlgorithm(population_size=30, generations=10, initial_pm=0.2, initial_pc=0.7, max_depth=3)
    best_candidate = algorithm.run()
    if best_candidate:
        print("\nBest Candidate Equation:")
        print(best_candidate)
    else:
        print("Algorithm failed to produce a valid candidate.")
