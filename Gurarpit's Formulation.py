#!/usr/bin/env python3
"""
Gurarpit's Module: Experimental Validation

This module implements the experimental validation component of the AI-driven
discovery framework as described in:
"AI-Driven Discovery of Fundamental Physical Laws: Toward an Autonomous Scientific Framework".

It includes:
  - Sampling of experimental protocols.
  - The Babel operator to translate a candidate equation into simulated observables.
  - Reverse engineering of the candidate’s derivational history.
  - A detailed validation cost function combining:
      • Reverse-engineering error Λ_rev(E)
      • Experimental protocol complexity C_exp(E)
      • Simulation deviation Δ_sim(E) (RMSE between simulated and reference data)
      • Experimental reproducibility Δ_exp(E)
  - A simple reinforcement learning (RL) module to adapt simulation parameters.

The overall validation cost is computed as:
   J_G(E) = Λ_rev(E) + λ5 * C_exp(E) + λ6 * Δ_sim(E) + λ7 * Δ_exp(E)
"""

import logging
import random
import numpy as np
import sympy
from sympy import symbols, sin, cos, exp, Float, lambdify

# --------------------------- Setup and Global Constants ---------------------------

# Setup logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GurarpitsModule")

# Reproducibility.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Define symbolic variables (should match those in Mehar's module).
m, L, T, X = symbols('m L T X')
VARIABLES = [m, L, T, X]

# Global weights for the validation cost function.
LAMBDA_REV = 1.0  # Weight for reverse engineering error Λ_rev(E)
LAMBDA_CEXP = 1.0  # Weight for experimental protocol complexity C_exp(E)
LAMBDA_DSIM = 1.0  # Weight for simulation deviation Δ_sim(E)
LAMBDA_DEXP = 1.0  # Weight for experimental reproducibility Δ_exp(E)

# Dummy reference observable (in practice, use actual experimental data).
REFERENCE_OBSERVABLE = np.zeros(50)  # 50 data points, here all zeros


# --------------------------- Experimental Protocol and Simulation ---------------------------

class Protocol:
    """
    Represents an experimental protocol.

    Each protocol is a sequence of steps. Each step is a dictionary with keys:
      - 'action': either 'measure' or 'inject'
      - 'quantity': the physical quantity involved (e.g., 'V' for voltage, 'T' for temperature)
      - 'time': the time at which the step occurs (or duration for injection)
      - 'value': (optional) for injection steps, the magnitude of the injected signal
    """

    def __init__(self, steps):
        self.steps = steps

    def complexity(self):
        """Return the complexity as the number of steps."""
        return len(self.steps)

    def __str__(self):
        return " -> ".join([str(step) for step in self.steps])


def sample_protocols(num_protocols=5):
    """
    Samples a set of experimental protocols from the infinite protocol space.
    For demonstration, this function generates protocols with a random number of steps.
    """
    protocols = []
    for _ in range(num_protocols):
        num_steps = random.randint(2, 5)  # between 2 and 5 steps
        steps = []
        for _ in range(num_steps):
            action = random.choice(['measure', 'inject'])
            quantity = random.choice(['V', 'T', 'F'])  # Example: Voltage, Temperature, Force
            time_val = round(random.uniform(0.0, 2.0), 2)
            if action == 'inject':
                value = round(random.uniform(1.0, 10.0), 2)
                step = {'action': action, 'quantity': quantity, 'time': time_val, 'value': value}
            else:
                step = {'action': action, 'quantity': quantity, 'time': time_val}
            steps.append(step)
        protocols.append(Protocol(steps))
    return protocols


def babel_operator(candidate_expr, protocol):
    """
    The Babel operator translates a candidate equation and an experimental protocol
    into a simulated observable.

    For demonstration:
      - We assume candidate_expr is a function of time T.
      - A time grid from 0 to 2 seconds (50 points) is defined.
      - The candidate expression is evaluated over this grid.
      - Injection steps add perturbations at the appropriate time index.

    Returns:
      A numpy array of simulated observable values.
    """
    try:
        # Convert candidate expression to a numerical function of T.
        func = lambdify(T, candidate_expr, "numpy")
        time_grid = np.linspace(0, 2, 50)
        y = func(time_grid)
        y = np.array(y, dtype=np.float64)

        # Process protocol steps.
        for step in protocol.steps:
            if step['action'] == 'inject':
                t_inj = step['time']
                index = int((t_inj / 2) * (len(time_grid) - 1))
                y[index:] += step['value']
        return y
    except Exception as e:
        logger.error(f"Error in Babel operator with protocol {protocol}: {e}")
        return np.zeros(50)


def reverse_engineer(candidate):
    """
    Reverse-engineers the candidate’s derivational history.
    For demonstration, we assume each transformation recorded in candidate.derivation_history
    contributes a fixed error delta. The cumulative derivation error is:
      Λ_rev(E) = delta * (number of transformation steps)
    """
    try:
        if not candidate.derivation_history:
            return 0.0
        delta = 0.1  # error per transformation step
        return delta * len(candidate.derivation_history)
    except Exception as e:
        logger.error(f"Error in reverse engineering for candidate {candidate}: {e}")
        return float('inf')


def compute_rmse(simulated, reference):
    """
    Computes the root mean squared error (RMSE) between the simulated observable and reference data.
    """
    try:
        mse = np.mean((simulated - reference) ** 2)
        return np.sqrt(mse)
    except Exception as e:
        logger.error(f"Error computing RMSE: {e}")
        return float('inf')


def experimental_reproducibility(candidate_expr, protocol, repetitions=10):
    """
    Computes an experimental reproducibility metric Δ_exp(E) by simulating the candidate expression
    multiple times with slight noise variations in the protocol.

    Returns:
      The average standard deviation of the simulated observable over all repetitions.
    """
    try:
        observables = []
        for _ in range(repetitions):
            # Introduce slight noise in injection values.
            noisy_protocol = Protocol([
                {**step, 'value': step.get('value', 0) * (1 + random.uniform(-0.05, 0.05))}
                for step in protocol.steps
            ])
            y = babel_operator(candidate_expr, noisy_protocol)
            observables.append(y)
        observables = np.array(observables)
        std_dev = np.mean(np.std(observables, axis=0))
        return std_dev
    except Exception as e:
        logger.error(f"Error computing experimental reproducibility: {e}")
        return float('inf')


def compute_validation_cost(candidate):
    """
    Computes the overall validation cost J_G(E) for a candidate equation:
      J_G(E) = Λ_rev(E) + λ5 * C_exp(E) + λ6 * Δ_sim(E) + λ7 * Δ_exp(E)
    where:
      - Λ_rev(E) is the cumulative reverse engineering error.
      - C_exp(E) is the experimental protocol complexity (average number of steps).
      - Δ_sim(E) is the RMSE between simulated and reference observables.
      - Δ_exp(E) is the experimental reproducibility error.
    """
    try:
        protocols = sample_protocols(num_protocols=3)
        rmse_list = []
        reproducibility_list = []
        complexity_list = []
        for protocol in protocols:
            simulated = babel_operator(candidate.expression, protocol)
            rmse_list.append(compute_rmse(simulated, REFERENCE_OBSERVABLE))
            complexity_list.append(protocol.complexity())
            reproducibility_list.append(experimental_reproducibility(candidate.expression, protocol))
        delta_sim = np.mean(rmse_list)
        delta_exp = np.mean(reproducibility_list)
        c_exp = np.mean(complexity_list)
        rev_error = reverse_engineer(candidate)
        validation_cost = rev_error + LAMBDA_CEXP * c_exp + LAMBDA_DSIM * delta_sim + LAMBDA_DEXP * delta_exp
        return validation_cost
    except Exception as e:
        logger.error(f"Error computing validation cost for candidate {candidate}: {e}")
        return float('inf')


# --------------------------- RL for Experimental Adaptation ---------------------------

class ExperimentalRLAgent:
    """
    A simple Q-learning agent for adapting simulation parameters (e.g., noise amplitude).
    The state is defined as (mean_validation_cost, std_validation_cost) over several protocol simulations.
    """

    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table = {}  # Maps state -> {action: Q-value}

    def get_state(self, validation_costs):
        mean_cost = np.mean(validation_costs)
        std_cost = np.std(validation_costs)
        return (round(mean_cost, 2), round(std_cost, 2))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = self.q_table.get(state, {a: 0 for a in self.actions})
            return max(q_values, key=q_values.get)

    def update_q(self, state, action, reward, next_state):
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
            logger.error(f"Error in experimental Q-learning update: {e}")


# Define possible actions for experimental RL adaptation.
EXP_ACTIONS = ['increase_noise', 'decrease_noise']
SIM_NOISE_AMPLITUDE = 0.05  # Global noise amplitude for simulation adjustments.


# --------------------------- Gurarpit's Validation Module ---------------------------

class GurarpitsModule:
    """
    Implements experimental validation for candidate equations.

    Workflow:
      1. Sample experimental protocols.
      2. For each protocol, use the Babel operator to simulate observables.
      3. Reverse-engineer the candidate’s derivation history.
      4. Compute simulation error (RMSE), protocol complexity, and reproducibility.
      5. Aggregate these into an overall validation cost J_G(E).
      6. Optionally adapt simulation parameters using RL.
    """

    def __init__(self):
        self.rl_agent = ExperimentalRLAgent(actions=EXP_ACTIONS)

    def validate_candidate(self, candidate):
        try:
            # Compute the overall validation cost.
            validation_cost = compute_validation_cost(candidate)

            # For RL adaptation, simulate over multiple protocols.
            protocols = sample_protocols(num_protocols=5)
            costs = []
            for protocol in protocols:
                simulated = babel_operator(candidate.expression, protocol)
                rmse = compute_rmse(simulated, REFERENCE_OBSERVABLE)
                rep = experimental_reproducibility(candidate.expression, protocol)
                c_exp = protocol.complexity()
                rev_error = reverse_engineer(candidate)
                cost = rev_error + LAMBDA_CEXP * c_exp + LAMBDA_DSIM * rmse + LAMBDA_DEXP * rep
                costs.append(cost)

            state = self.rl_agent.get_state(costs)
            action = self.rl_agent.choose_action(state)
            # Adjust the simulation noise amplitude based on the action.
            global SIM_NOISE_AMPLITUDE
            if action == 'increase_noise':
                SIM_NOISE_AMPLITUDE = min(0.1, SIM_NOISE_AMPLITUDE + 0.005)
            elif action == 'decrease_noise':
                SIM_NOISE_AMPLITUDE = max(0.01, SIM_NOISE_AMPLITUDE - 0.005)
            next_state = self.rl_agent.get_state(costs)
            reward = -validation_cost  # Lower validation cost yields a higher reward.
            self.rl_agent.update_q(state, action, reward, next_state)

            # Generate a detailed validation report.
            report = {
                'validation_cost': validation_cost,
                'reverse_engineering_error': reverse_engineer(candidate),
                'protocol_complexity': np.mean([p.complexity() for p in protocols]),
                'rmse': np.mean([compute_rmse(babel_operator(candidate.expression, p), REFERENCE_OBSERVABLE)
                                 for p in protocols]),
                'reproducibility': np.mean([experimental_reproducibility(candidate.expression, p)
                                            for p in protocols]),
                'SIM_NOISE_AMPLITUDE': SIM_NOISE_AMPLITUDE
            }
            return report
        except Exception as e:
            logger.error(f"Error validating candidate {candidate}: {e}")
            return None


# --------------------------- Main Execution ---------------------------

if __name__ == "__main__":
    # For testing purposes, we create a dummy candidate equation.
    # In practice, this candidate would come from Mehar's Algorithm.
    candidate_expr = sin(T) + Float(2.0) * cos(T)
    candidate_derivation = ["Initial generation", "Mutation applied"]


    # Mimic a candidate object similar to CandidateEquation.
    class DummyCandidate:
        pass


    candidate = DummyCandidate()
    candidate.expression = candidate_expr
    candidate.derivation_history = candidate_derivation

    # Create an instance of Gurarpit's Module.
    gurarpit = GurarpitsModule()
    report = gurarpit.validate_candidate(candidate)

    if report:
        print("\nGurarpit's Validation Report:")
        for key, value in report.items():
            print(f"{key}: {value}")
    else:
        print("Validation failed.")
