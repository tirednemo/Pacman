# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for k in range(self.iterations):
            v_k = self.values.copy()
            for state in self.mdp.getStates():
                _, bestQValue = self.computeMaxQValues(state)
                if len(self.mdp.getPossibleActions(state)) > 0:
                    v_k[state] = bestQValue
                    
            self.values = v_k


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        qValue = 0
        for sPrime, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            qValue += probability * (self.mdp.getReward(state, action, sPrime) + self.discount * self.getValue(sPrime))
        return qValue
    
    def computeMaxQValues(self, state):
        qValues = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            qValues[action] = self.getQValue(state, action)
        return qValues.argMax(), qValues[qValues.argMax()]

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestAction, _ = self.computeMaxQValues(state)
        return bestAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        i = 0
        for k in range(self.iterations):
            v_k = self.values.copy()
            _, bestQValue = self.computeMaxQValues(states[i])
            if len(self.mdp.getPossibleActions(states[i])) > 0:
                v_k[states[i]] = bestQValue
                           
            self.values = v_k
            i = (i + 1) % len(states) 

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        
        # 1. Compute predecessors of all states and store them in set.
        predecessors = {s: set() for s in self.mdp.getStates()}
        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for successor, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                        predecessors[successor].add(state)

        # 2. Initialize an empty priority queue util.PriorityQueue.
        priority_queue = util.PriorityQueue()

        # 3. For each non-terminal state s, do: 
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                # Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s 
                diff = abs(self.values[s] - max([self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)]))
                # Push s into the priority queue with priority -diff 
                priority_queue.update(s, -diff)

        # 4. For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        for _ in range(self.iterations):
            # If the priority queue is empty, then terminate.
            if priority_queue.isEmpty(): break
            # Pop a state s off the priority queue.
            s = priority_queue.pop()
            # Update s’s value (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(s):
                self.values[s] = max([self.getQValue(s, action) for action in self.mdp.getPossibleActions(s)])
            # For each predecessor p of s, do:
            for p in predecessors[s]:
                #Find the absolute value of the difference between the current value of p in self.values and the highest Q-value across all possible actions from p
                diff = abs(self.values[p] - max([self.getQValue(p, action) for action in self.mdp.getPossibleActions(p)]))
                #If diff > theta, push p into the priority queue with priority -diff as long as it does not already exist in the priority queue with equal or lower priority.
                if diff > self.theta and p not in priority_queue.heap:
                    priority_queue.update(p, -diff)