# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        g = []
        for ghost in newGhostStates:
            g.append(manhattanDistance(newPos, ghost.getPosition()))
        ghost_closest = min(g)
        if ghost_closest != 0:
            ghost_distance = -1 / ghost_closest
        else:
            ghost_distance = -1000
        food_list = newFood.asList()
        food_closet = 0
        if len(food_list) != 0:
            f = []
            for food in food_list:
                f.append(manhattanDistance(newPos, food))
            food_closet = min(f)
        value = (-0.2 * food_closet) + ghost_distance - (10 * len(food_list))
        return value

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def minimax(state, agentIndex, depth):
            if agentIndex == state.getNumAgents():
                if depth == self.depth:
                    return self.evaluationFunction(state)
                else:
                    return minimax(state, 0, depth + 1)
            else:
                moves = state.getLegalActions(agentIndex)
                if len(moves) == 0:
                    return self.evaluationFunction(state)
                next = []
                for m in moves:
                    next.append(minimax(state.generateSuccessor(agentIndex, m), agentIndex + 1, depth))
                if agentIndex != 0:
                    return min(next)
                else:
                    return max(next)
        result = max(gameState.getLegalActions(0), key=lambda x: minimax(gameState.generateSuccessor(0, x), 1, 1))
        return result




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      An expectimax agent you can use or modify
    """
    def expectimax_value(self, gameState, agentIndex, nodeDepth):
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0
            nodeDepth += 1
        if nodeDepth == self.depth :
            return self.evaluationFunction(gameState)
        if agentIndex == self.index :
            return self.max_value(gameState, agentIndex, nodeDepth)

        else:
            return self.exp_value(gameState, agentIndex, nodeDepth)

        return 'None'

    def max_value(self, gameState, agentIndex, nodeDepth):
      if gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)

      value = float("-inf")
      actionValue = "Stop"

      for legalActions in gameState.getLegalActions(agentIndex) :
        if legalActions == Directions.STOP:
          continue
        successor = gameState.generateSuccessor(agentIndex, legalActions)
        temp = self.expectimax_value(successor, agentIndex+1, nodeDepth)
        if temp > value:
          value = temp
          actionValue = legalActions

      if nodeDepth == 0 :
        return actionValue
      else:
        return value

    def exp_value(self, gameState, agentIndex, nodeDepth):
      if gameState.isWin() or gameState.isLose() :
          return self.evaluationFunction(gameState)
      value = 0
      pr = 1.0/len(gameState.getLegalActions(agentIndex))

      for legalActions in gameState.getLegalActions(agentIndex):
        if legalActions == Directions.STOP:
            continue
        successor = gameState.generateSuccessor(agentIndex, legalActions)
        value = value + (self.expectimax_value(successor, agentIndex+1, nodeDepth) * pr)
      return value

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.expectimax_value(gameState,0,0)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 3). 

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newGhostStates = currentGameState.getGhostStates()
    newCapsules = currentGameState.getCapsules()
    Score = currentGameState.getScore()
    food = 10 / (currentGameState.getNumFood() + 10)
    ghost_distance = float("inf")
    for ghost in newGhostStates:
        ghostPos = ghost.getPosition()
        if newPos == ghostPos:
            return float("-inf")
        else:
            ghost_distance = min(ghost_distance, manhattanDistance(newPos, ghostPos))
    ghost_distance = 10 / (10 + (ghost_distance / (len(newGhostStates))))
    cap_distance = float("inf")
    for capsule in newCapsules:
        cap_distance = min(cap_distance, manhattanDistance(newPos, capsule))
    cap_distance = 10 / (10 + len(newCapsules))
    value = Score + food + ghost_distance + cap_distance
    return value
# Abbreviation
better = betterEvaluationFunction

