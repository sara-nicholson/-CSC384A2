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
	#find the remaining food on the board
	oldFood = currentGameState.getFood()
	newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
	score = 0
	x = newPos[0]
	y = newPos[1]
	for i in range(len(newGhostStates)):
	    ghostPos = newGhostStates[i].getPosition()
	    closeGhost = abs(ghostPos[0] - x) + abs(ghostPos[1]-y)
	    if closeGhost <= 1 and newScaredTimes[i] == 0:
		return score
	    if closeGhost <= 1 and newScaredTimes[i] > 0:
		score += 1
	    if ghostPos != newPos:
		score += 1
	minDist = float('inf')
 	#calc dist to nearest food
	for i in range(newFood.width):
	    for j in range(newFood.height):
		if oldFood[i][j] == True:
		    dist = abs(i-x) + abs(j-y)
		    if dist < minDist:
		        minDist = dist
	if minDist != float('inf'):
	    score += (newFood.width*newFood.height) -minDist
        return score

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

    def DFminimax(self,gameState, agentIndex, leftToPlay):
	"""
	Deterrmine the moves for pacman and ghosts based on minimax
	gameState = gameState to be examined
	agentIndex = index of agent that is moving
	leftTOPlay = number of moves left to examine with the given depth 
	"""
	numOfAgents = gameState.getNumAgents()
	# if a terminal node, return best_move as None and the evaluation from the evaluation function
	best_move = None
	actions = gameState.getLegalActions(agentIndex)
	if leftToPlay == 0 or len(actions) == 0:
	    value = self.evaluationFunction(gameState)
	    return best_move, value
	if agentIndex == 0:
	    value = float('-inf')
	if agentIndex != 0:
	    value = float('inf')
	for i in range(len(actions)):
	    nextGameState = gameState.generateSuccessor(agentIndex,actions[i])
	    nextAgent = (agentIndex + 1) % numOfAgents
	    leftToPlay -= 1 
	    next_move, next_value = self.DFminimax(nextGameState,nextAgent,leftToPlay)
	    leftToPlay += 1
	    if agentIndex == 0 and value < next_value:
		value,best_move = next_value, actions[i]
	    if agentIndex != 0 and value > next_value:
		value,best_move = next_value, actions[i]

	return best_move, value

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
	numOfAgents = gameState.getNumAgents()
        leftToPlay = (self.depth)*numOfAgents
	best_move, score = self.DFminimax(gameState,0,leftToPlay)
	return best_move

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
	numOfAgents = gameState.getNumAgents()
	leftToPlay = (self.depth)*numOfAgents
	alpha = float('-inf')
	beta = float('inf')
	best_move, score = self.DFminimaxAB(gameState,0, leftToPlay,alpha,beta)
	return best_move

    def DFminimaxAB(self, gameState, agentIndex, leftToPlay, alpha, beta):
	"""
	Determine minimax with alpha beta pruning implimented. Returns the best move available and the score of it
	gameState = the state of the game(state)
	agentIndex = current agent, 0 = pacman ghost>0 (int)
	leftToPlay = the number of moves left based on depth being examined (int)
	alpha = alpha value for pruning (int)
	beta = beta value for pruning (int)
	"""
	best_move = None
	actions = gameState.getLegalActions(agentIndex)
	if leftToPlay == 0 or len(actions) == 0:
	    value = self.evaluationFunction(gameState)
	    return best_move, value
	if agentIndex == 0:
	    value  = float('-inf')
	if agentIndex != 0:
	    value = float('inf')
	numOfAgents=gameState.getNumAgents()
	for i in range(len(actions)):
	    next_pos = gameState.generateSuccessor(agentIndex,actions[i])
	    next_agent = (agentIndex + 1) % numOfAgents
	    leftToPlay -= 1
	    next_move, next_value = self.DFminimaxAB(next_pos, next_agent, leftToPlay, alpha,beta)
	    leftToPlay += 1
	    if agentIndex == 0:
	        if next_value > value:
		    value,best_move = next_value, actions[i]
	        if value >= beta:
		    return best_move, value
	        alpha = max(alpha,value)
	    if agentIndex != 0:
	        if next_value < value:
		    value, best_move = next_value, actions[i]	
	        if value <= alpha:
		    return best_move,value
	        beta = min(value,beta)
	return best_move, value
	
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        numOfAgents = gameState.getNumAgents()
	leftToPlay = (self.depth)*numOfAgents
	move, value = self.Expectimax(gameState, leftToPlay, 0)
	return move
	
    def Expectimax(self, gameState, leftToPlay, agentIndex):
	"""
	Determine the expectimax search with the ghosts probability being a uniform 
	distribution
	gameState: state of the game
	leftToPlay: total moves left for the given depth int
	agentIndex: which agent is playing 0 = pacman else = ghosts
	return the best move to take and its value
	"""
	
	actions = gameState.getLegalActions(agentIndex)
	best_move = None
	if leftToPlay == 0 or len(actions) == 0:
	    value = self.evaluationFunction(gameState)
	    return best_move, float(value)
	
	if agentIndex == 0:
	    value = float('-inf')
	if agentIndex!= 0:
	    value = float(0)

	numOfAgents = gameState.getNumAgents()
	for  i in range(len(actions)):
	    nextGameState = gameState.generateSuccessor(agentIndex, actions[i])
	    leftToPlay -= 1
	    nextAgent = (agentIndex + 1) % numOfAgents
	    move, next_value = self.Expectimax(nextGameState, leftToPlay, nextAgent)
	    leftToPlay += 1
	    if agentIndex == 0 and value < next_value:
		value, best_move = next_value, actions[i]
	    if agentIndex != 0:
  		value += (float(next_value) * float(1/float(len(actions))))

	return best_move, value
	    
	


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    gameScore = currentGameState.getScore() 
    score = 0.0
    if currentGameState.isWin():
	gameScore = currentGameState.getScore()
	score = 1000000
	return score + gameScore
    location = currentGameState.getPacmanPosition()
    x = location[0]
    y = location[1]
    Food = currentGameState.getFood()
    gameSize = Food.height+ Food.width
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    for i in range(len(GhostStates)):
 	ghostPos = GhostStates[i].getPosition()
	closeGhost = abs(ghostPos[0] - x) + abs(ghostPos[1]-y)
 	if closeGhost <= 1 and ScaredTimes[i] == 0:
	    return 0
	if closeGhost <=1 and ScaredTimes[i] > 0:
	    score +=10 
    totalFood = currentGameState.getNumFood() 
    minDist = float('inf')
    #calc dist to nearest food
    for i in range(Food.width):
	for j in range(Food.height):
	    if Food[i][j] == True:
		dist = abs(i-x) + abs(j-y)
		if dist < minDist:
		    minDist = dist
   
    score =3.0*(1.0/float(minDist))+  100.0*(1.0/float(totalFood)) + 25.0*float(gameScore)
    return float(score)


# Abbreviation
better = betterEvaluationFunction

