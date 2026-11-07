package agents

import (
	"math"
	"math/rand"
	"sync"

	"github.com/pbures/3d3t-ai-ml-learning/internal/engine"
)

// MCTSAgent implements Monte Carlo Tree Search with Upper Confidence Bound for Trees (UCT).
type MCTSAgent struct {
	Simulations      int
	ExplorationConst float64
	mu               sync.Mutex
	rng              *rand.Rand
}

type mctsNode struct {
	parent        *mctsNode
	actionFromPar uint8
	playerJustMoved engine.Player
	state         *engine.Game
	untriedActions []uint8
	children      []*mctsNode
	visits        int
	wins          float64 // Cumulative reward from the perspective of playerJustMoved
}

// NewMCTSAgent creates an MCTSAgent with the specified number of simulations per move and RNG seed.
func NewMCTSAgent(simulations int, seed int64) *MCTSAgent {
	if simulations <= 0 {
		simulations = 250
	}
	return &MCTSAgent{
		Simulations:      simulations,
		ExplorationConst: math.Sqrt2,
		rng:              rand.New(rand.NewSource(seed)),
	}
}

func (a *MCTSAgent) Name() string {
	return "mcts"
}

func newMCTSNode(state *engine.Game, parent *mctsNode, action uint8, playerJustMoved engine.Player) *mctsNode {
	return &mctsNode{
		parent:          parent,
		actionFromPar:   action,
		playerJustMoved: playerJustMoved,
		state:           state,
		untriedActions:  ValidActions(state),
	}
}

func (n *mctsNode) selectUCTChild(c float64) *mctsNode {
	var best *mctsNode
	bestScore := -math.MaxFloat64
	logParent := math.Log(float64(n.visits))

	for _, ch := range n.children {
		exploit := ch.wins / float64(ch.visits)
		explore := c * math.Sqrt(logParent/float64(ch.visits))
		uct := exploit + explore
		if uct > bestScore {
			bestScore = uct
			best = ch
		}
	}
	return best
}

// SelectAction runs MCTS from `g` and returns the most visited action (or immediate tactical win/block).
func (a *MCTSAgent) SelectAction(g *engine.Game) (uint8, error) {
	bestAction, _, _, err := a.SearchWithPolicy(g)
	return bestAction, err
}

// SearchWithPolicy runs MCTS simulations and returns:
//   - bestAction: the selected column action in {0..15}
//   - policy: normalized visit-count distribution across all 16 actions (used for RL policy targets)
//   - rootValue: estimated value in [-1, +1] from the perspective of g.CurrentPlayer
func (a *MCTSAgent) SearchWithPolicy(g *engine.Game) (uint8, [engine.MaxActions]float32, float32, error) {
	var policy [engine.MaxActions]float32
	valid := ValidActions(g)
	if len(valid) == 0 {
		return 0, policy, 0, ErrNoValidMoves
	}

	me := g.CurrentPlayer
	opp := me.Opponent()

	// 1. Immediate 1-ply win short-circuit
	if myWins := FindImmediateWinningActions(g, me); len(myWins) > 0 {
		winAct := myWins[0]
		policy[winAct] = 1.0
		return winAct, policy, 1.0, nil
	}

	// 2. Immediate 1-ply forced block short-circuit
	if oppWins := FindImmediateWinningActions(g, opp); len(oppWins) > 0 {
		blockAct := oppWins[0]
		for _, cand := range oppWins {
			if !GivesOpponentImmediateWin(g, cand) {
				blockAct = cand
				break
			}
		}
		policy[blockAct] = 1.0
		return blockAct, policy, 0.0, nil
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	root := newMCTSNode(g.Clone(), nil, 0, g.CurrentPlayer.Opponent())
	// Filter poison moves at root if safe alternatives exist
	safeRoot := make([]uint8, 0, len(root.untriedActions))
	for _, act := range root.untriedActions {
		if !GivesOpponentImmediateWin(g, act) {
			safeRoot = append(safeRoot, act)
		}
	}
	if len(safeRoot) > 0 {
		root.untriedActions = safeRoot
	}

	for sim := 0; sim < a.Simulations; sim++ {
		node := root

		// 1. Selection
		for len(node.untriedActions) == 0 && len(node.children) > 0 && node.state.Status == engine.StatusOngoing {
			node = node.selectUCTChild(a.ExplorationConst)
		}

		// 2. Expansion
		if len(node.untriedActions) > 0 && node.state.Status == engine.StatusOngoing {
			idx := a.rng.Intn(len(node.untriedActions))
			act := node.untriedActions[idx]
			// Remove act from untriedActions
			node.untriedActions[idx] = node.untriedActions[len(node.untriedActions)-1]
			node.untriedActions = node.untriedActions[:len(node.untriedActions)-1]

			mover := node.state.CurrentPlayer
			nextState := node.state.Clone()
			_, _ = nextState.ApplyMove(act)

			child := newMCTSNode(nextState, node, act, mover)
			node.children = append(node.children, child)
			node = child
		}

		// 3. Simulation (Rollout with 1-ply win/block awareness)
		winner := a.rollout(node.state)

		// 4. Backpropagation
		for cur := node; cur != nil; cur = cur.parent {
			cur.visits++
			switch winner {
			case engine.PlayerNone:
				cur.wins += 0.0
			case cur.playerJustMoved:
				cur.wins += 1.0
			default:
				cur.wins -= 1.0
			}
		}
	}

	// Compute normalized visit policy and pick highest-visit child
	totalVisits := 0
	var bestChild *mctsNode
	for _, ch := range root.children {
		totalVisits += ch.visits
		if bestChild == nil || ch.visits > bestChild.visits ||
			(ch.visits == bestChild.visits && ch.wins > bestChild.wins) {
			bestChild = ch
		}
	}

	if totalVisits > 0 {
		for _, ch := range root.children {
			policy[ch.actionFromPar] = float32(ch.visits) / float32(totalVisits)
		}
	}

	bestAction := valid[0]
	var rootValue float32
	if bestChild != nil {
		bestAction = bestChild.actionFromPar
		rootValue = float32(bestChild.wins / float64(bestChild.visits))
	}

	return bestAction, policy, rootValue, nil
}

func (a *MCTSAgent) rollout(startState *engine.Game) engine.Player {
	simGame := startState.Clone()
	for simGame.Status == engine.StatusOngoing {
		// Check if active player has an immediate 1-ply win
		if wins := FindImmediateWinningActions(simGame, simGame.CurrentPlayer); len(wins) > 0 {
			_, _ = simGame.ApplyMove(wins[0])
			break
		}
		// Check if active player must block an opponent 1-ply win
		if oppWins := FindImmediateWinningActions(simGame, simGame.CurrentPlayer.Opponent()); len(oppWins) > 0 {
			_, _ = simGame.ApplyMove(oppWins[0])
			continue
		}
		valid := ValidActions(simGame)
		act := valid[a.rng.Intn(len(valid))]
		_, _ = simGame.ApplyMove(act)
	}
	return simGame.Winner
}
