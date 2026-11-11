package agents

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"math"
	"math/bits"
	"sync"

	"github.com/pbures/3d3t-ai-ml-learning/internal/engine"
)

const stemChannels = 11

//go:embed weights/ttt3d_latest.bin
var embeddedTTT3DWeights []byte

type resBlockWeights struct {
	w1, b1           []float32
	wGlobal, bGlobal []float32
	w2, b2           []float32
}

// TTT3DModel holds the BatchNorm-fused 3D ResNet weights exported from PyTorch `TTT3DNet`.
type TTT3DModel struct {
	Channels   int
	NumBlocks  int
	stemW      []float32
	stemB      []float32
	blocks     []resBlockWeights
	policyConvW []float32
	policyConvB []float32
	policyFCW  []float32
	policyFCB  []float32
	valueConvW []float32
	valueConvB []float32
	valueFC1W  []float32
	valueFC1B  []float32
	valueFC2W  []float32
	valueFC2B  []float32
}

var (
	defaultModelOnce sync.Once
	defaultModel     *TTT3DModel
	defaultModelErr  error
)

// DefaultTTT3DModel returns the singleton embedded PyTorch TTT3DNet model.
func DefaultTTT3DModel() (*TTT3DModel, error) {
	defaultModelOnce.Do(func() {
		defaultModel, defaultModelErr = ParseFusedWeights(embeddedTTT3DWeights)
	})
	return defaultModel, defaultModelErr
}

// ParseFusedWeights decodes a binary weights blob produced by `TTT3DNet.export_fused_binary`.
func ParseFusedWeights(data []byte) (*TTT3DModel, error) {
	if len(data) < 16 || string(data[:4]) != "3T3D" {
		return nil, fmt.Errorf("invalid TTT3D weights header")
	}
	version := binary.LittleEndian.Uint32(data[4:8])
	if version != 1 {
		return nil, fmt.Errorf("unsupported TTT3D weights version %d", version)
	}
	c := int(binary.LittleEndian.Uint32(data[8:12]))
	numBlocks := int(binary.LittleEndian.Uint32(data[12:16]))
	if c <= 0 || numBlocks <= 0 {
		return nil, fmt.Errorf("invalid model dimensions: C=%d, B=%d", c, numBlocks)
	}

	offset := 16
	readFloats := func(count int) ([]float32, error) {
		bytesNeeded := count * 4
		if offset+bytesNeeded > len(data) {
			return nil, fmt.Errorf("truncated weights buffer at offset %d (need %d bytes)", offset, bytesNeeded)
		}
		out := make([]float32, count)
		for i := 0; i < count; i++ {
			bits := binary.LittleEndian.Uint32(data[offset+i*4 : offset+(i+1)*4])
			out[i] = math.Float32frombits(bits)
		}
		offset += bytesNeeded
		return out, nil
	}

	m := &TTT3DModel{
		Channels:  c,
		NumBlocks: numBlocks,
		blocks:    make([]resBlockWeights, numBlocks),
	}
	var err error
	if m.stemW, err = readFloats(c * stemChannels * 27); err != nil {
		return nil, err
	}
	if m.stemB, err = readFloats(c); err != nil {
		return nil, err
	}
	for b := 0; b < numBlocks; b++ {
		if m.blocks[b].w1, err = readFloats(c * c * 27); err != nil {
			return nil, err
		}
		if m.blocks[b].b1, err = readFloats(c); err != nil {
			return nil, err
		}
		if m.blocks[b].wGlobal, err = readFloats(c * c * 64); err != nil {
			return nil, err
		}
		if m.blocks[b].bGlobal, err = readFloats(c); err != nil {
			return nil, err
		}
		if m.blocks[b].w2, err = readFloats(c * c * 27); err != nil {
			return nil, err
		}
		if m.blocks[b].b2, err = readFloats(c); err != nil {
			return nil, err
		}
	}
	if m.policyConvW, err = readFloats(16 * c); err != nil {
		return nil, err
	}
	if m.policyConvB, err = readFloats(16); err != nil {
		return nil, err
	}
	if m.policyFCW, err = readFloats(16 * 1024); err != nil {
		return nil, err
	}
	if m.policyFCB, err = readFloats(16); err != nil {
		return nil, err
	}
	if m.valueConvW, err = readFloats(8 * c); err != nil {
		return nil, err
	}
	if m.valueConvB, err = readFloats(8); err != nil {
		return nil, err
	}
	if m.valueFC1W, err = readFloats(64 * 512); err != nil {
		return nil, err
	}
	if m.valueFC1B, err = readFloats(64); err != nil {
		return nil, err
	}
	if m.valueFC2W, err = readFloats(1 * 64); err != nil {
		return nil, err
	}
	if m.valueFC2B, err = readFloats(1); err != nil {
		return nil, err
	}
	return m, nil
}

type conv3x3Tap struct {
	inCell int
	kOff   int
}

var conv3x3Taps [64][]conv3x3Tap

func init() {
	for z := 0; z < 4; z++ {
		for y := 0; y < 4; y++ {
			for x := 0; x < 4; x++ {
				cell := 16*z + 4*y + x
				taps := make([]conv3x3Tap, 0, 27)
				for kz := 0; kz < 3; kz++ {
					iz := z + kz - 1
					if iz < 0 || iz >= 4 {
						continue
					}
					for ky := 0; ky < 3; ky++ {
						iy := y + ky - 1
						if iy < 0 || iy >= 4 {
							continue
						}
						for kx := 0; kx < 3; kx++ {
							ix := x + kx - 1
							if ix < 0 || ix >= 4 {
								continue
							}
							taps = append(taps, conv3x3Tap{
								inCell: 16*iz + 4*iy + ix,
								kOff:   9*kz + 3*ky + kx,
							})
						}
					}
				}
				conv3x3Taps[cell] = taps
			}
		}
	}
}

func conv3d3x3Pad1(in []float32, inC, outC int, w, b []float32) []float32 {
	out := make([]float32, outC*64)
	for oc := 0; oc < outC; oc++ {
		bias := b[oc]
		ocBase := oc * 64
		for cell := 0; cell < 64; cell++ {
			out[ocBase+cell] = bias
		}
		wOcBase := oc * inC * 27
		for ic := 0; ic < inC; ic++ {
			inIcBase := ic * 64
			wIcBase := wOcBase + ic*27
			for cell := 0; cell < 64; cell++ {
				sum := out[ocBase+cell]
				for _, tap := range conv3x3Taps[cell] {
					sum += in[inIcBase+tap.inCell] * w[wIcBase+tap.kOff]
				}
				out[ocBase+cell] = sum
			}
		}
	}
	return out
}

func conv3d1x1(in []float32, inC, outC int, w, b []float32) []float32 {
	out := make([]float32, outC*64)
	for oc := 0; oc < outC; oc++ {
		bias := b[oc]
		ocBase := oc * 64
		wOcBase := oc * inC
		for cell := 0; cell < 64; cell++ {
			sum := bias
			for ic := 0; ic < inC; ic++ {
				sum += in[ic*64+cell] * w[wOcBase+ic]
			}
			out[ocBase+cell] = sum
		}
	}
	return out
}

func reluInPlace(x []float32) {
	for i, v := range x {
		if v < 0 {
			x[i] = 0
		}
	}
}

func linearLayer(in []float32, inDim, outDim int, w, b []float32) []float32 {
	out := make([]float32, outDim)
	for o := 0; o < outDim; o++ {
		sum := b[o]
		wBase := o * inDim
		for i := 0; i < inDim; i++ {
			sum += in[i] * w[wBase+i]
		}
		out[o] = sum
	}
	return out
}

func extractGeometricFeatures(input [engine.CanonicalTensorSize]float32) [stemChannels * 64]float32 {
	var out [stemChannels * 64]float32
	copy(out[:engine.CanonicalTensorSize], input[:])

	var myMask, oppMask, surfMask uint64
	for bit := uint8(0); bit < 64; bit++ {
		if input[bit] > 0.5 {
			myMask |= uint64(1) << bit
		}
		if input[64+int(bit)] > 0.5 {
			oppMask |= uint64(1) << bit
		}
		if input[128+int(bit)] > 0.5 {
			surfMask |= uint64(1) << bit
		}
	}
	occMask := myMask | oppMask

	var cellMy3, cellOpp3, cellMy2, cellOpp2 [64]float32
	for _, mask := range engine.WinMasks {
		myBits := myMask & mask
		oppBits := oppMask & mask
		emptyOnLine := mask &^ occMask

		if oppBits == 0 {
			switch bits.OnesCount64(myBits) {
			case 3:
				for rem := emptyOnLine; rem != 0; rem &= rem - 1 {
					b := bits.TrailingZeros64(rem)
					cellMy3[b] += 1.0
				}
			case 2:
				for rem := emptyOnLine; rem != 0; rem &= rem - 1 {
					b := bits.TrailingZeros64(rem)
					cellMy2[b] += 1.0
				}
			}
		}
		if myBits == 0 {
			switch bits.OnesCount64(oppBits) {
			case 3:
				for rem := emptyOnLine; rem != 0; rem &= rem - 1 {
					b := bits.TrailingZeros64(rem)
					cellOpp3[b] += 1.0
				}
			case 2:
				for rem := emptyOnLine; rem != 0; rem &= rem - 1 {
					b := bits.TrailingZeros64(rem)
					cellOpp2[b] += 1.0
				}
			}
		}
	}

	for b := 0; b < 64; b++ {
		isSurf := (surfMask & (uint64(1) << b)) != 0
		if isSurf && cellMy3[b] > 0.5 {
			out[3*64+b] = 1.0
		}
		if isSurf && cellOpp3[b] > 0.5 {
			out[4*64+b] = 1.0
		}
		if b < 48 && isSurf && cellOpp3[b+16] > 0.5 {
			out[5*64+b] = 1.0
		}
		if b < 48 && isSurf && cellMy3[b+16] > 0.5 {
			out[6*64+b] = 1.0
		}
		m3 := cellMy3[b]
		if m3 > 2.0 {
			m3 = 2.0
		}
		out[7*64+b] = m3 * 0.5

		o3 := cellOpp3[b]
		if o3 > 2.0 {
			o3 = 2.0
		}
		out[8*64+b] = o3 * 0.5

		m2 := cellMy2[b]
		if m2 > 4.0 {
			m2 = 4.0
		}
		out[9*64+b] = m2 * 0.25

		o2 := cellOpp2[b]
		if o2 > 4.0 {
			o2 = 4.0
		}
		out[10*64+b] = o2 * 0.25
	}
	return out
}

func addConv3d4x4GlobalInPlace(dst []float32, in []float32, c int, wGlobal, bGlobal []float32) {
	inLen := c * 64
	for oc := 0; oc < c; oc++ {
		sum := bGlobal[oc]
		wBase := oc * inLen
		for i := 0; i < inLen; i++ {
			sum += in[i] * wGlobal[wBase+i]
		}
		ocBase := oc * 64
		for cell := 0; cell < 64; cell++ {
			dst[ocBase+cell] += sum
		}
	}
}

// Forward executes the exact 3D ResNet forward pass on a 192-element canonical float32 tensor.
func (m *TTT3DModel) Forward(input [engine.CanonicalTensorSize]float32) (logits [16]float32, value float32) {
	c := m.Channels
	stemIn := extractGeometricFeatures(input)
	h := conv3d3x3Pad1(stemIn[:], stemChannels, c, m.stemW, m.stemB)
	reluInPlace(h)

	for _, blk := range m.blocks {
		out1 := conv3d3x3Pad1(h, c, c, blk.w1, blk.b1)
		addConv3d4x4GlobalInPlace(out1, h, c, blk.wGlobal, blk.bGlobal)
		reluInPlace(out1)
		out2 := conv3d3x3Pad1(out1, c, c, blk.w2, blk.b2)
		for i := range out2 {
			v := out2[i] + h[i]
			if v < 0 {
				v = 0
			}
			h[i] = v
		}
	}

	// Policy Head
	p := conv3d1x1(h, c, 16, m.policyConvW, m.policyConvB)
	reluInPlace(p)
	pLogits := linearLayer(p, 1024, 16, m.policyFCW, m.policyFCB)
	copy(logits[:], pLogits)

	// Value Head
	v := conv3d1x1(h, c, 8, m.valueConvW, m.valueConvB)
	reluInPlace(v)
	vHidden := linearLayer(v, 512, 64, m.valueFC1W, m.valueFC1B)
	reluInPlace(vHidden)
	vOut := linearLayer(vHidden, 64, 1, m.valueFC2W, m.valueFC2B)
	value = float32(math.Tanh(float64(vOut[0])))
	return logits, value
}

// PredictGame runs `Forward` on `g.CanonicalTensor()` and returns legal-move-masked softmax probabilities and value.
func (m *TTT3DModel) PredictGame(g *engine.Game) (probs [16]float32, value float32) {
	rawBytes := g.CanonicalTensor()
	var input [engine.CanonicalTensorSize]float32
	for i, b := range rawBytes {
		input[i] = float32(b)
	}
	logits, val := m.Forward(input)
	mask := g.ValidMovesMask()

	maxLogit := float32(-1e30)
	for a := uint8(0); a < engine.MaxActions; a++ {
		if (mask & (uint16(1) << a)) != 0 {
			if logits[a] > maxLogit {
				maxLogit = logits[a]
			}
		}
	}

	var sumExp float64
	for a := uint8(0); a < engine.MaxActions; a++ {
		if (mask & (uint16(1) << a)) != 0 {
			e := math.Exp(float64(logits[a] - maxLogit))
			probs[a] = float32(e)
			sumExp += e
		}
	}
	if sumExp > 0 {
		for a := uint8(0); a < engine.MaxActions; a++ {
			probs[a] = float32(float64(probs[a]) / sumExp)
		}
	}
	return probs, val
}

// TTT3DNetAgent wraps the trained PyTorch `TTT3DNet` model in either Pure Neural mode
// (`UseTacticalLookahead: false`) or Hybrid `NeuralArenaBot` mode (`UseTacticalLookahead: true`).
type TTT3DNetAgent struct {
	UseTacticalLookahead bool
	model                *TTT3DModel
}

// NewTTT3DNetAgent creates a TTT3DNetAgent using the embedded trained PyTorch checkpoint.
func NewTTT3DNetAgent(useTacticalLookahead bool) *TTT3DNetAgent {
	m, _ := DefaultTTT3DModel()
	return &TTT3DNetAgent{
		UseTacticalLookahead: useTacticalLookahead,
		model:                m,
	}
}

func (a *TTT3DNetAgent) Name() string {
	if a.UseTacticalLookahead {
		return "ttt3dnet"
	}
	return "ttt3dnet-pure"
}

func hasUnstoppableThreat(g *engine.Game, attacker engine.Player) bool {
	if g == nil || g.Status != engine.StatusOngoing {
		return false
	}
	defender := attacker.Opponent()
	myThreats := ThreatCellsMask(g, attacker)
	if myThreats == 0 {
		return false
	}
	oppThreats := ThreatCellsMask(g, defender)
	surface := LandingMask(g)
	if bits.OnesCount64(myThreats&surface) >= 2 {
		return true
	}
	// Any two vertically consecutive 3-in-a-row threats (z and z+1 in the same column, h <= z)
	// where the lower cell does not also complete a win for the defender is an unstoppable vertical fork.
	if (myThreats & (myThreats >> 16) &^ oppThreats) != 0 {
		return true
	}
	return false
}

// SelectActionWithPolicy returns the chosen action, 16-element policy distribution, and state value.
func (a *TTT3DNetAgent) SelectActionWithPolicy(g *engine.Game) (uint8, [16]float32, float32, error) {
	valid := ValidActions(g)
	if len(valid) == 0 {
		return 0, [16]float32{}, 0, ErrNoValidMoves
	}
	probs, val := a.model.PredictGame(g)

	// Best pure neural action by probability
	bestPure := valid[0]
	for _, act := range valid[1:] {
		if probs[act] > probs[bestPure] {
			bestPure = act
		}
	}

	if !a.UseTacticalLookahead {
		return bestPure, probs, val, nil
	}

	me := g.CurrentPlayer
	opp := me.Opponent()

	// 1. Immediate 1-ply winning moves ranked by TTT3DNet policy
	if myWins := FindImmediateWinningActions(g, me); len(myWins) > 0 {
		bestWin := myWins[0]
		for _, w := range myWins[1:] {
			if probs[w] > probs[bestWin] {
				bestWin = w
			}
		}
		return bestWin, probs, 1.0, nil
	}

	// 2. Forced 1-ply block if opponent threatens an immediate win
	if oppWins := FindImmediateWinningActions(g, opp); len(oppWins) > 0 {
		bestBlock := oppWins[0]
		bestSafeFound := false
		for _, blk := range oppWins {
			safe := !GivesOpponentImmediateWin(g, blk)
			if safe && (!bestSafeFound || probs[blk] > probs[bestBlock]) {
				bestBlock = blk
				bestSafeFound = true
			} else if !bestSafeFound && probs[blk] > probs[bestBlock] {
				bestBlock = blk
			}
		}
		return bestBlock, probs, val, nil
	}

	// 3. Filter out blunders that hand the opponent an immediate 1-ply win
	candidates := make([]uint8, 0, len(valid))
	for _, act := range valid {
		if !GivesOpponentImmediateWin(g, act) {
			candidates = append(candidates, act)
		}
	}
	if len(candidates) == 0 {
		candidates = valid
	}

	// 4. Check for offensive double-threat or stacked forks
	for _, act := range candidates {
		clone := g.Clone()
		_, _ = clone.ApplyMove(act)
		if hasUnstoppableThreat(clone, me) {
			return act, probs, val, nil
		}
	}

	// 5. Filter out moves allowing the opponent to play a reply creating a double-threat or stacked fork
	nonForkVulnerable := make([]uint8, 0, len(candidates))
	for _, act := range candidates {
		clone := g.Clone()
		_, _ = clone.ApplyMove(act)
		myThreats := FindImmediateWinningActions(clone, me)
		oppReplies := myThreats
		if len(oppReplies) == 0 {
			oppReplies = ValidActions(clone)
		}
		oppCanFork := false
		for _, r := range oppReplies {
			afterReply := clone.Clone()
			_, _ = afterReply.ApplyMove(r)
			if afterReply.Status != engine.StatusOngoing {
				oppCanFork = true
				break
			}
			if hasUnstoppableThreat(afterReply, opp) && len(FindImmediateWinningActions(afterReply, me)) == 0 {
				oppCanFork = true
				break
			}
		}
		if !oppCanFork {
			nonForkVulnerable = append(nonForkVulnerable, act)
		}
	}
	if len(nonForkVulnerable) > 0 {
		candidates = nonForkVulnerable
	}

	// 6. Hybrid Neural Policy + 1-Ply Value + 3-Ply Bitboard Tactical Search over candidate moves
	ab := AlphaBetaAgent{MaxDepth: 3}
	bestCandidate := candidates[0]
	bestCombined := float32(-1e9)
	for _, act := range candidates {
		clone := g.Clone()
		_, _ = clone.ApplyMove(act)
		_, oppVal := a.model.PredictGame(clone)
		myNextVal := -oppVal
		abVal := -ab.negamax(clone, 2, -1_000_000, 1_000_000)
		threatBonus := float32(0.15) * float32(len(FindImmediateWinningActions(clone, me)))
		combined := probs[act] + 0.5*myNextVal + threatBonus + 0.008*float32(abVal)
		if combined > bestCombined {
			bestCombined = combined
			bestCandidate = act
		}
	}

	return bestCandidate, probs, val, nil
}

func (a *TTT3DNetAgent) SelectAction(g *engine.Game) (uint8, error) {
	act, _, _, err := a.SelectActionWithPolicy(g)
	return act, err
}
