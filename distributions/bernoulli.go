package distributions

import (
	"math"

	"github.com/mrtkp9993/gobayes/random"
)

type Bernoulli struct {
	P float64
}

func (Bernoulli) IsDiscrete() {}

func (b Bernoulli) valid() bool {
	return b.P >= 0 && b.P <= 1
}

func (b Bernoulli) LogProb(x float64) float64 {
	if !b.valid() {
		return math.Inf(-1)
	}
	switch x {
	case 0:
		return math.Log1p(-b.P)
	case 1:
		return math.Log(b.P)
	default:
		return math.Inf(-1)
	}
}

func (b Bernoulli) Sample() float64 {
	if !b.valid() {
		panic("bernoulli requires 0 <= P <= 1")
	}
	if random.Float64() < b.P {
		return 1
	}
	return 0
}
