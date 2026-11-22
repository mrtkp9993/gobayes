package distributions

import (
	"math"

	"github.com/mrtkp9993/gobayes/internal/utils"
	"github.com/mrtkp9993/gobayes/random"
)

type Normal struct {
	Mu, Sigma float64
}

func (Normal) IsContinuous() {}

func (n Normal) LogProb(x float64) float64 {
	if n.Sigma <= 0 {
		return math.Inf(-1)
	}
	delta := x - n.Mu
	z := delta / n.Sigma
	if math.IsInf(delta, 0) && !math.IsInf(x, 0) && !math.IsInf(n.Mu, 0) {
		z = x/n.Sigma - n.Mu/n.Sigma
	}
	return -utils.LogPositive(n.Sigma) - 0.5*math.Log(2*math.Pi) - 0.5*z*z
}

func (n Normal) Sample() float64 {
	if n.Sigma <= 0 {
		panic("sigma must be positive")
	}
	return n.Mu + n.Sigma*random.NormFloat64()
}
