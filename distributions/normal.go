package distributions

import (
	"math"

	"github.com/mrtkp9993/gobayes/random"
)

type Normal struct {
	Mu, Sigma float64
}

func (n Normal) LogProb(x float64) float64 {
	if n.Sigma <= 0 {
		return math.Inf(-1)
	}
	variance := n.Sigma * n.Sigma
	return -0.5*math.Log(2*math.Pi*variance) - math.Pow(x-n.Mu, 2)/(2*variance)
}

func (n Normal) Sample() float64 {
	if n.Sigma <= 0 {
		panic("sigma must be positive")
	}
	return n.Mu + n.Sigma*random.NormFloat64()
}
