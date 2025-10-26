package distributions

import (
	"math"
	"math/rand/v2"
)

type Normal struct {
	Mu, Sigma float64
}

func (n Normal) LogProb(x float64) float64 {
	variance := n.Sigma * n.Sigma
	return -0.5*math.Log(2*math.Pi*variance) - math.Pow(x-n.Mu, 2)/(2*variance)
}

func (n Normal) Sample() float64 {
	return n.Mu + n.Sigma*rand.NormFloat64()
}
