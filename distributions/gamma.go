package distributions

import (
	"math"

	"github.com/mrtkp9993/gobayes/random"
)

type Gamma struct {
	Alpha float64
	Beta  float64
}

func (Gamma) IsContinuous() {}

func (g Gamma) LogProb(x float64) float64 {
	if !(g.Alpha > 0) || math.IsInf(g.Alpha, 0) || !(g.Beta > 0) || math.IsInf(g.Beta, 0) || !(x >= 0) || math.IsInf(x, 1) {
		return math.Inf(-1)
	}
	if x == 0 {
		switch {
		case g.Alpha < 1:
			return math.Inf(1)
		case g.Alpha == 1:
			return math.Log(g.Beta)
		default:
			return math.Inf(-1)
		}
	}

	lgamma, _ := math.Lgamma(g.Alpha)
	return (g.Alpha-1)*math.Log(x) - g.Beta*x + g.Alpha*math.Log(g.Beta) - lgamma
}

func (g Gamma) Sample() float64 {
	alpha := g.Alpha
	beta := g.Beta
	if !(alpha > 0) || math.IsInf(alpha, 0) || !(beta > 0) || math.IsInf(beta, 0) {
		panic("gamma parameters must be finite and positive")
	}
	if alpha < 1 {
		x := Gamma{Alpha: alpha + 1, Beta: beta}.Sample()
		u := random.Float64()
		return x * math.Pow(u, 1/alpha)
	}

	d := alpha - 1.0/3.0
	c := 1.0 / math.Sqrt(9.0*d)
	for {
		x := random.NormFloat64()
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := random.Float64()

		if u < 1-0.0331*math.Pow(x, 4) {
			return d * v / beta
		}
		if math.Log(u) < 0.5*x*x+d*(1-v+math.Log(v)) {
			return d * v / beta
		}
	}
}
