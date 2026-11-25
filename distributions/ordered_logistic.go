package distributions

import (
	"math"
	"sort"

	"github.com/mrtkp9993/gobayes/internal/utils"
	"github.com/mrtkp9993/gobayes/random"
)

type OrderedLogistic struct {
	Eta       float64
	Cutpoints []float64
}

func (OrderedLogistic) IsDiscrete() {}

func validOrdered(eta float64, cuts []float64) bool {
	if len(cuts) == 0 || math.IsNaN(eta) || math.IsInf(eta, 0) {
		return false
	}
	for k, c := range cuts {
		if math.IsNaN(c) || math.IsInf(c, 0) || (k > 0 && !(c > cuts[k-1])) {
			return false
		}
	}
	return true
}

func logSigmoid(x float64) float64 {
	if x >= 0 {
		return -math.Log1p(math.Exp(-x))
	}
	return x - math.Log1p(math.Exp(x))
}

func (d OrderedLogistic) LogProb(x float64) float64 {
	if !validOrdered(d.Eta, d.Cutpoints) || !(x >= 0 && x <= float64(len(d.Cutpoints))) || math.Trunc(x) != x {
		return math.Inf(-1)
	}
	k := int(x)
	if k == 0 {
		return logSigmoid(d.Cutpoints[0] - d.Eta)
	}
	if k == len(d.Cutpoints) {
		return logSigmoid(d.Eta - d.Cutpoints[k-1])
	}
	lower, upper := d.Cutpoints[k-1], d.Cutpoints[k]
	return logSigmoid(upper-d.Eta) + logSigmoid(d.Eta-lower) + utils.LogPositive(-math.Expm1(lower-upper))
}

func (d OrderedLogistic) Sample() float64 {
	if !validOrdered(d.Eta, d.Cutpoints) {
		panic("ordered logistic requires finite Eta and nonempty strictly increasing finite Cutpoints")
	}
	u := random.Float64()
	for u == 0 {
		u = random.Float64()
	}
	noise := math.Log(u) - math.Log1p(-u)
	return float64(sort.Search(len(d.Cutpoints), func(k int) bool { return noise <= d.Cutpoints[k]-d.Eta }))
}
