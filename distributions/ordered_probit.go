package distributions

import (
	"math"
	"sort"

	"github.com/mrtkp9993/gobayes/internal/utils"
	"github.com/mrtkp9993/gobayes/random"
)

type OrderedProbit struct {
	Eta       float64
	Cutpoints []float64
}

func (OrderedProbit) IsDiscrete() {}

func logNormalCDF(x float64) float64 {
	if x >= 0 {
		return math.Log1p(-0.5 * math.Erfc(x/math.Sqrt2))
	}
	if x > -20 {
		return math.Log(0.5 * math.Erfc(-x/math.Sqrt2))
	}
	z := -x
	sum, term := 1.0, 1.0
	for k := 1.0; ; k++ {
		nextTerm := term * (-(2*k - 1) / z / z)
		if math.Abs(nextTerm) >= math.Abs(term) {
			break
		}
		next := sum + nextTerm
		if next == sum {
			break
		}
		sum, term = next, nextTerm
	}
	return -(0.5*z)*z - math.Log(z) - 0.5*math.Log(2*math.Pi) + math.Log(sum)
}

func (d OrderedProbit) LogProb(x float64) float64 {
	if !validOrdered(d.Eta, d.Cutpoints) || !(x >= 0 && x <= float64(len(d.Cutpoints))) || math.Trunc(x) != x {
		return math.Inf(-1)
	}
	k := int(x)
	if k == 0 {
		return logNormalCDF(d.Cutpoints[0] - d.Eta)
	}
	if k == len(d.Cutpoints) {
		return logNormalCDF(d.Eta - d.Cutpoints[k-1])
	}
	lower, upper := d.Cutpoints[k-1], d.Cutpoints[k]
	width := upper - lower
	mid := lower/2 + upper/2 - d.Eta
	if width*(1+math.Abs(mid)) < 0.1 {
		nodes := [4]float64{0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363}
		weights := [4]float64{0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763}
		var integral float64
		for i, node := range nodes {
			t := node * (width / 2)
			integral += weights[i] * math.Exp(-0.5*t*t) * math.Cosh(mid*t)
		}
		return utils.LogPositive(width) - (0.5*mid)*mid - 0.5*math.Log(2*math.Pi) + math.Log(integral)
	}
	a, b := lower-d.Eta, upper-d.Eta
	if a >= 0 {
		a, b = -b, -a
	}
	logUpper, logLower := logNormalCDF(b), logNormalCDF(a)
	if logUpper == logLower {
		return math.Inf(-1)
	}
	return logUpper + math.Log(-math.Expm1(logLower-logUpper))
}

func (d OrderedProbit) Sample() float64 {
	if !validOrdered(d.Eta, d.Cutpoints) {
		panic("ordered probit requires finite Eta and nonempty strictly increasing finite Cutpoints")
	}
	noise := random.NormFloat64()
	return float64(sort.Search(len(d.Cutpoints), func(k int) bool { return noise <= d.Cutpoints[k]-d.Eta }))
}
