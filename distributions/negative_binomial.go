package distributions

import (
	"math"

	"github.com/mrtkp9993/gobayes/internal/utils"
)

type NegativeBinomial struct {
	R float64
	P float64
}

func (NegativeBinomial) IsDiscrete() {}

func (n NegativeBinomial) valid() bool {
	return n.R > 0 && !math.IsInf(n.R, 0) && n.P > 0 && n.P <= 1
}

func (n NegativeBinomial) LogProb(x float64) float64 {
	if !n.valid() || !(x >= 0) || math.IsInf(x, 0) || math.Trunc(x) != x {
		return math.Inf(-1)
	}
	if x == 0 {
		return n.R * utils.LogPositive(n.P)
	}
	if n.P == 1 {
		return math.Inf(-1)
	}
	if x <= 32 {
		lp := n.R*utils.LogPositive(n.P) + x*math.Log1p(-n.P)
		for i := 0.0; i < x; i++ {
			lp += utils.LogPositive(n.R+i) - math.Log(i+1)
		}
		return lp
	}
	total := x + n.R
	deviance := 0.0
	logTotal := math.Log(total)
	if math.IsInf(total, 1) {
		half := x/2 + n.R/2
		logTotal = math.Log(half) + math.Ln2
		deviance = 2 * (negativeBinomialDeviance(x/2, half*(1-n.P)) + negativeBinomialDeviance(n.R/2, half*n.P))
	} else {
		deviance = negativeBinomialDeviance(x, total*(1-n.P)) + negativeBinomialDeviance(n.R, total*n.P)
	}
	shapeError := stirlingError(n.R)
	if n.R < 0x1p-1022 {
		shapeError = -(n.R+0.5)*utils.LogPositive(n.R) + n.R - 0.5*math.Log(2*math.Pi)
	}
	return stirlingError(total) - stirlingError(x) - shapeError - deviance +
		0.5*(utils.LogPositive(n.R)-logTotal-math.Log(x)-math.Log(2*math.Pi))
}

func negativeBinomialDeviance(x, mean float64) float64 {
	if math.IsInf(x+mean, 1) {
		return 2 * negativeBinomialDeviance(x/2, mean/2)
	}
	if math.Abs(x-mean) >= 0.1*(x+mean) {
		return x*(utils.LogPositive(x)-utils.LogPositive(mean)) - (x - mean)
	}
	return binomialDeviance(x, mean)
}

func (n NegativeBinomial) Sample() float64 {
	if !n.valid() {
		panic("negative binomial requires finite R > 0 and 0 < P <= 1")
	}
	if n.P == 1 {
		return 0
	}
	lambda := (Gamma{Alpha: n.R, Beta: 1}).Sample() * (1 - n.P) / n.P
	return (Poisson{Lambda: lambda}).Sample()
}
