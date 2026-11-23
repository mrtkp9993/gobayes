package distributions

import (
	"math"

	"github.com/mrtkp9993/gobayes/random"
)

type Poisson struct {
	Lambda float64
}

func (Poisson) IsDiscrete() {}

func (p Poisson) valid() bool {
	return p.Lambda >= 0 && !math.IsInf(p.Lambda, 0)
}

func (p Poisson) LogProb(x float64) float64 {
	if !p.valid() || x < 0 || math.IsInf(x, 0) || math.Trunc(x) != x {
		return math.Inf(-1)
	}
	if x == 0 {
		return -p.Lambda
	}
	if p.Lambda == 0 {
		return math.Inf(-1)
	}
	deviance := 0.0
	if math.IsInf(x+p.Lambda, 1) {
		deviance = 2 * binomialDeviance(x/2, p.Lambda/2)
	} else {
		deviance = binomialDeviance(x, p.Lambda)
	}
	return -stirlingError(x) - deviance - 0.5*(math.Log(2*math.Pi)+math.Log(x))
}

func (p Poisson) Sample() float64 {
	if !p.valid() || p.Lambda > 1<<53 {
		panic("poisson sampling requires finite 0 <= Lambda <= 2^53")
	}
	if p.Lambda == 0 {
		return 0
	}
	if p.Lambda < 10 {
		limit := math.Exp(-p.Lambda)
		product := 1.0
		for k := 0.0; ; k++ {
			product *= random.Float64()
			if product <= limit {
				return k
			}
		}
	}
	root := math.Sqrt(p.Lambda)
	b := 0.931 + 2.53*root
	a := -0.059 + 0.02483*b
	inverseAlpha := 1.1239 + 1.1328/(b-3.4)
	quickAccept := 0.9277 - 3.6224/(b-2)
	for {
		u, v := random.Float64()-0.5, random.Float64()
		us := 0.5 - math.Abs(u)
		if us == 0 {
			continue
		}
		k := math.Floor((2*a/us+b)*u + p.Lambda + 0.43)
		if k < 0 {
			continue
		}
		if us >= 0.07 && v <= quickAccept {
			return k
		}
		if us < 0.013 && v > us {
			continue
		}
		if math.Log(v)+math.Log(inverseAlpha)-math.Log(a/(us*us)+b) <= p.LogProb(k) {
			return k
		}
	}
}
