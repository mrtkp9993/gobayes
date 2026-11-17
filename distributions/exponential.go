// NOT TESTED YET
package distributions

import (
	"math"
	"math/rand/v2"
)

type Exponential struct {
	Lambda float64
}

func (e Exponential) LogProb(x float64) float64 {
	if e.Lambda <= 0 || x < 0 {
		return math.Inf(-1)
	}
	return math.Log(e.Lambda) - e.Lambda*x
}

func (e Exponential) Sample() float64 {
	if e.Lambda <= 0 {
		panic("exponential lambda must be positive")
	}
	return -math.Log(1-rand.Float64()) / e.Lambda
}
