// NOT TESTED YET
package distributions

import (
	"math"
	"math/rand/v2"
)

type Uniform struct {
	A, B float64
}

func (u Uniform) LogProb(x float64) float64 {
	if x < u.A || x > u.B {
		return math.Inf(-1)
	}
	return -math.Log(u.B - u.A)
}

func (u Uniform) Sample() float64 {
	return u.A + rand.Float64()*(u.B-u.A)
}
