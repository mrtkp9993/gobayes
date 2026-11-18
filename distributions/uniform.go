package distributions

import (
	"math"

	"github.com/mrtkp9993/gobayes/random"
)

type Uniform struct {
	A, B float64
}

func (u Uniform) LogProb(x float64) float64 {
	if u.B <= u.A || x < u.A || x > u.B {
		return math.Inf(-1)
	}
	return -math.Log(u.B - u.A)
}

func (u Uniform) Sample() float64 {
	if u.B <= u.A {
		panic("uniform requires A < B")
	}
	return u.A + random.Float64()*(u.B-u.A)
}
