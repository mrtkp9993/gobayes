package distributions

import (
	"math"

	"github.com/mrtkp9993/gobayes/internal/utils"
	"github.com/mrtkp9993/gobayes/random"
)

type Uniform struct {
	A, B float64
}

func (Uniform) IsContinuous() {}

// https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.uniform.html
func (u Uniform) LogProb(x float64) float64 {
	if u.B <= u.A || x < u.A || x > u.B {
		return math.Inf(-1)
	}
	width := u.B - u.A
	if math.IsInf(width, 1) {
		return -math.Log(u.B/2-u.A/2) - math.Ln2
	}
	return -utils.LogPositive(width)
}

func (u Uniform) Sample() float64 {
	if u.B <= u.A {
		panic("uniform requires A < B")
	}
	draw := random.Float64()
	width := u.B - u.A
	if math.IsInf(width, 1) {
		return (1-draw)*u.A + draw*u.B
	}
	return u.A + draw*width
}
