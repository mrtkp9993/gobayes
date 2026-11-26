package distributions

import (
	"math"

	"github.com/mrtkp9993/gobayes/internal/utils"
	"github.com/mrtkp9993/gobayes/random"
)

type Exponential struct {
	Lambda float64
}

func (Exponential) IsContinuous() {}

func (e Exponential) LogProb(x float64) float64 {
	if !(e.Lambda > 0) || math.IsInf(e.Lambda, 0) || x < 0 {
		return math.Inf(-1)
	}
	return utils.LogPositive(e.Lambda) - e.Lambda*x
}

func (e Exponential) Sample() float64 {
	if !(e.Lambda > 0) || math.IsInf(e.Lambda, 0) {
		panic("exponential requires finite positive Lambda")
	}
	return -math.Log1p(-random.Float64()) / e.Lambda
}
