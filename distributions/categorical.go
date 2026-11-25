package distributions

import (
	"math"

	"github.com/mrtkp9993/gobayes/internal/utils"
	"github.com/mrtkp9993/gobayes/random"
)

type Categorical struct {
	P []float64
}

func (Categorical) IsDiscrete() {}

func (c Categorical) total() (float64, bool) {
	if len(c.P) == 0 {
		return 0, false
	}
	var sum, correction float64
	for _, p := range c.P {
		if !(p >= 0 && p <= 1) {
			return 0, false
		}
		y := p - correction
		t := sum + y
		correction = (t - sum) - y
		sum = t
	}
	return sum, math.Abs(sum-1) <= 1e-12
}

func (c Categorical) LogProb(x float64) float64 {
	total, valid := c.total()
	if !valid || !(x >= 0 && x < float64(len(c.P))) || math.Trunc(x) != x {
		return math.Inf(-1)
	}
	return utils.LogPositive(c.P[int(x)]) - math.Log(total)
}

func (c Categorical) Sample() float64 {
	total, valid := c.total()
	if !valid {
		panic("categorical requires nonempty finite P in [0,1] summing to 1 within 1e-12")
	}
	u := random.Float64() * total
	var sum float64
	lastPositive := 0
	for k, p := range c.P {
		if p == 0 {
			continue
		}
		lastPositive = k
		sum += p
		if u < sum {
			return float64(k)
		}
	}
	return float64(lastPositive)
}
