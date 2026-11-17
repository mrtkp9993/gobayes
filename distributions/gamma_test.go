package distributions

import (
	"fmt"
	"math"
	"testing"
)

func TestGammaLogProb(t *testing.T) {
	t.Run("alpha=3,beta=2", func(t *testing.T) {
		runLogProbCases(t, Gamma{Alpha: 3, Beta: 2}, []logProbCase{
			{"x=2", 2, -1.2274112777602189},
			{"x=0", 0, math.Inf(-1)},
			{"x<0", -1, math.Inf(-1)},
		})
	})
	t.Run("alpha=0.5,beta=1.5", func(t *testing.T) {
		runLogProbCases(t, Gamma{Alpha: 0.5, Beta: 1.5}, []logProbCase{
			{"x=0.1", 0.1, 0.6316601576264049},
		})
	})
	t.Run("alpha=10,beta=1", func(t *testing.T) {
		runLogProbCases(t, Gamma{Alpha: 10, Beta: 1}, []logProbCase{
			{"x=7.5", 7.5, -2.1677002952010884},
		})
	})
	t.Run("x=0 boundary (scipy)", func(t *testing.T) {
		assertClose(t, Gamma{Alpha: 1, Beta: 2}.LogProb(0), 0.6931471805599453, "alpha=1")
		assertClose(t, Gamma{Alpha: 0.5, Beta: 1.5}.LogProb(0), math.Inf(1), "alpha<1")
		assertClose(t, Gamma{Alpha: 3, Beta: 2}.LogProb(0), math.Inf(-1), "alpha>1")
	})
	t.Run("alpha=1 matches Exponential", func(t *testing.T) {
		g, e := Gamma{Alpha: 1, Beta: 2}, Exponential{Lambda: 2}
		for _, x := range []float64{0, 0.1, 1.3, 5} {
			assertClose(t, g.LogProb(x), e.LogProb(x), fmt.Sprintf("x=%v", x))
		}
	})
}

func TestGammaLogProbInvalidParams(t *testing.T) {
	for _, p := range []Gamma{{0, 1}, {-0.5, 1}, {2, 0}, {2, -1}} {
		t.Run(fmt.Sprintf("alpha=%v,beta=%v", p.Alpha, p.Beta), func(t *testing.T) {
			assertClose(t, p.LogProb(1), math.Inf(-1), "LogProb")
		})
	}
}

func TestGammaSample(t *testing.T) {
	for _, d := range []Gamma{{0.3, 1}, {0.5, 2}, {1, 1}, {2.5, 0.5}, {10, 3}} {
		t.Run(fmt.Sprintf("alpha=%v,beta=%v", d.Alpha, d.Beta), func(t *testing.T) {
			xs := drawSamples(t, d, nSamples, func(x float64) bool { return x >= 0 })
			checkMoments(t, xs, d.Alpha/d.Beta, d.Alpha/(d.Beta*d.Beta))
		})
	}
}

func TestGammaSampleInvalidParamsPanics(t *testing.T) {
	for _, p := range []Gamma{{0, 1}, {-1, 1}, {1, 0}, {1, -2}} {
		assertPanics(t, func() { p.Sample() }, fmt.Sprintf("alpha=%v,beta=%v", p.Alpha, p.Beta))
	}
}
