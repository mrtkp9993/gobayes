package distributions

import (
	"fmt"
	"math"
	"testing"
)

func TestNormalLogProb(t *testing.T) {
	t.Run("standard", func(t *testing.T) {
		runLogProbCases(t, Normal{Mu: 0, Sigma: 1}, []logProbCase{
			{"x=1.5", 1.5, -2.0439385332046727},
			{"x=0", 0, -0.5 * math.Log(2*math.Pi)},
		})
	})
	t.Run("mu=3,sigma=2.5", func(t *testing.T) {
		runLogProbCases(t, Normal{Mu: 3, Sigma: 2.5}, []logProbCase{
			{"x=-2", -2.0, -3.8352292650788278},
		})
	})
	t.Run("far tail", func(t *testing.T) {
		runLogProbCases(t, Normal{Mu: 0, Sigma: 0.5}, []logProbCase{
			{"x=10", 10, -200.22579135264473},
		})
	})
}

func TestNormalLogProbInvalidParams(t *testing.T) {
	for _, sigma := range []float64{0, -1} {
		d := Normal{Mu: 0, Sigma: sigma}
		for _, x := range []float64{0, 1} {
			t.Run(fmt.Sprintf("sigma=%v,x=%v", sigma, x), func(t *testing.T) {
				assertClose(t, d.LogProb(x), math.Inf(-1), "LogProb")
			})
		}
	}
}

func TestNormalSample(t *testing.T) {
	for _, d := range []Normal{{0, 1}, {-3, 0.5}, {10, 4}} {
		t.Run(fmt.Sprintf("mu=%v,sigma=%v", d.Mu, d.Sigma), func(t *testing.T) {
			xs := drawSamples(t, d, nSamples, func(float64) bool { return true })
			checkMoments(t, xs, d.Mu, d.Sigma*d.Sigma)
		})
	}
}

func TestNormalSampleInvalidParamsPanics(t *testing.T) {
	for _, sigma := range []float64{0, -1} {
		assertPanics(t, func() { Normal{Mu: 0, Sigma: sigma}.Sample() }, fmt.Sprintf("sigma=%v", sigma))
	}
}
