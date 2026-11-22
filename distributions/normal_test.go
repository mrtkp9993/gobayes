package distributions

import (
	"fmt"
	"math"
	"testing"

	"github.com/mrtkp9993/gobayes/internal/disttest"
)

func TestNormalLogProb(t *testing.T) {
	t.Run("standard", func(t *testing.T) {
		disttest.RunLogProbCases(t, Normal{Mu: 0, Sigma: 1}, []disttest.LogProbCase{
			{Name: "x=1.5", X: 1.5, Want: -2.0439385332046727},
			{Name: "x=0", X: 0, Want: -0.5 * math.Log(2*math.Pi)},
		})
	})
	t.Run("mu=3,sigma=2.5", func(t *testing.T) {
		disttest.RunLogProbCases(t, Normal{Mu: 3, Sigma: 2.5}, []disttest.LogProbCase{
			{Name: "x=-2", X: -2.0, Want: -3.8352292650788278},
		})
	})
	t.Run("far tail", func(t *testing.T) {
		disttest.RunLogProbCases(t, Normal{Mu: 0, Sigma: 0.5}, []disttest.LogProbCase{
			{Name: "x=10", X: 10, Want: -200.22579135264473},
		})
	})
}

func TestNormalLogProbInvalidParams(t *testing.T) {
	for _, sigma := range []float64{0, -1} {
		d := Normal{Mu: 0, Sigma: sigma}
		for _, x := range []float64{0, 1} {
			t.Run(fmt.Sprintf("sigma=%v,x=%v", sigma, x), func(t *testing.T) {
				disttest.AssertClose(t, d.LogProb(x), math.Inf(-1), "LogProb")
			})
		}
	}
}

func TestNormalLogProbExtremeScales(t *testing.T) {
	const logNormalizer = 0.9189385332046727
	for _, tc := range []struct {
		name string
		d    Normal
		x    float64
		want float64
	}{
		{"large sigma at mean", Normal{Mu: 3, Sigma: 1e200}, 3, -200*math.Ln10 - logNormalizer},
		{"large sigma one deviation", Normal{Sigma: 1e200}, 1e200, -200*math.Ln10 - logNormalizer - 0.5},
		{"small sigma at mean", Normal{Sigma: 1e-200}, 0, 200*math.Ln10 - logNormalizer},
		{"small sigma one deviation", Normal{Sigma: 1e-200}, 1e-200, 200*math.Ln10 - logNormalizer - 0.5},
		{"smallest sigma at mean", Normal{Sigma: math.SmallestNonzeroFloat64}, 0, 1074*math.Ln2 - logNormalizer},
		{"overflowing difference", Normal{Mu: -1e308, Sigma: 1e308}, 1e308, -308*math.Ln10 - logNormalizer - 2},
		{"negative overflowing difference", Normal{Mu: 1e308, Sigma: 1e308}, -1e308, -308*math.Ln10 - logNormalizer - 2},
		{"large standardized value", Normal{Sigma: 1}, 1.5e154, -1.125e308},
	} {
		t.Run(tc.name, func(t *testing.T) {
			disttest.AssertClose(t, tc.d.LogProb(tc.x), tc.want, "LogProb")
		})
	}
}

func TestNormalSample(t *testing.T) {
	for _, d := range []Normal{{0, 1}, {-3, 0.5}, {10, 4}} {
		t.Run(fmt.Sprintf("mu=%v,sigma=%v", d.Mu, d.Sigma), func(t *testing.T) {
			xs := disttest.DrawSamples(t, d, disttest.NSamples, func(float64) bool { return true })
			disttest.CheckStats(t, d, xs, disttest.TheoreticalStats{
				Mean: d.Mu, Median: d.Mu, Var: d.Sigma * d.Sigma,
				MAD:  d.Sigma * 0.6744897501960817,
				Skew: 0, ExKurt: 0,
			})
		})
	}
}

func TestNormalIntegratesToOne(t *testing.T) {
	for _, d := range []Normal{{0, 1}, {-3, 0.5}, {10, 4}, {1e3, 1e-2}} {
		t.Run(fmt.Sprintf("mu=%v,sigma=%v", d.Mu, d.Sigma), func(t *testing.T) {
			disttest.CheckIntegral(t, disttest.IntegrateRealLine(t, d, d.Mu, d.Sigma))
		})
	}
}

func TestNormalSampleInvalidParamsPanics(t *testing.T) {
	for _, sigma := range []float64{0, -1} {
		disttest.AssertPanics(t, func() { Normal{Mu: 0, Sigma: sigma}.Sample() }, fmt.Sprintf("sigma=%v", sigma))
	}
}
