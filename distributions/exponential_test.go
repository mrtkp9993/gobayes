package distributions

import (
	"fmt"
	"math"
	"testing"

	"github.com/mrtkp9993/gobayes/internal/disttest"
)

func TestExponentialLogProb(t *testing.T) {
	t.Run("lambda=2", func(t *testing.T) {
		disttest.RunLogProbCases(t, Exponential{Lambda: 2}, []disttest.LogProbCase{
			{Name: "x=0 (support boundary)", X: 0, Want: 0.6931471805599453},
			{Name: "x=1.3", X: 1.3, Want: -1.9068528194400547},
			{Name: "x<0", X: -0.5, Want: math.Inf(-1)},
			{Name: "x=-Inf", X: math.Inf(-1), Want: math.Inf(-1)},
		})
	})
	t.Run("lambda=0.25", func(t *testing.T) {
		disttest.RunLogProbCases(t, Exponential{Lambda: 0.25}, []disttest.LogProbCase{
			{Name: "x=4", X: 4, Want: -2.386294361119891},
		})
	})
}

func TestExponentialLogProbInvalidParams(t *testing.T) {
	for _, lambda := range []float64{0, -1} {
		d := Exponential{Lambda: lambda}
		t.Run(fmt.Sprintf("lambda=%v", lambda), func(t *testing.T) {
			disttest.AssertClose(t, d.LogProb(1), math.Inf(-1), "LogProb")
		})
	}
}

func TestExponentialSample(t *testing.T) {
	for _, lambda := range []float64{0.5, 1, 3} {
		d := Exponential{Lambda: lambda}
		t.Run(fmt.Sprintf("lambda=%v", lambda), func(t *testing.T) {
			xs := disttest.DrawSamples(t, d, disttest.NSamples, func(x float64) bool { return x >= 0 })
			disttest.CheckStats(t, d, xs, disttest.TheoreticalStats{
				Mean: 1 / lambda, Median: math.Ln2 / lambda, Var: 1 / (lambda * lambda),
				MAD:  math.Asinh(0.5) / lambda,
				Skew: 2, ExKurt: 6,
			})
		})
	}
}

func TestExponentialIntegratesToOne(t *testing.T) {
	for _, lambda := range []float64{0.5, 1, 3, 1e-3} {
		t.Run(fmt.Sprintf("lambda=%v", lambda), func(t *testing.T) {
			disttest.CheckIntegral(t, disttest.IntegrateHalfLine(t, Exponential{Lambda: lambda}, 0))
		})
	}
}

func TestExponentialSampleInvalidParamsPanics(t *testing.T) {
	for _, lambda := range []float64{0, -1} {
		disttest.AssertPanics(t, func() { Exponential{Lambda: lambda}.Sample() }, fmt.Sprintf("lambda=%v", lambda))
	}
}
