package distributions

import (
	"fmt"
	"math"
	"testing"
)

func TestExponentialLogProb(t *testing.T) {
	t.Run("lambda=2", func(t *testing.T) {
		runLogProbCases(t, Exponential{Lambda: 2}, []logProbCase{
			{"x=0 (support boundary)", 0, 0.6931471805599453},
			{"x=1.3", 1.3, -1.9068528194400547},
			{"x<0", -0.5, math.Inf(-1)},
			{"x=-Inf", math.Inf(-1), math.Inf(-1)},
		})
	})
	t.Run("lambda=0.25", func(t *testing.T) {
		runLogProbCases(t, Exponential{Lambda: 0.25}, []logProbCase{
			{"x=4", 4, -2.386294361119891},
		})
	})
}

func TestExponentialLogProbInvalidParams(t *testing.T) {
	for _, lambda := range []float64{0, -1} {
		d := Exponential{Lambda: lambda}
		t.Run(fmt.Sprintf("lambda=%v", lambda), func(t *testing.T) {
			assertClose(t, d.LogProb(1), math.Inf(-1), "LogProb")
		})
	}
}

func TestExponentialSample(t *testing.T) {
	for _, lambda := range []float64{0.5, 1, 3} {
		d := Exponential{Lambda: lambda}
		t.Run(fmt.Sprintf("lambda=%v", lambda), func(t *testing.T) {
			xs := drawSamples(t, d, nSamples, func(x float64) bool { return x >= 0 })
			checkMoments(t, xs, 1/lambda, 1/(lambda*lambda))
		})
	}
}

func TestExponentialSampleInvalidParamsPanics(t *testing.T) {
	for _, lambda := range []float64{0, -1} {
		assertPanics(t, func() { Exponential{Lambda: lambda}.Sample() }, fmt.Sprintf("lambda=%v", lambda))
	}
}
