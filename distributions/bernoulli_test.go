package distributions

import (
	"math"
	"testing"

	"github.com/mrtkp9993/gobayes/internal/disttest"
	"github.com/mrtkp9993/gobayes/random"
)

func TestBernoulliLogProb(t *testing.T) {
	disttest.AssertClose(t, (Bernoulli{P: math.SmallestNonzeroFloat64}).LogProb(1), -1074*math.Ln2, "subnormal P")
	disttest.RunLogProbCases(t, Bernoulli{P: 0.25}, []disttest.LogProbCase{
		{Name: "failure", X: 0, Want: math.Log(0.75)},
		{Name: "success", X: 1, Want: math.Log(0.25)},
		{Name: "negative", X: -1, Want: math.Inf(-1)},
		{Name: "too large", X: 2, Want: math.Inf(-1)},
		{Name: "fractional", X: 0.5, Want: math.Inf(-1)},
		{Name: "infinite", X: math.Inf(1), Want: math.Inf(-1)},
		{Name: "negative infinite", X: math.Inf(-1), Want: math.Inf(-1)},
		{Name: "NaN", X: math.NaN(), Want: math.Inf(-1)},
	})
	for _, p := range []float64{0, 1e-20, 0.25, 0.5, 0.99, 1} {
		d := Bernoulli{P: p}
		disttest.AssertClose(t, math.Exp(d.LogProb(0))+math.Exp(d.LogProb(1)), 1, "total mass")
		for _, x := range []float64{0, 1} {
			disttest.AssertClose(t, d.LogProb(x), (Binomial{N: 1, P: p}).LogProb(x), "Binomial equivalence")
		}
	}
	if got, want := (Bernoulli{P: 1e-20}).LogProb(0), math.Log1p(-1e-20); got != want {
		t.Errorf("small P: LogProb(0) = %g, want %g", got, want)
	}
}

func TestBernoulliInvalidParameters(t *testing.T) {
	for _, p := range []float64{-0.1, 1.1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		d := Bernoulli{P: p}
		for _, x := range []float64{0, 1} {
			disttest.AssertClose(t, d.LogProb(x), math.Inf(-1), "invalid P")
		}
		disttest.AssertPanics(t, func() { d.Sample() }, "invalid P")
	}
}

func TestBernoulliSample(t *testing.T) {
	random.Seed(20261123)
	const draws = 20_000
	for _, p := range []float64{0, 0.01, 0.25, 0.5, 0.99, 1} {
		d := Bernoulli{P: p}
		samples := disttest.DrawSamples(t, d, draws, func(x float64) bool { return x == 0 || x == 1 })
		var sum float64
		for _, x := range samples {
			sum += x
		}
		if math.Abs(sum/draws-p) > disttest.SETol*math.Sqrt(p*(1-p)/draws) {
			t.Errorf("P=%g: success frequency = %g", p, sum/draws)
		}
	}
}
