package distributions

import (
	"math"
	"testing"

	"github.com/mrtkp9993/gobayes/internal/disttest"
)

func TestBinomialLogProb(t *testing.T) {
	disttest.RunLogProbCases(t, Binomial{N: 4, P: 0.25}, []disttest.LogProbCase{
		{Name: "zero successes", X: 0, Want: math.Log(81.0 / 256)},
		{Name: "two successes", X: 2, Want: math.Log(54.0 / 256)},
		{Name: "all successes", X: 4, Want: math.Log(1.0 / 256)},
		{Name: "negative", X: -1, Want: math.Inf(-1)},
		{Name: "too large", X: 5, Want: math.Inf(-1)},
		{Name: "fractional", X: 1.5, Want: math.Inf(-1)},
		{Name: "infinite", X: math.Inf(1), Want: math.Inf(-1)},
		{Name: "NaN", X: math.NaN(), Want: math.Inf(-1)},
	})

	for _, d := range []Binomial{{N: 20, P: 0.01}, {N: 1000, P: 0.3}, {N: 100, P: 0.99}} {
		var sum float64
		for k := 0; k <= d.N; k++ {
			sum += math.Exp(d.LogProb(float64(k)))
		}
		disttest.AssertClose(t, sum, 1, "sum of probability masses")
	}

	const m = 1_000_000_000
	want := -0.5*math.Log(math.Pi*m) - 1.0/(8*m)
	disttest.AssertClose(t, (Binomial{N: 2 * m, P: 0.5}).LogProb(m), want, "large central mass")
}

func TestBinomialDegenerate(t *testing.T) {
	for _, c := range []struct {
		d    Binomial
		want float64
	}{
		{d: Binomial{N: 0, P: 0.5}, want: 0},
		{d: Binomial{N: 4, P: 0}, want: 0},
		{d: Binomial{N: 4, P: 1}, want: 4},
	} {
		disttest.AssertClose(t, c.d.LogProb(c.want), 0, "certain outcome")
		disttest.AssertClose(t, c.d.LogProb(1), math.Inf(-1), "impossible outcome")
		disttest.AssertClose(t, c.d.Sample(), c.want, "sample")
	}
}

func TestBinomialInvalidParameters(t *testing.T) {
	for _, d := range []Binomial{{N: -1, P: 0.5}, {N: 4, P: -0.1}, {N: 4, P: 1.1}, {N: 4, P: math.NaN()}} {
		disttest.AssertClose(t, d.LogProb(0), math.Inf(-1), "invalid parameters")
		disttest.AssertPanics(t, func() { d.Sample() }, "invalid parameters")
	}
}

func TestBinomialSample(t *testing.T) {
	const draws = 20_000
	const n = 4
	d := Binomial{N: n, P: 0.75}
	samples := disttest.DrawSamples(t, d, draws, func(x float64) bool {
		return x >= 0 && x <= n && math.Trunc(x) == x
	})
	var counts [n + 1]int
	for _, sample := range samples {
		counts[int(sample)]++
	}
	for k, count := range counts {
		prob := math.Exp(d.LogProb(float64(k)))
		frequency := float64(count) / draws
		se := math.Sqrt(prob * (1 - prob) / draws)
		if math.Abs(frequency-prob) > disttest.SETol*se {
			t.Errorf("frequency for %d = %.5f, want %.5f", k, frequency, prob)
		}
	}
}
