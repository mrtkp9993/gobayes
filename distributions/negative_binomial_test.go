package distributions

import (
	"math"
	"testing"

	"github.com/mrtkp9993/gobayes/internal/disttest"
	"github.com/mrtkp9993/gobayes/random"
)

func TestNegativeBinomialLogProb(t *testing.T) {
	disttest.RunLogProbCases(t, NegativeBinomial{R: 2, P: 0.5}, []disttest.LogProbCase{
		{Name: "zero", X: 0, Want: math.Log(0.25)},
		{Name: "one", X: 1, Want: math.Log(0.25)},
		{Name: "two", X: 2, Want: math.Log(0.1875)},
		{Name: "negative", X: -1, Want: math.Inf(-1)},
		{Name: "fractional", X: 0.5, Want: math.Inf(-1)},
		{Name: "NaN", X: math.NaN(), Want: math.Inf(-1)},
		{Name: "infinity", X: math.Inf(1), Want: math.Inf(-1)},
	})
	for _, r := range []float64{0.01, 0.5, 1, 2, 20, 1e6} {
		for _, p := range []float64{0.1, 0.5, 0.9} {
			d := NegativeBinomial{R: r, P: p}
			for _, x := range []float64{1, 7, 32, 33, 100, 10000} {
				want := math.Log(r+x-1) - math.Log(x) + math.Log1p(-p)
				got := d.LogProb(x) - d.LogProb(x-1)
				if math.Abs(got-want) > 1e-8 {
					t.Errorf("R=%g P=%g x=%g recurrence: got %g want %g", r, p, x, got, want)
				}
			}
			if r <= 20 {
				var sum float64
				for x := 0.0; x < 10000; x++ {
					sum += math.Exp(d.LogProb(x))
				}
				disttest.AssertClose(t, sum, 1, "normalization")
			}
		}
	}
	for _, x := range []float64{33, 1e12, 1e100, math.MaxFloat64} {
		d := NegativeBinomial{R: 1, P: 0.5}
		disttest.AssertClose(t, d.LogProb(x), -(x+1)*math.Ln2, "geometric")
	}
	disttest.AssertClose(t, (NegativeBinomial{R: 1e300, P: 1 - 1e-15}).LogProb(1),
		1e300*math.Log(1-1e-15)+math.Log(1e300)+math.Log1p(-(1-1e-15)), "large shape")
	d := NegativeBinomial{R: math.MaxFloat64, P: 0.5}
	disttest.AssertClose(t, d.LogProb(math.MaxFloat64), -0.5*(math.Log(4*math.Pi)+math.Log(math.MaxFloat64)), "overflowing sum at mode")
	d.P = 0.25
	disttest.AssertClose(t, d.LogProb(math.MaxFloat64), -math.MaxFloat64*math.Log(4.0/3), "overflowing sum away from mode")
	disttest.AssertClose(t, (NegativeBinomial{R: 2, P: math.SmallestNonzeroFloat64}).LogProb(0), -2148*math.Ln2, "subnormal probability")
	for _, x := range []float64{1, 33} {
		disttest.AssertClose(t, (NegativeBinomial{R: math.SmallestNonzeroFloat64, P: 0.5}).LogProb(x),
			-1074*math.Ln2-x*math.Ln2-math.Log(x), "subnormal shape")
	}
}

func TestNegativeBinomialInvalidAndDegenerate(t *testing.T) {
	for _, d := range []NegativeBinomial{
		{R: 0, P: 0.5}, {R: -1, P: 0.5}, {R: math.NaN(), P: 0.5},
		{R: math.Inf(1), P: 0.5}, {R: 2, P: 0}, {R: 2, P: -1},
		{R: 2, P: 1.1}, {R: 2, P: math.NaN()}, {R: 2, P: math.Inf(1)},
	} {
		disttest.AssertClose(t, d.LogProb(0), math.Inf(-1), "invalid parameters")
		disttest.AssertPanics(t, func() { d.Sample() }, "invalid parameters")
	}
	d := NegativeBinomial{R: 0.5, P: 1}
	disttest.AssertClose(t, d.LogProb(0), 0, "certain zero")
	disttest.AssertClose(t, d.LogProb(1), math.Inf(-1), "impossible count")
	disttest.AssertClose(t, d.Sample(), 0, "degenerate sample")
	random.Seed(42)
	disttest.AssertPanics(t, func() { (NegativeBinomial{R: 1e100, P: 0.5}).Sample() }, "unrepresentable mixing rate")
}

func TestNegativeBinomialSample(t *testing.T) {
	random.Seed(20261124)
	const draws = 100000
	for _, d := range []NegativeBinomial{{R: 0.5, P: 0.7}, {R: 2, P: 1.0 / 3}, {R: 20, P: 0.1}, {R: 1000, P: 0.9}} {
		xs := disttest.DrawSamples(t, d, draws, func(x float64) bool { return x >= 0 && math.Trunc(x) == x })
		st := disttest.ComputeStats(xs)
		mean := d.R * (1 - d.P) / d.P
		variance := mean / d.P
		kurtosis := 6/d.R + d.P*d.P/(d.R*(1-d.P))
		if math.Abs(st.Mean-mean) > disttest.SETol*math.Sqrt(variance/draws) {
			t.Errorf("%+v mean: got %g want %g", d, st.Mean, mean)
		}
		if math.Abs(st.Var-variance) > disttest.SETol*variance*math.Sqrt((kurtosis+2)/draws) {
			t.Errorf("%+v variance: got %g want %g", d, st.Var, variance)
		}
	}
	random.Seed(17)
	want := (NegativeBinomial{R: 2, P: 0.5}).Sample()
	random.Seed(17)
	if got := (NegativeBinomial{R: 2, P: 0.5}).Sample(); got != want {
		t.Fatalf("seeded draws differ: %g != %g", got, want)
	}
}
