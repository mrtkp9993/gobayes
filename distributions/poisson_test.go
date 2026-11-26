package distributions

import (
	"math"
	"testing"

	"github.com/mrtkp9993/gobayes/internal/disttest"
	"github.com/mrtkp9993/gobayes/random"
)

func TestPoissonLogProb(t *testing.T) {
	disttest.AssertClose(t, (Poisson{Lambda: math.SmallestNonzeroFloat64}).LogProb(1), -1074*math.Ln2, "subnormal mean")
	disttest.RunLogProbCases(t, Poisson{Lambda: 3}, []disttest.LogProbCase{
		{Name: "zero", X: 0, Want: -3},
		{Name: "one", X: 1, Want: math.Log(3) - 3},
		{Name: "mode", X: 3, Want: 3*math.Log(3) - 3 - math.Log(6)},
		{Name: "negative", X: -1, Want: math.Inf(-1)},
		{Name: "fractional", X: 0.5, Want: math.Inf(-1)},
		{Name: "infinity", X: math.Inf(1), Want: math.Inf(-1)},
		{Name: "NaN", X: math.NaN(), Want: math.Inf(-1)},
	})
	for _, lambda := range []float64{0.01, 1, 10, 100, 1000} {
		d := Poisson{Lambda: lambda}
		var sum float64
		for k := 0.0; k <= lambda+15*math.Sqrt(lambda)+50; k++ {
			sum += math.Exp(d.LogProb(k))
		}
		disttest.AssertClose(t, sum, 1, "total mass")
		for _, k := range []float64{1, 7, 16, 100} {
			disttest.AssertClose(t, d.LogProb(k)-d.LogProb(k-1), math.Log(lambda)-math.Log(k), "mass recurrence")
		}
	}
	const lambda = 1e12
	disttest.AssertClose(t, (Poisson{Lambda: lambda}).LogProb(lambda), -0.5*math.Log(2*math.Pi*lambda)-1/(12*lambda), "large mode")
	disttest.AssertClose(t, (Poisson{Lambda: math.MaxFloat64}).LogProb(math.MaxFloat64), -0.5*(math.Log(2*math.Pi)+math.Log(math.MaxFloat64)), "extreme finite mode")
}

func TestPoissonZeroAndInvalid(t *testing.T) {
	d := Poisson{}
	disttest.AssertClose(t, d.LogProb(0), 0, "certain zero")
	disttest.AssertClose(t, d.LogProb(1), math.Inf(-1), "impossible count")
	disttest.AssertClose(t, d.Sample(), 0, "sample")
	for _, lambda := range []float64{-1, math.NaN(), math.Inf(1), math.Inf(-1)} {
		d := Poisson{Lambda: lambda}
		disttest.AssertClose(t, d.LogProb(0), math.Inf(-1), "invalid mean")
		disttest.AssertPanics(t, func() { d.Sample() }, "invalid mean")
	}
	disttest.AssertPanics(t, func() { (Poisson{Lambda: 1 << 54}).Sample() }, "unrepresentable counts")
}

func TestPoissonSample(t *testing.T) {
	random.Seed(20261123)
	const draws = 100_000
	for _, lambda := range []float64{0.1, 3, 9.99, 10, 30, 1000, 1e6} {
		xs := disttest.DrawSamples(t, Poisson{Lambda: lambda}, draws, func(x float64) bool { return x >= 0 && math.Trunc(x) == x })
		var sum float64
		for _, x := range xs {
			sum += x
		}
		mean := sum / draws
		var variance float64
		for _, x := range xs {
			variance += (x - mean) * (x - mean)
		}
		variance /= draws - 1
		if math.Abs(mean-lambda) > disttest.SETol*math.Sqrt(lambda/draws) {
			t.Errorf("Lambda=%g: mean %g", lambda, mean)
		}
		if math.Abs(variance-lambda) > disttest.SETol*math.Sqrt((lambda+2*lambda*lambda)/draws) {
			t.Errorf("Lambda=%g: variance %g", lambda, variance)
		}
	}
}

func TestPoissonSampleAtFloat64Boundary(t *testing.T) {
	random.Seed(13)
	const lambda = 1 << 53
	const draws = 100_000
	d := Poisson{Lambda: lambda}
	var above, divisibleByFour int
	for i := 0; i < draws; i++ {
		x := d.Sample()
		if x > lambda {
			above++
			if int64(x)%4 == 0 {
				divisibleByFour++
			}
		}
	}
	if above == 0 {
		t.Fatal("no samples above Lambda")
	}
	ratio := float64(divisibleByFour) / float64(above)
	if math.Abs(ratio-0.75) > 0.02 {
		t.Fatalf("fraction divisible by 4 above Lambda = %g, want 0.75 ± 0.02", ratio)
	}
}
