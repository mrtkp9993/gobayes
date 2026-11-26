package distributions

import (
	"fmt"
	"math"
	"testing"

	"github.com/mrtkp9993/gobayes/internal/disttest"
	"github.com/mrtkp9993/gobayes/random"
)

func TestBinomialLogProb(t *testing.T) {
	disttest.AssertClose(t, (Binomial{N: 1, P: math.SmallestNonzeroFloat64}).LogProb(1), -1074*math.Ln2, "subnormal P")
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
	random.Seed(20261126)
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
	for _, d := range []Binomial{{100, 0.1}, {100, 0.2}, {100, 0.2999}, {100, 0.3}, {100, 0.5}, {200, 0.5}, {100, 0.7}, {1000, 0.03}} {
		t.Run(fmt.Sprintf("N=%d/P=%g", d.N, d.P), func(t *testing.T) {
			const draws = 100_000
			counts := make([]int, d.N+1)
			for range draws {
				x := d.Sample()
				if x < 0 || x > float64(d.N) || math.Trunc(x) != x {
					t.Fatalf("sample outside support: %g", x)
				}
				counts[int(x)]++
			}
			mass := math.Pow(1-d.P, float64(d.N))
			mean := float64(d.N) * d.P
			sd := math.Sqrt(mean * (1 - d.P))
			var tailMass [2]float64
			var tailCount [2]int
			for k, count := range counts {
				if math.Abs(float64(count)/draws-mass) > 6*math.Sqrt(mass*(1-mass)/draws)+2.0/draws {
					t.Errorf("mass at %d: got %g, want %g", k, float64(count)/draws, mass)
				}
				for tail, outside := range []bool{float64(k) <= mean-2*sd, float64(k) >= mean+2*sd} {
					if outside {
						tailMass[tail] += mass
						tailCount[tail] += count
					}
				}
				mass *= float64(d.N-k) / float64(k+1) * d.P / (1 - d.P)
			}
			for tail, want := range tailMass {
				if math.Abs(float64(tailCount[tail])/draws-want) > 6*math.Sqrt(want*(1-want)/draws) {
					t.Errorf("tail %d: got %g, want %g", tail, float64(tailCount[tail])/draws, want)
				}
			}
		})
	}
	for _, p := range []float64{0.5, 0.25, 0.75, 30.0 / (1 << 53), 1 - 0x1p-48} {
		d := Binomial{N: 1 << 53, P: p}
		const draws = 20_000
		mean := float64(d.N) * p
		variance := mean * (1 - p)
		var sum, squares float64
		for range draws {
			x := d.Sample()
			if x < 0 || x > float64(d.N) || math.Trunc(x) != x {
				t.Fatalf("P=%g: sample outside support: %g", p, x)
			}
			delta := x - mean
			sum += delta
			squares += delta * delta
		}
		if math.Abs(sum/draws) > 6*math.Sqrt(variance/draws) {
			t.Errorf("P=%g: centered mean %g", p, sum/draws)
		}
		if math.Abs(squares/draws-variance) > 6*math.Sqrt((2*variance*variance+variance)/draws) {
			t.Errorf("P=%g: centered second moment %g, want %g", p, squares/draws, variance)
		}
	}
}

func BenchmarkBinomialSample(b *testing.B) {
	for _, n := range []int{200, 1_000, 1_000_000} {
		for _, mean := range []float64{10, 15, 20, 25, 30, 40, 50, 100} {
			d := Binomial{N: n, P: mean / float64(n)}
			b.Run(fmt.Sprintf("N=%d/mean=%.0f", n, mean), func(b *testing.B) {
				random.Seed(42)
				b.ReportAllocs()
				for b.Loop() {
					d.Sample()
				}
			})
		}
	}
}
