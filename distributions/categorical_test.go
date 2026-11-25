package distributions

import (
	"math"
	"testing"

	"github.com/mrtkp9993/gobayes/random"
)

func TestCategoricalLogProb(t *testing.T) {
	c := Categorical{P: []float64{0.2, 0, 0.5, 0.3}}
	for k, p := range c.P {
		got := c.LogProb(float64(k))
		if p == 0 {
			if !math.IsInf(got, -1) {
				t.Fatalf("zero mass: %g", got)
			}
		} else if math.Abs(got-math.Log(p)) > 1e-14 {
			t.Fatalf("category %d: got %g, want %g", k, got, math.Log(p))
		}
	}
	for _, x := range []float64{-1, 4, 0.5, math.NaN(), math.Inf(1), math.Inf(-1)} {
		if !math.IsInf(c.LogProb(x), -1) {
			t.Fatalf("accepted %g", x)
		}
	}
	for _, p := range []float64{0, 0.2, 0.5, 1} {
		c := Categorical{P: []float64{1 - p, p}}
		for _, k := range []float64{0, 1} {
			got, want := c.LogProb(k), (Bernoulli{P: p}).LogProb(k)
			if got != want && math.Abs(got-want) > 1e-14 {
				t.Fatalf("Bernoulli mismatch: %g != %g", got, want)
			}
		}
	}
	c = Categorical{P: []float64{0.5, 0.5 + 1e-13}}
	if math.Abs(math.Exp(c.LogProb(0))+math.Exp(c.LogProb(1))-1) > 1e-15 {
		t.Fatal("mass is not normalized")
	}
	c = Categorical{P: []float64{math.SmallestNonzeroFloat64, 1}}
	if math.IsInf(c.LogProb(0), 0) {
		t.Fatal("subnormal probability lost")
	}
}

func TestCategoricalInvalid(t *testing.T) {
	for _, p := range [][]float64{nil, {}, {0, 0}, {0.2, 0.3}, {0.6, 0.6}, {-0.1, 1.1}, {math.NaN(), 1}, {math.Inf(1)}, {1 + 1e-13}, {0.5, 0.5 + 2e-12}} {
		c := Categorical{P: p}
		if !math.IsInf(c.LogProb(0), -1) {
			t.Errorf("accepted %v", p)
		}
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("Sample did not panic for %v", p)
				}
			}()
			c.Sample()
		}()
	}
}

func TestCategoricalSampling(t *testing.T) {
	random.Seed(123)
	c := Categorical{P: []float64{0, 0.2, 0.5, 0.3, 0}}
	const n = 100000
	counts := make([]int, len(c.P))
	for i := 0; i < n; i++ {
		x := c.Sample()
		if x < 0 || x >= float64(len(counts)) || math.Trunc(x) != x {
			t.Fatalf("invalid draw %g", x)
		}
		counts[int(x)]++
	}
	for k, p := range c.P {
		if p == 0 && counts[k] != 0 {
			t.Fatalf("sampled zero mass category %d", k)
		}
		if math.Abs(float64(counts[k])-n*p) > 5*math.Sqrt(n*p*(1-p)) {
			t.Fatalf("category %d: count %d", k, counts[k])
		}
	}
	random.Seed(42)
	first := make([]float64, 100)
	for i := range first {
		first[i] = c.Sample()
	}
	random.Seed(42)
	for i, want := range first {
		if got := c.Sample(); got != want {
			t.Fatalf("draw %d: %g != %g", i, got, want)
		}
	}
	for _, p := range [][]float64{{1}, {0, 1, 0}} {
		c := Categorical{P: p}
		for i := 0; i < 100; i++ {
			if c.P[int(c.Sample())] != 1 {
				t.Fatal("point mass")
			}
		}
	}
}
