package distributions

import (
	"fmt"
	"math"
	"testing"

	"github.com/mrtkp9993/gobayes/internal/disttest"
	"github.com/mrtkp9993/gobayes/random"
)

func TestOrderedProbitProbabilities(t *testing.T) {
	for _, eta := range []float64{-40, -2, 0, 1.5, 40} {
		cuts := []float64{-3, -1, 0.5, 2}
		name := "probit"
		d := OrderedProbit{Eta: eta, Cutpoints: cuts}
		t.Run(fmt.Sprintf("%s/eta=%g", name, eta), func(t *testing.T) {
			var total float64
			for k := 0; k <= len(cuts); k++ {
				lp := d.LogProb(float64(k))
				if math.IsNaN(lp) || lp > 0 {
					t.Fatalf("category %d: invalid log probability %g", k, lp)
				}
				total += math.Exp(lp)
			}
			if math.Abs(total-1) > 2e-15 {
				t.Fatalf("total mass = %.17g", total)
			}
			for _, x := range []float64{-1, 5, 0.5, math.NaN(), math.Inf(-1), math.Inf(1)} {
				if !math.IsInf(d.LogProb(x), -1) {
					t.Errorf("accepted observation %g", x)
				}
			}
		})
	}
	// Independent central probabilities, including the location/sign convention.
	for _, tc := range []struct {
		d    OrderedProbit
		want []float64
	}{
		{OrderedProbit{Cutpoints: []float64{-1, 1}}, []float64{0.15865525393145705, 0.6826894921370859, 0.15865525393145705}},
		{OrderedProbit{Eta: 1, Cutpoints: []float64{1}}, []float64{0.5, 0.5}},
	} {
		for k, want := range tc.want {
			if got := math.Exp(tc.d.LogProb(float64(k))); math.Abs(got-want) > 2e-15 {
				t.Errorf("%+v category %d: got %.17g, want %.17g", tc.d, k, got, want)
			}
		}
	}
}

func TestOrderedProbitInvalidParameters(t *testing.T) {
	for _, tc := range []struct {
		eta  float64
		cuts []float64
	}{
		{0, nil}, {0, []float64{}}, {0, []float64{1, 1}}, {0, []float64{1, -1}},
		{0, []float64{math.NaN()}}, {0, []float64{math.Inf(-1), 1}},
		{0, []float64{0, math.Inf(1)}}, {math.NaN(), []float64{0}},
		{math.Inf(-1), []float64{0}}, {math.Inf(1), []float64{0}},
	} {
		d := OrderedProbit{Eta: tc.eta, Cutpoints: tc.cuts}
		for _, x := range []float64{0, 1} {
			if !math.IsInf(d.LogProb(x), -1) {
				t.Errorf("accepted parameters %+v", d)
			}
		}
		disttest.AssertPanics(t, func() { d.Sample() }, fmt.Sprintf("invalid parameters %+v", d))
	}
}

func TestOrderedProbitTailAndNarrowIntervals(t *testing.T) {
	// Reference log probabilities computed with mpmath at 100 decimal digits,
	// using the exact float64 cutpoints as inputs and independent CDF formulas.
	for _, tc := range []struct {
		name    string
		d       OrderedProbit
		x, want float64
	}{
		{"probit left tail", OrderedProbit{Cutpoints: []float64{-40}}, 0, -804.60844201375378817},
		{"probit right tail", OrderedProbit{Cutpoints: []float64{40}}, 1, -804.60844201375378817},
		{"probit left interval", OrderedProbit{Cutpoints: []float64{-41, -40}}, 1, -804.60844201375378817},
		{"probit right interval", OrderedProbit{Cutpoints: []float64{8, 9}}, 1, -35.013618593437148117},
		{"probit narrow center", OrderedProbit{Cutpoints: []float64{0, 1e-12}}, 1, -28.549959649133220970},
		{"probit symmetric narrow", OrderedProbit{Cutpoints: []float64{-1e-12, 1e-12}}, 1, -27.856812468573275661},
		{"probit narrow tail", OrderedProbit{Cutpoints: []float64{30, 30 + 1e-6}}, 1, -464.73446409010367254},
		{"probit subnormal width", OrderedProbit{Cutpoints: []float64{0, math.SmallestNonzeroFloat64}}, 1, -1074*math.Ln2 - 0.5*math.Log(2*math.Pi)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.d.LogProb(tc.x); math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-tc.want) > 2e-12 {
				t.Fatalf("got %.17g, want %.17g", got, tc.want)
			}
		})
	}
}

func TestOrderedProbitSampling(t *testing.T) {
	name := "probit"
	d := OrderedProbit{Eta: 0.7, Cutpoints: []float64{-1, 0.5, 2}}
	t.Run(name, func(t *testing.T) {
		random.Seed(20261125)
		const n = 100000
		var counts [4]int
		for i := 0; i < n; i++ {
			x := d.Sample()
			if !(x >= 0 && x < 4) || math.Trunc(x) != x {
				t.Fatalf("invalid draw %g", x)
			}
			counts[int(x)]++
		}
		// Independent CDF at moderate thresholds.
		cdf := func(x float64) float64 { return 0.5 * math.Erfc(-x/math.Sqrt2) }
		bounds := []float64{0, cdf(-1 - 0.7), cdf(0.5 - 0.7), cdf(2 - 0.7), 1}
		for k, count := range counts {
			p := bounds[k+1] - bounds[k]
			if math.Abs(float64(count)-n*p) > 6*math.Sqrt(n*p*(1-p)) {
				t.Errorf("category %d: count %d, expected %.2f", k, count, n*p)
			}
		}
		random.Seed(42)
		var first [100]float64
		for i := range first {
			first[i] = d.Sample()
		}
		random.Seed(42)
		for i, want := range first {
			if got := d.Sample(); got != want {
				t.Fatalf("seeded draw %d: %g != %g", i, got, want)
			}
		}
	})
}
