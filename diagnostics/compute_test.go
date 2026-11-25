package diagnostics

import (
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

func TestComputeIndependentAndShiftedChains(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 0))
	chains := make([][]float64, 4)
	for c := range chains {
		chains[c] = make([]float64, 2000)
		for i := range chains[c] {
			chains[c][i] = rng.NormFloat64()
		}
	}
	d, err := Compute(chains)
	if err != nil {
		t.Fatal(err)
	}
	if d.RHat == nil || math.Abs(*d.RHat-1) > 0.02 || d.ESS < 4000 || d.ESS > 8000 || d.MCSEMean <= 0 {
		t.Fatalf("unexpected diagnostics: %+v", d)
	}
	for i := range chains[0] {
		chains[0][i] += 4
	}
	shifted, err := Compute(chains)
	if err == nil || !strings.Contains(err.Error(), "lag limit") {
		t.Fatalf("expected unresolved autocorrelation tail for shifted chains, got %v", err)
	}
	if shifted.RHat == nil || *shifted.RHat < 1.2 {
		t.Fatalf("shifted chains not detected: %+v", shifted)
	}
}

func TestAutocorrelationTimeLagLimit(t *testing.T) {
	const rho = 0.9999
	correlations := make([]float64, 1001)
	for lag := range correlations {
		correlations[lag] = math.Pow(rho, float64(lag))
	}
	if _, err := autocorrelationTime(correlations, 100_000); err == nil {
		t.Fatal("expected error for a positive AR(1) tail at the lag limit")
	}
}

func TestAutocorrelationTimePairs(t *testing.T) {
	for _, tc := range []struct {
		name         string
		correlations []float64
		draws        int
		want         float64
		wantError    bool
	}{
		{"negative pair before limit", []float64{1, 0.5, -0.2, -0.1, 0.9}, 10000, 2, false},
		{"zero pair at limit", []float64{1, 0.5, 0.2, -0.2}, 10000, 2, false},
		{"unpaired negative lag", []float64{1, 0.5, -0.9}, 10000, 0, true},
		{"all available pairs", []float64{1, 0.5, 0.2, 0.1}, 4, 2.6, false},
		{"unpaired final draw", []float64{1, 0.5, 0.2}, 3, 2, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := autocorrelationTime(tc.correlations, tc.draws)
			if (err != nil) != tc.wantError {
				t.Fatalf("unexpected error: %v", err)
			}
			if !tc.wantError && math.Abs(got-tc.want) > 1e-12 {
				t.Fatalf("integrated autocorrelation time=%g, want %g", got, tc.want)
			}
		})
	}
}

func TestComputeLongAutocorrelationTail(t *testing.T) {
	const rho = 0.9999
	rng := rand.New(rand.NewPCG(42, 0))
	chain := make([]float64, 100_000)
	chain[0] = rng.NormFloat64()
	innovationScale := math.Sqrt(1 - rho*rho)
	for i := 1; i < len(chain); i++ {
		chain[i] = rho*chain[i-1] + innovationScale*rng.NormFloat64()
	}
	result, err := Compute([][]float64{chain})
	if err == nil || !strings.Contains(err.Error(), "lag limit 1000") {
		t.Fatalf("expected lag limit error for a slowly mixing AR(1) chain, got %v", err)
	}
	if result.ESS != 0 || result.MCSEMean != 0 {
		t.Fatalf("unexpected estimates from a truncated tail: %+v", result)
	}
}

func TestComputeRHatDetectsOpposingTrends(t *testing.T) {
	const draws = 101
	up, down := make([]float64, draws), make([]float64, draws)
	for i := range up {
		up[i], down[i] = float64(i), float64(draws-1-i)
	}
	d, err := Compute([][]float64{up, down})
	if err != nil {
		t.Fatal(err)
	}
	const want = 1.826757939856068
	if d.RHat == nil || !(math.Abs(*d.RHat-want) < 1e-12) {
		t.Fatalf("expected R-hat %g for opposing trends, got %+v", want, d)
	}
	up[draws/2] = 1e6
	changed, err := Compute([][]float64{up, down})
	if err != nil {
		t.Fatal(err)
	}
	if changed.RHat == nil || *changed.RHat != *d.RHat {
		t.Fatal("changing the omitted middle draw changed R-hat")
	}
}

func TestComputeFoldedRHatDetectsUnequalScales(t *testing.T) {
	narrow, wide := make([]float64, 16), make([]float64, 16)
	pattern := []float64{-2, -1, 1, 2}
	for i := range narrow {
		narrow[i] = pattern[i%len(pattern)]
		wide[i] = 10 * narrow[i]
	}
	d, err := Compute([][]float64{narrow, wide})
	if err != nil {
		t.Fatal(err)
	}
	const want = 2.1290692481655986
	if d.RHat == nil || !(math.Abs(*d.RHat-want) < 1e-12) {
		t.Fatalf("expected folded R-hat %g for unequal scales, got %+v", want, d)
	}
}

func TestComputeRHatWithConstantFoldedDraws(t *testing.T) {
	chain := []float64{-1, 1, -1, 1, -1, 1, -1, 1}
	d, err := Compute([][]float64{chain, chain})
	if err != nil {
		t.Fatal(err)
	}
	want := math.Sqrt(3.0 / 4)
	if d.RHat == nil || !(math.Abs(*d.RHat-want) < 1e-12) {
		t.Fatalf("expected rank R-hat %g with constant folded draws, got %+v", want, d)
	}
}

func TestComputeInvalidAndSingleChain(t *testing.T) {
	for _, chains := range [][][]float64{nil, {{1, 2}}, {{1, 1, 1, 1}}, {{1, 2, 3, math.NaN()}}, {{1, 2, 3, 4}, {1, 2, 3}}} {
		if _, err := Compute(chains); err == nil {
			t.Errorf("expected error for %v", chains)
		}
	}
	d, err := Compute([][]float64{{1, 2, 3, 4, 5}})
	if err != nil || d.RHat != nil || d.Autocorrelation[0] != 1 {
		t.Fatalf("single chain: %+v %v", d, err)
	}
}

func TestComputeConstantChains(t *testing.T) {
	for _, chains := range [][][]float64{
		{{0, 0, 0, 0}, {0, 0, 0, 0}},
		{{0, 0, 0, 0}, {1, 1, 1, 1}},
		{{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}},
		{{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}, {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2}},
	} {
		_, err := Compute(chains)
		if err == nil || !strings.Contains(err.Error(), "ESS, MCSE and R-hat are unavailable") {
			t.Fatalf("expected explicit unavailable diagnostics, got %v", err)
		}
	}
}

func TestComputeSingleChainACF(t *testing.T) {
	d, err := Compute([][]float64{{-1, 1, -1, 1}})
	if err != nil {
		t.Fatal(err)
	}
	if d.Autocorrelation[1] != -0.75 {
		t.Fatalf("lag 1 ACF = %g, want -0.75", d.Autocorrelation[1])
	}
	if d.ESS != 4 || d.MCSEMean != 0.5 || d.RHat != nil {
		t.Fatalf("unexpected single-chain diagnostics: %+v", d)
	}
	// Nontrivial positive correlation also uses the corrected ACF in ESS.
	d, err = Compute([][]float64{{1, 2, 3, 4, 5}})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(d.ESS-5.0/1.8) > 1e-14 || math.Abs(d.MCSEMean-math.Sqrt(0.72)) > 1e-14 {
		t.Fatalf("ESS/MCSE did not use the corrected ACF: %+v", d)
	}
}

func TestComputeVarianceUnderflow(t *testing.T) {
	_, err := Compute([][]float64{{0, 1e-200, 0, 1e-200}})
	if err == nil || !strings.Contains(err.Error(), "numerically zero") {
		t.Fatalf("expected numerical variance error, got %v", err)
	}
}
