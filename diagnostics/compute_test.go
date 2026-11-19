package diagnostics

import (
	"math"
	"math/rand/v2"
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
	if err != nil {
		t.Fatal(err)
	}
	if *shifted.RHat < 1.2 || shifted.ESS >= d.ESS {
		t.Fatalf("shifted chains not detected: %+v", shifted)
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
