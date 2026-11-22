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
	if err != nil {
		t.Fatal(err)
	}
	if *shifted.RHat < 1.2 || shifted.ESS >= d.ESS {
		t.Fatalf("shifted chains not detected: %+v", shifted)
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
	} {
		_, err := Compute(chains)
		if err == nil || !strings.Contains(err.Error(), "ESS, MCSE and R-hat are unavailable") {
			t.Fatalf("expected explicit unavailable diagnostics, got %v", err)
		}
	}
}
