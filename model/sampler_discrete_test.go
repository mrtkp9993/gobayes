package model

import (
	"fmt"
	"math"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/mrtkp9993/gobayes/distributions"
	"github.com/mrtkp9993/gobayes/random"
	"github.com/mrtkp9993/gobayes/variable"
)

type discretePriorWithoutSample struct {
	distributions.Binomial
}

func (discretePriorWithoutSample) Sample() float64 {
	panic("MCMC must use its own generator instead of Prior.Sample")
}

func TestDiscreteSymmetricProposal(t *testing.T) {
	prior := distributions.Binomial{N: 4, P: 0.25}
	m := &Model{
		Variables: []*variable.Variable{
			{Name: "k", Value: 1, Prior: prior},
			{Name: "x", Prior: distributions.Normal{Sigma: 1}, ProposalScale: 0.5},
		},
		LogLik: func(map[string]float64) float64 { return 0 },
	}
	current := m.initialState(nil)
	for _, componentwise := range []bool{false, true} {
		proposal := Proposal{Params: make(map[string]float64)}
		rng := random.New(42, 0)
		moved := false
		for range 100 {
			if componentwise {
				m.proposeComponent(current, rng, 0, 0, &proposal)
				if proposal.Params["x"] != current.Params["x"] {
					t.Fatal("discrete component update changed another variable")
				}
			} else {
				m.proposeWithScales(current, rng, nil, &proposal)
			}
			if proposal.LogQRatio != 0 {
				t.Fatalf("componentwise=%v: symmetric proposal has log ratio %g", componentwise, proposal.LogQRatio)
			}
			value := proposal.Params["k"]
			if math.Trunc(value) != value {
				t.Fatalf("noninteger proposal: %g", value)
			}
			moved = moved || value != current.Params["k"]
		}
		if !moved {
			t.Fatal("discrete proposal did not move with the default scale")
		}
	}
}

func TestDiscreteMixedPosterior(t *testing.T) {
	for _, componentwise := range []bool{false, true} {
		name := "joint"
		if componentwise {
			name = "componentwise"
		}
		t.Run(name, func(t *testing.T) {
			prior := distributions.Binomial{N: 4, P: 0.25}
			m := &Model{
				Variables: []*variable.Variable{
					{Name: "k", Value: 1, Prior: prior},
					{Name: "x", Prior: distributions.Normal{Sigma: 1}, ProposalScale: 0.5},
				},
				LogLik: func(p map[string]float64) float64 { return p["k"] * math.Log(2) },
			}
			result, err := m.Run(Options{
				Iterations: 50_000, BurnIn: 2_000, Seed: 42,
				Adapt: true, Componentwise: componentwise,
			})
			if err != nil {
				t.Fatal(err)
			}
			if result.Scales[0][0] != 1 {
				t.Fatal("discrete scale was adapted")
			}
			if result.Scales[0][1] == 0.5 {
				t.Fatal("continuous scale was not adapted")
			}
			var counts [5]int
			for _, value := range result.Draws[0][0] {
				if value < 0 || value > 4 || math.Trunc(value) != value {
					t.Fatalf("discrete draw outside support: %g", value)
				}
				counts[int(value)]++
			}
			var masses [5]float64
			var total float64
			for k := range masses {
				masses[k] = math.Exp(prior.LogProb(float64(k)) + float64(k)*math.Log(2))
				total += masses[k]
			}
			for k, count := range counts {
				got := float64(count) / float64(len(result.Draws[0][0]))
				if want := masses[k] / total; math.Abs(got-want) > 0.02 {
					t.Errorf("posterior frequency at %d=%g, want %g", k, got, want)
				}
			}
			d, err := result.ParameterDiagnostics("k")
			if err != nil || d.ESS <= 0 {
				t.Fatalf("invalid discrete diagnostics: %v, %v", d, err)
			}
			if rate := result.AcceptRate()[0]; rate <= 0 || rate > 1 {
				t.Fatalf("invalid acceptance rate: %g", rate)
			}
		})
	}
}

func TestDependentDiscreteContinuousPosterior(t *testing.T) {
	// k ~ Binomial(4, 1/4), x ~ Normal(0, 1), y | k,x ~ Normal(k+x, 1).
	// For y=2, integrating out x gives weights proportional to
	// choose(4,k)*3^(4-k)*exp(-(2-k)^2/4).
	// The conditional posterior x | k,y is Normal((2-k)/2, 1/2).
	const observed = 2.0
	weights := [5]float64{81, 108, 54, 12, 1}
	var total float64
	for k := range weights {
		delta := observed - float64(k)
		weights[k] *= math.Exp(-delta * delta / 4)
		total += weights[k]
	}
	var meanK, secondK float64
	for k := range weights {
		weights[k] /= total
		meanK += float64(k) * weights[k]
		secondK += float64(k*k) * weights[k]
	}
	varK := secondK - meanK*meanK
	meanX := (observed - meanK) / 2
	varX := 0.5 + varK/4

	for _, componentwise := range []bool{false, true} {
		name := "joint"
		if componentwise {
			name = "componentwise"
		}
		t.Run(name, func(t *testing.T) {
			m := &Model{
				Variables: []*variable.Variable{
					{Name: "k", Value: 1, Prior: distributions.Binomial{N: 4, P: 0.25}},
					{Name: "x", Prior: distributions.Normal{Sigma: 1}, ProposalScale: 0.5},
				},
				LogLik: func(p map[string]float64) float64 {
					return (distributions.Normal{Mu: p["k"] + p["x"], Sigma: 1}).LogProb(observed)
				},
			}
			result, err := m.Run(Options{
				Iterations: testIters, BurnIn: testBurnIn, Chains: 2, Seed: testSeed,
				Adapt: true, Componentwise: componentwise,
			})
			if err != nil {
				t.Fatal(err)
			}
			for c, chain := range result.Draws {
				t.Run(fmt.Sprintf("chain%d", c), func(t *testing.T) {
					ks, xs := chain[0], chain[1]
					checkPosterior(t, ks, meanK, varK)
					checkPosterior(t, xs, meanX, varX)
					for k, probability := range weights {
						t.Run(fmt.Sprintf("k%d", k), func(t *testing.T) {
							frequency, se := batchMeans(ks, func(value float64) float64 {
								if value == float64(k) {
									return 1
								}
								return 0
							})
							if !(math.Abs(frequency-probability) <= mcseTol*se) {
								t.Errorf("posterior mass=%g, want %g (MCSE %g)", frequency, probability, se)
							}
							// Check E[1(k=j)*(x-E[x|k=j,y])] = 0.
							// This detects incorrect dependence even if marginals agree.
							residuals := make([]float64, len(xs))
							for i, value := range ks {
								if value == float64(k) {
									residuals[i] = xs[i] - (observed-float64(k))/2
								}
							}
							residual, se := batchMeans(residuals, func(value float64) float64 { return value })
							if !(math.Abs(residual) <= mcseTol*se) {
								t.Errorf("conditional mean residual=%g, want 0 (MCSE %g)", residual, se)
							}
						})
					}
				})
			}
		})
	}
}

func TestDiscreteMultipleChainsReproducible(t *testing.T) {
	for _, componentwise := range []bool{false, true} {
		prior := distributions.Binomial{N: 1, P: 0.3}
		for _, distribution := range []distributions.Distribution{prior, &prior, discretePriorWithoutSample{prior}} {
			m := &Model{
				Variables: []*variable.Variable{{Name: "k", Prior: distribution}},
				LogLik:    func(p map[string]float64) float64 { return p["k"] * math.Log(2) },
			}
			options := Options{Iterations: 200, BurnIn: 50, Chains: 4, Seed: 42, Componentwise: componentwise}
			want, err := m.Run(options)
			if err != nil {
				t.Fatal(err)
			}
			var wg sync.WaitGroup
			for range 3 {
				wg.Add(1)
				go func() {
					defer wg.Done()
					got, err := m.Run(options)
					if err != nil {
						t.Error(err)
					} else if !reflect.DeepEqual(got, want) {
						t.Error("concurrent runs with the same seed differ")
					}
				}()
			}
			wg.Wait()
		}
	}
}

func TestDiscreteInitialStatesAndConstantDiagnostics(t *testing.T) {
	m := &Model{
		Variables: []*variable.Variable{{Name: "k", Prior: distributions.Binomial{N: 1, P: 0.5}}},
		LogLik:    func(map[string]float64) float64 { return 0 },
	}
	first, err := m.randomizedInitialState(random.New(42, 0), nil)
	if err != nil {
		t.Fatal(err)
	}
	second, err := m.randomizedInitialState(random.New(43, 0), []Proposal{first})
	if err != nil || first.Params["k"] == second.Params["k"] {
		t.Fatalf("expected different starts while available: %v, %v", second, err)
	}
	third, err := m.randomizedInitialState(random.New(44, 0), []Proposal{first, second})
	if err != nil || (third.Params["k"] != 0 && third.Params["k"] != 1) {
		t.Fatalf("expected a repeated valid start: %v, %v", third, err)
	}

	m.Variables[0].Prior = distributions.Binomial{N: 1, P: 0}
	result, err := m.Run(Options{Iterations: 10, Chains: 4, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := result.ParameterDiagnostics("k"); err == nil || !strings.Contains(err.Error(), "constant draws") {
		t.Fatalf("expected explicit constant-draw diagnostics error, got %v", err)
	}
}

func TestDiscreteMultiplicativeRejected(t *testing.T) {
	m := &Model{
		Variables: []*variable.Variable{{
			Name: "k", Value: 1, Prior: distributions.Binomial{N: 4, P: 0.5}, Multiplicative: true,
		}},
		LogLik: func(map[string]float64) float64 { return 0 },
	}
	if _, err := m.Run(Options{Iterations: 10}); err == nil {
		t.Fatal("expected an error for a multiplicative discrete proposal")
	}
}

func TestDiscreteRandomWalkValidation(t *testing.T) {
	for _, scale := range []float64{-1, math.NaN(), math.Inf(1)} {
		m := &Model{
			Variables: []*variable.Variable{{
				Name: "k", Prior: distributions.Binomial{N: 4, P: 0.5}, ProposalScale: scale,
			}},
			LogLik: func(map[string]float64) float64 { return 0 },
		}
		if _, err := m.Run(Options{Iterations: 10}); err == nil || !strings.Contains(err.Error(), "ProposalScale") {
			t.Fatalf("expected invalid scale error for %g, got %v", scale, err)
		}
	}
	for _, value := range []float64{0.5, 1 << 54} {
		m := &Model{
			Variables: []*variable.Variable{{
				Name: "k", Value: value, Prior: distributions.Binomial{N: 4, P: 0.5},
			}},
			LogLik: func(map[string]float64) float64 { return 0 },
		}
		if _, err := m.Run(Options{Iterations: 10}); err == nil || !strings.Contains(err.Error(), "discrete Value") {
			t.Fatalf("expected integer value error for %g, got %v", value, err)
		}
	}
}

func TestDiscreteRandomWalkRejectsOutsideSupport(t *testing.T) {
	m := &Model{
		Variables: []*variable.Variable{{Name: "k", Prior: distributions.Binomial{N: 1, P: 0.3}}},
		LogLik:    func(map[string]float64) float64 { return 0 },
	}
	current := m.initialState(nil)
	for _, componentwise := range []bool{false, true} {
		rng := random.New(42, 0)
		proposal := Proposal{Params: make(map[string]float64)}
		outside := 0
		for range 100 {
			if componentwise {
				m.proposeComponent(current, rng, 2, 0, &proposal)
			} else {
				m.proposeWithScales(current, rng, []float64{2}, &proposal)
			}
			if proposal.Params["k"] < 0 || proposal.Params["k"] > 1 {
				outside++
				if m.accept(&proposal, current, rng) {
					t.Fatal("accepted a proposal outside the prior support")
				}
			}
		}
		if outside == 0 {
			t.Fatal("expected unbounded integer proposals")
		}
	}
}

func TestDiscreteRandomWalkIntegerBoundary(t *testing.T) {
	const limit = 1 << 53
	rng := random.New(42, 0)
	for _, current := range []float64{-limit, limit} {
		rejected := 0
		for range 100 {
			value := proposeDiscrete(current, 2, rng)
			if math.IsInf(value, 1) {
				rejected++
				continue
			}
			if math.Abs(value) > limit || math.Trunc(value) != value {
				t.Fatalf("invalid integer proposal at boundary: %g", value)
			}
		}
		if rejected == 0 {
			t.Fatal("expected out-of-range integer steps to be rejected")
		}
	}
}
