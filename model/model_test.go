package model

import (
	"math"
	"reflect"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/mrtkp9993/gobayes/distributions"
	"github.com/mrtkp9993/gobayes/variable"
)

func TestValidateReportsMultipleErrors(t *testing.T) {
	v := &variable.Variable{Name: "x", Value: math.NaN(), ProposalScale: math.Inf(1)}
	m := &Model{Variables: []*variable.Variable{nil, v, {Name: "x", Value: 0, Prior: distributions.Normal{Sigma: 1}, ProposalScale: 1, Multiplicative: true}}}
	_, err := m.Run(Options{Iterations: -1})
	if err == nil {
		t.Fatal("expected validation errors")
	}
	for _, want := range []string{"LogLik is nil", "variable 0 is nil", "Prior is nil", "ProposalScale", "Value must be finite", "duplicate", "multiplicative", "invalid Iterations"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("missing %q in %v", want, err)
		}
	}
}

const (
	testIters  = 50_000
	testBurnIn = 2_000
	testSeed   = 20261118
	nBatches   = 50
	mcseTol    = 5.0
)

func draws(t *testing.T, m *Model, name string, seed uint64) []float64 {
	t.Helper()
	r, err := m.Run(Options{Iterations: testIters, BurnIn: testBurnIn, Seed: seed})
	if err != nil {
		t.Fatal(err)
	}
	xs, err := r.Column(name)
	if err != nil {
		t.Fatal(err)
	}
	return xs[0]
}

func batchMeans(xs []float64, f func(float64) float64) (mean, mcse float64) {
	size := len(xs) / nBatches
	bm := make([]float64, nBatches)
	for b := range bm {
		var s float64
		for _, x := range xs[b*size : (b+1)*size] {
			s += f(x)
		}
		bm[b] = s / float64(size)
		mean += bm[b]
	}
	mean /= nBatches
	var v float64
	for _, m := range bm {
		v += (m - mean) * (m - mean)
	}
	v /= nBatches - 1
	return mean, math.Sqrt(v / nBatches)
}

func checkPosterior(t *testing.T, xs []float64, wantMean, wantVar float64) {
	t.Helper()
	mean, se := batchMeans(xs, func(x float64) float64 { return x })
	if d := math.Abs(mean - wantMean); d > mcseTol*se {
		t.Errorf("posterior mean: got %.6g, want %.6g (|diff| %.3g > %.3g)", mean, wantMean, d, mcseTol*se)
	}
	v, seV := batchMeans(xs, func(x float64) float64 { d := x - wantMean; return d * d })
	if d := math.Abs(v - wantVar); d > mcseTol*seV {
		t.Errorf("posterior variance: got %.6g, want %.6g (|diff| %.3g > %.3g)", v, wantVar, d, mcseTol*seV)
	}
}

func TestConjugateBetaBinomial(t *testing.T) {
	k, n := 8.0, 12.0
	theta := &variable.Variable{Name: "theta", Value: 0.5, Prior: distributions.Uniform{A: 0, B: 1}, ProposalScale: 0.2}
	m := &Model{
		Variables: []*variable.Variable{theta},
		LogLik: func(p map[string]float64) float64 {
			th := p["theta"]
			if th <= 0 || th >= 1 {
				return math.Inf(-1)
			}
			return k*math.Log(th) + (n-k)*math.Log(1-th)
		},
	}
	a, b := 1+k, 1+n-k
	checkPosterior(t, draws(t, m, "theta", testSeed), a/(a+b), a*b/((a+b)*(a+b)*(a+b+1)))
}

func TestConjugateGammaExponential(t *testing.T) {
	r, T := 2.0, 1909.0
	lambda := &variable.Variable{
		Name: "lambda", Value: 0.001, Prior: distributions.Gamma{Alpha: 2, Beta: 1400},
		ProposalScale: 0.6, Multiplicative: true,
	}
	m := &Model{
		Variables: []*variable.Variable{lambda},
		LogLik: func(p map[string]float64) float64 {
			l := p["lambda"]
			if l <= 0 {
				return math.Inf(-1)
			}
			return r*math.Log(l) - l*T
		},
	}
	a, b := 4.0, 3309.0
	checkPosterior(t, draws(t, m, "lambda", testSeed), a/b, a/(b*b))
}

func TestConjugateNormalNormal(t *testing.T) {
	data := []float64{3.1, 4.7, 2.2, 5.0, 3.9, 4.4, 2.8, 3.6, 4.1, 3.3}
	sigma, mu0, tau0 := 2.0, 0.0, 10.0
	mu := &variable.Variable{Name: "mu", Value: 0, Prior: distributions.Normal{Mu: mu0, Sigma: tau0}, ProposalScale: 1}
	m := &Model{
		Variables: []*variable.Variable{mu},
		LogLik: func(p map[string]float64) float64 {
			d := distributions.Normal{Mu: p["mu"], Sigma: sigma}
			var ll float64
			for _, x := range data {
				ll += d.LogProb(x)
			}
			return ll
		},
	}
	var sum float64
	for _, x := range data {
		sum += x
	}
	prec := 1/(tau0*tau0) + float64(len(data))/(sigma*sigma)
	postMean := (mu0/(tau0*tau0) + sum/(sigma*sigma)) / prec
	checkPosterior(t, draws(t, m, "mu", testSeed), postMean, 1/prec)
}

func simpleModel() (*Model, *variable.Variable) {
	v := &variable.Variable{Name: "mu", Value: 0, Prior: distributions.Normal{Mu: 0, Sigma: 10}, ProposalScale: 1}
	return &Model{
		Variables: []*variable.Variable{v},
		LogLik:    func(p map[string]float64) float64 { return distributions.Normal{Mu: p["mu"], Sigma: 1}.LogProb(2) },
	}, v
}

func TestSeedReproducible(t *testing.T) {
	run := func(seed uint64) []float64 {
		m, _ := simpleModel()
		return draws(t, m, "mu", seed)
	}
	a, b, c := run(42), run(42), run(7)
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("same seed differs at draw %d: %v vs %v", i, a[i], b[i])
		}
	}
	same := true
	for i := range a {
		if a[i] != c[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("different seeds produced identical chains")
	}
}

func TestRunKeepsInitialValue(t *testing.T) {
	m, v := simpleModel()
	initial := v.Value
	samples, err := m.Run(Options{Iterations: 100, BurnIn: 10, Seed: testSeed})
	if err != nil {
		t.Fatal(err)
	}
	if v.Value != initial {
		t.Fatalf("initial value changed: got %v, want %v", v.Value, initial)
	}
	moved := false
	for _, sample := range samples.Draws[0][0] {
		if sample != initial {
			moved = true
			break
		}
	}
	if !moved {
		t.Fatal("chain did not move from its initial value")
	}
}

func TestParallelRunsReproducible(t *testing.T) {
	m, _ := simpleModel()
	want, err := m.Run(Options{Iterations: 1000, BurnIn: 100, Seed: testSeed})
	if err != nil {
		t.Fatal(err)
	}
	const chains = 4
	results := make([]*Result, chains)
	errs := make([]error, chains)
	var wg sync.WaitGroup
	for i := range results {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			results[i], errs[i] = m.Run(Options{Iterations: 1000, BurnIn: 100, Seed: testSeed})
		}(i)
	}
	wg.Wait()
	for i := range results {
		if errs[i] != nil {
			t.Fatalf("chain %d: %v", i, errs[i])
		}
		if !reflect.DeepEqual(results[i], want) {
			t.Errorf("chain %d differs from sequential run", i)
		}
	}
}

func TestRunMultipleChainsAndThin(t *testing.T) {
	m, _ := simpleModel()
	o := Options{Iterations: 40, BurnIn: 10, Thin: 3, Chains: 3, Seed: testSeed}
	r, err := m.Run(o)
	if err != nil {
		t.Fatal(err)
	}
	again, err := m.Run(o)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(r, again) {
		t.Fatal("runs with the same seed differ")
	}
	if reflect.DeepEqual(r.Draws[0], r.Draws[1]) {
		t.Fatal("independent streams produced identical chains")
	}
	col, err := r.Column("mu")
	if err != nil {
		t.Fatal(err)
	}
	for c := range r.Draws {
		if len(r.Draws[c]) != len(r.Names) || len(col[c]) != o.Iterations {
			t.Fatal("wrong number of draws")
		}
		for i := range col[c] {
			if col[c][i] != r.Draws[c][0][i] {
				t.Fatal("incorrect column")
			}
		}
		if r.AcceptRate()[c] < 0 || r.AcceptRate()[c] > 1 {
			t.Fatal("invalid acceptance rate")
		}
	}
	if _, err := r.Column("missing"); err == nil {
		t.Fatal("expected unknown column error")
	}
	full, err := m.Run(Options{Iterations: o.Iterations * o.Thin, BurnIn: o.BurnIn, Seed: o.Seed})
	if err != nil {
		t.Fatal(err)
	}
	for i := range r.Draws[0][0] {
		if r.Draws[0][0][i] != full.Draws[0][0][(i+1)*o.Thin-1] {
			t.Fatal("thinning selected wrong draw")
		}
	}
}

func TestDrawLayoutAndColumnView(t *testing.T) {
	m := &Model{Variables: []*variable.Variable{
		{Name: "x", Value: 0, Prior: distributions.Normal{Sigma: 1}, ProposalScale: 1},
		{Name: "y", Value: 0, Prior: distributions.Normal{Sigma: 1}, ProposalScale: 1},
	}, LogLik: func(map[string]float64) float64 { return 0 }}
	r, err := m.Run(Options{Iterations: 20, Chains: 2, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	col, err := r.Column("y")
	if err != nil {
		t.Fatal(err)
	}
	for c := range r.Draws {
		if len(r.Draws[c]) != 2 || len(col[c]) != 20 {
			t.Fatal("incorrect draw dimensions")
		}
		col[c][0] = 123
		if r.Draws[c][1][0] != 123 {
			t.Fatal("column is not a view")
		}
		if cap(r.Draws[c][0]) != 20 {
			t.Fatal("variable slice exposes adjacent storage")
		}
	}
}

func TestJointAdaptationStopsAfterBurnIn(t *testing.T) {
	m, v := simpleModel()
	o := Options{Iterations: 50, BurnIn: 200, Seed: testSeed, Adapt: true}
	r, err := m.Run(o)
	if err != nil {
		t.Fatal(err)
	}
	if r.Scales[0][0] == v.ProposalScale {
		t.Fatal("scale did not adapt")
	}
	o.Iterations = 500
	longer, err := m.Run(o)
	if err != nil {
		t.Fatal(err)
	}
	if r.Scales[0][0] != longer.Scales[0][0] {
		t.Fatal("scale changed after burn-in")
	}
	if v.ProposalScale != 1 {
		t.Fatal("model scale was modified")
	}
	o.TargetAccept = 1
	if _, err := m.Run(o); err == nil {
		t.Fatal("expected invalid target error")
	}
}

func TestChainDiagnostics(t *testing.T) {
	v := &variable.Variable{Name: "x", Value: 0.5, Prior: distributions.Uniform{A: 0, B: 1}, ProposalScale: 2}
	m := &Model{Variables: []*variable.Variable{v}, LogLik: func(p map[string]float64) float64 {
		if p["x"] < 0.25 {
			return math.NaN()
		}
		if p["x"] > 0.75 {
			return math.Inf(-1)
		}
		return 0
	}}
	r, err := m.Run(Options{Iterations: 500, BurnIn: 100, Thin: 2, Seed: testSeed})
	if err != nil {
		t.Fatal(err)
	}
	d := r.Diagnostics[0]
	if d.Proposed != 1000 || d.Accepted+d.RejectedPrior+d.RejectedNonFinite+d.RejectedProbability != d.Proposed {
		t.Fatalf("incorrect diagnostics: %+v", d)
	}
	if d.Accepted == 0 || d.RejectedPrior == 0 || d.RejectedNonFinite == 0 || d.RejectedProbability == 0 {
		t.Fatalf("expected each decision category: %+v", d)
	}
}

func TestInitialStateValidatedBeforeChains(t *testing.T) {
	var calls atomic.Int64
	v := &variable.Variable{Name: "x", Value: 0, Prior: distributions.Normal{Sigma: 1}, ProposalScale: 1}
	m := &Model{Variables: []*variable.Variable{v}, LogLik: func(map[string]float64) float64 {
		calls.Add(1)
		return math.Inf(-1)
	}}
	if _, err := m.Run(Options{Iterations: 10, Chains: 4}); err == nil {
		t.Fatal("expected invalid initial state")
	}
	if calls.Load() != 1 {
		t.Fatalf("likelihood called %d times, want 1", calls.Load())
	}
}

func TestNoLogLikCallOutsidePriorSupport(t *testing.T) {
	theta := &variable.Variable{Name: "theta", Value: 0.5, Prior: distributions.Uniform{A: 0, B: 1}, ProposalScale: 2}
	outside := 0
	m := &Model{
		Variables: []*variable.Variable{theta},
		LogLik: func(p map[string]float64) float64 {
			if th := p["theta"]; th < 0 || th > 1 {
				outside++
			}
			return 0
		},
	}
	if _, err := m.Run(Options{Iterations: 2000, Seed: testSeed}); err != nil {
		t.Fatal(err)
	}
	if outside > 0 {
		t.Errorf("LogLik called %d times with theta outside the prior support", outside)
	}
}

func TestRunArgumentErrors(t *testing.T) {
	m, _ := simpleModel()
	for _, c := range []struct{ iters, burn int }{{0, 0}, {-1, 0}, {10, -1}} {
		if _, err := m.Run(Options{Iterations: c.iters, BurnIn: c.burn}); err == nil {
			t.Errorf("Run(%d, %d): want error, got nil", c.iters, c.burn)
		}
	}
}
