package model

import (
	"math"
	"testing"

	"github.com/mrtkp9993/gobayes/distributions"
	"github.com/mrtkp9993/gobayes/variable"
)

const (
	testIters  = 50_000
	testBurnIn = 2_000
	testSeed   = 20261118
	nBatches   = 50
	mcseTol    = 5.0
)

func draws(t *testing.T, m *Model, name string) []float64 {
	t.Helper()
	samples, err := m.RunMCMC(testIters, testBurnIn)
	if err != nil {
		t.Fatalf("RunMCMC: %v", err)
	}
	xs := make([]float64, len(samples))
	for i, s := range samples {
		xs[i] = s[name]
	}
	return xs
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
		Seed: testSeed,
	}
	a, b := 1+k, 1+n-k
	checkPosterior(t, draws(t, m, "theta"), a/(a+b), a*b/((a+b)*(a+b)*(a+b+1)))
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
		Seed: testSeed,
	}
	a, b := 4.0, 3309.0
	checkPosterior(t, draws(t, m, "lambda"), a/b, a/(b*b))
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
		Seed: testSeed,
	}
	var sum float64
	for _, x := range data {
		sum += x
	}
	prec := 1/(tau0*tau0) + float64(len(data))/(sigma*sigma)
	postMean := (mu0/(tau0*tau0) + sum/(sigma*sigma)) / prec
	checkPosterior(t, draws(t, m, "mu"), postMean, 1/prec)
}

func simpleModel(seed uint64) (*Model, *variable.Variable) {
	v := &variable.Variable{Name: "mu", Value: 0, Prior: distributions.Normal{Mu: 0, Sigma: 10}, ProposalScale: 1}
	return &Model{
		Variables: []*variable.Variable{v},
		LogLik:    func(p map[string]float64) float64 { return distributions.Normal{Mu: p["mu"], Sigma: 1}.LogProb(2) },
		Seed:      seed,
	}, v
}

func TestSeedReproducible(t *testing.T) {
	run := func(seed uint64) []float64 {
		m, _ := simpleModel(seed)
		return draws(t, m, "mu")
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
		Seed: testSeed,
	}
	if _, err := m.RunMCMC(2000, 0); err != nil {
		t.Fatal(err)
	}
	if outside > 0 {
		t.Errorf("LogLik called %d times with theta outside the prior support", outside)
	}
}

func TestRunMCMCArgumentErrors(t *testing.T) {
	m, _ := simpleModel(testSeed)
	for _, c := range []struct{ iters, burn int }{{0, 0}, {-1, 0}, {10, -1}} {
		if _, err := m.RunMCMC(c.iters, c.burn); err == nil {
			t.Errorf("RunMCMC(%d, %d): want error, got nil", c.iters, c.burn)
		}
	}
}
