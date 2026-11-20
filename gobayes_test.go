package gobayes_test

import (
	"testing"

	"github.com/mrtkp9993/gobayes"
)

func TestRunFromRootPackage(t *testing.T) {
	var prior gobayes.Distribution = gobayes.Normal{Sigma: 1}
	m := gobayes.Model{
		Variables: []*gobayes.Variable{{Name: "x", Prior: prior, ProposalScale: 1}},
		LogLik:    func(map[string]float64) float64 { return 0 },
	}
	const iterations = 10
	var result *gobayes.Result
	result, err := m.Run(gobayes.Options{Iterations: iterations, Seed: 42})
	if err != nil {
		t.Fatal(err)
	}
	chains, err := result.Column("x")
	if err != nil {
		t.Fatal(err)
	}
	if len(chains) != 1 || len(chains[0]) != iterations {
		t.Fatalf("expected one chain with %d draws, got %v", iterations, chains)
	}
}
