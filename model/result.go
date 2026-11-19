package model

import (
	"fmt"

	"github.com/mrtkp9993/gobayes/diagnostics"
)

type Result struct {
	Names       []string
	Draws       [][][]float64 // [chain][variable][draw]
	Scales      [][]float64
	Seed        uint64
	Diagnostics []diagnostics.ChainDiagnostics
}

func (r *Result) AcceptRate() []float64 {
	rates := make([]float64, len(r.Diagnostics))
	for i, d := range r.Diagnostics {
		if d.Proposed > 0 {
			rates[i] = float64(d.Accepted) / float64(d.Proposed)
		}
	}
	return rates
}

func (r *Result) ParameterDiagnostics(name string) (diagnostics.ParameterDiagnostics, error) {
	chains, err := r.Column(name)
	if err != nil {
		return diagnostics.ParameterDiagnostics{}, err
	}
	return diagnostics.Compute(chains)
}

func (r *Result) Column(name string) ([][]float64, error) {
	index := -1
	for i, n := range r.Names {
		if n == name {
			index = i
			break
		}
	}
	if index < 0 {
		return nil, fmt.Errorf("unknown variable %q", name)
	}
	out := make([][]float64, len(r.Draws))
	for c, chain := range r.Draws {
		out[c] = chain[index]
	}
	return out, nil
}
