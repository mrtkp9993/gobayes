package model

import (
	"errors"
	"fmt"
	"github.com/mrtkp9993/gobayes/diagnostics"
	"github.com/mrtkp9993/gobayes/random"
	"math"
	"sync"
)

func (m *Model) Run(o Options) (*Result, error) {
	if o.Thin == 0 {
		o.Thin = 1
	}
	if o.Chains == 0 {
		o.Chains = 1
	}
	if err := m.validate(o); err != nil {
		return nil, err
	}
	initial := m.initialState()
	if lp := initial.LogPosterior(); math.IsNaN(lp) || math.IsInf(lp, 0) {
		return nil, errors.New("initial log-posterior must be finite")
	}
	if o.TargetAccept == 0 {
		if o.Componentwise {
			o.TargetAccept = 0.44
		} else {
			o.TargetAccept = 0.234
		}
	}
	names := make([]string, len(m.Variables))
	for i, v := range m.Variables {
		names[i] = v.Name
	}
	seed := o.Seed
	if seed == 0 {
		seed = random.Uint64()
	}
	r := &Result{Names: names, Seed: seed, Draws: make([][][]float64, o.Chains), Scales: make([][]float64, o.Chains), Diagnostics: &diagnostics.Diagnostics{Chains: make([]diagnostics.ChainDiagnostics, o.Chains)}}
	if o.Componentwise {
		r.Diagnostics.Components = make([][]diagnostics.ChainDiagnostics, o.Chains)
	}
	errs := make([]error, o.Chains)
	var wg sync.WaitGroup
	for c := 0; c < o.Chains; c++ {
		wg.Add(1)
		go func(c int) {
			defer wg.Done()
			curr := initial
			curr.Params = make(map[string]float64, len(initial.Params))
			for name, value := range initial.Params {
				curr.Params[name] = value
			}
			errs[c] = m.runChain(o, random.New(seed, uint64(c)), r, c, curr)
		}(c)
	}
	wg.Wait()
	for c, err := range errs {
		if err != nil {
			return nil, fmt.Errorf("chain %d: %w", c, err)
		}
	}
	return r, nil
}
