package model

import (
	"errors"
	"math"

	"github.com/mrtkp9993/gobayes/random"
	"github.com/mrtkp9993/gobayes/variable"
)

type Model struct {
	Variables []*variable.Variable
	LogLik    func(params map[string]float64) float64
	Seed      uint64
}

func (m *Model) initialState() Proposal {
	s := Proposal{Params: make(map[string]float64, len(m.Variables))}
	for _, v := range m.Variables {
		s.Params[v.Name] = v.Value
		s.LogPrior += v.Prior.LogProb(v.Value)
	}
	s.LogLik = m.LogLik(s.Params)
	return s
}

func (m *Model) propose() Proposal {
	p := Proposal{Params: make(map[string]float64, len(m.Variables))}
	for _, v := range m.Variables {
		var newVal float64
		if v.Multiplicative {
			factor := math.Exp(v.ProposalScale * random.NormFloat64())
			newVal = v.Value * factor
			p.LogQRatio += math.Log(factor)
		} else {
			newVal = v.Value + v.ProposalScale*random.NormFloat64()
		}
		p.Params[v.Name] = newVal
		p.LogPrior += v.Prior.LogProb(newVal)
	}
	return p
}

func (m *Model) accept(p *Proposal, curr Proposal) bool {
	if math.IsInf(p.LogPrior, -1) || math.IsNaN(p.LogPrior) {
		return false
	}
	p.LogLik = m.LogLik(p.Params)
	logA := p.LogPosterior() - curr.LogPosterior() + p.LogQRatio
	if math.IsNaN(logA) {
		return false
	}
	return logA >= 0 || math.Log(random.Float64()) < logA
}

func (m *Model) RunMCMC(iterations, burnIn int) ([]map[string]float64, error) {
	if iterations <= 0 {
		return nil, errors.New("iterations must be positive")
	}
	if burnIn < 0 {
		return nil, errors.New("burn-in period cannot be negative")
	}
	if m.Seed != 0 {
		random.Seed(m.Seed)
	}

	curr := m.initialState()
	if lp := curr.LogPosterior(); math.IsInf(lp, -1) || math.IsNaN(lp) {
		return nil, errors.New("initial values have zero posterior density (log-posterior is -Inf or NaN)")
	}

	samples := make([]map[string]float64, iterations)
	for i := 0; i < iterations+burnIn; i++ {
		prop := m.propose()
		if m.accept(&prop, curr) {
			for _, v := range m.Variables {
				v.Value = prop.Params[v.Name]
			}
			curr = prop
		}

		if i >= burnIn {
			samples[i-burnIn] = make(map[string]float64)
			for _, v := range m.Variables {
				samples[i-burnIn][v.Name] = v.Value
			}
		}
	}
	return samples, nil
}
