package model

import (
	"fmt"
	"maps"
	"math"
	"slices"

	"github.com/mrtkp9993/gobayes/random"
)

func (m *Model) initialState(rng *random.Generator) Proposal {
	s := Proposal{Params: make(map[string]float64, len(m.Variables))}
	for _, v := range m.Variables {
		value := v.Value
		if rng != nil {
			step := v.ProposalScale * (2*rng.Float64() - 1)
			if v.Multiplicative {
				value *= math.Exp(step)
			} else {
				value += step
			}
		}
		if math.IsNaN(value) || math.IsInf(value, 0) || (v.Multiplicative && value <= 0) {
			s.LogPrior = math.Inf(-1)
			return s
		}
		s.Params[v.Name] = value
		s.LogPrior += v.Prior.LogProb(value)
	}
	if !math.IsNaN(s.LogPrior) && !math.IsInf(s.LogPrior, 0) {
		s.LogLik = m.LogLik(s.Params)
	}
	return s
}

func (m *Model) randomizedInitialState(rng *random.Generator, previous []Proposal) (Proposal, error) {
	const maxAttempts = 1000
	for range maxAttempts {
		candidate := m.initialState(rng)
		if lp := candidate.LogPosterior(); math.IsNaN(lp) || math.IsInf(lp, 0) {
			continue
		}
		if slices.ContainsFunc(previous, func(start Proposal) bool {
			return maps.Equal(start.Params, candidate.Params)
		}) {
			continue
		}
		return candidate, nil
	}
	return Proposal{}, fmt.Errorf("could not find a distinct initial state with finite log-posterior after %d attempts", maxAttempts)
}

func (m *Model) proposeWithScales(curr Proposal, rng *random.Generator, scales []float64, p *Proposal) {
	p.LogPrior, p.LogLik, p.LogQRatio = 0, 0, 0
	for i, v := range m.Variables {
		scale := v.ProposalScale
		if scales != nil {
			scale = scales[i]
		}
		var newVal float64
		if v.Multiplicative {
			factor := math.Exp(scale * rng.NormFloat64())
			newVal = curr.Params[v.Name] * factor
			p.LogQRatio += math.Log(factor)
		} else {
			newVal = curr.Params[v.Name] + scale*rng.NormFloat64()
		}
		p.Params[v.Name] = newVal
		p.LogPrior += v.Prior.LogProb(newVal)
	}
}

func (m *Model) proposeComponent(curr Proposal, rng *random.Generator, scale float64, i int, p *Proposal) {
	for name, value := range curr.Params {
		p.Params[name] = value
	}
	v := m.Variables[i]
	old := curr.Params[v.Name]
	step := scale * rng.NormFloat64()
	value := old + step
	p.LogQRatio = 0
	if v.Multiplicative {
		value = old * math.Exp(step)
		p.LogQRatio = step
	}
	p.Params[v.Name] = value
	p.LogPrior = 0
	for _, variable := range m.Variables {
		p.LogPrior += variable.Prior.LogProb(p.Params[variable.Name])
	}
	p.LogLik = 0
}

func (m *Model) accept(p *Proposal, curr Proposal, rng *random.Generator) bool {
	if math.IsInf(p.LogPrior, -1) {
		return false
	}
	if math.IsNaN(p.LogPrior) || math.IsInf(p.LogPrior, 1) {
		return false
	}
	p.LogLik = m.LogLik(p.Params)
	if math.IsNaN(p.LogLik) || math.IsInf(p.LogLik, 1) || math.IsNaN(p.LogQRatio) || math.IsInf(p.LogQRatio, 0) {
		return false
	}
	if math.IsInf(p.LogLik, -1) {
		return false
	}
	logA := p.LogPosterior() - curr.LogPosterior() + p.LogQRatio
	if math.IsNaN(logA) {
		return false
	}
	if logA >= 0 || math.Log(rng.Float64()) < logA {
		return true
	}
	return false
}
