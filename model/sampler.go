package model

import (
	"math"

	"github.com/mrtkp9993/gobayes/random"
)

func (m *Model) initialState() Proposal {
	s := Proposal{Params: make(map[string]float64, len(m.Variables))}
	for _, v := range m.Variables {
		s.Params[v.Name] = v.Value
		s.LogPrior += v.Prior.LogProb(v.Value)
	}
	if !math.IsNaN(s.LogPrior) && !math.IsInf(s.LogPrior, 0) {
		s.LogLik = m.LogLik(s.Params)
	}
	return s
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
