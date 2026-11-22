package model

import (
	"github.com/mrtkp9993/gobayes/diagnostics"
	"github.com/mrtkp9993/gobayes/distributions"
	"github.com/mrtkp9993/gobayes/random"
	"math"
)

func (m *Model) runChain(o Options, rng *random.Generator, r *Result, c int, curr Proposal) error {
	r.Draws[c] = make([][]float64, len(m.Variables))
	backing := make([]float64, len(m.Variables)*o.Iterations)
	for i := range r.Draws[c] {
		r.Draws[c][i] = backing[i*o.Iterations : (i+1)*o.Iterations : (i+1)*o.Iterations]
	}
	proposal := Proposal{Params: make(map[string]float64, len(m.Variables))}
	r.Scales[c] = make([]float64, len(m.Variables))
	for i, v := range m.Variables {
		r.Scales[c][i] = v.ProposalScale
		if _, discrete := v.Prior.(distributions.DiscreteDistribution); discrete && v.ProposalScale == 0 {
			r.Scales[c][i] = 1
		}
	}
	if o.Componentwise {
		r.Diagnostics.Components[c] = make([]diagnostics.ChainDiagnostics, len(m.Variables))
	}
	for step := 0; step < o.BurnIn+o.Iterations*o.Thin; step++ {
		if o.Componentwise {
			for i := range m.Variables {
				m.proposeComponent(curr, rng, r.Scales[c][i], i, &proposal)
				accepted := m.accept(&proposal, curr, rng)
				if step >= o.BurnIn {
					d := &r.Diagnostics.Chains[c]
					component := &r.Diagnostics.Components[c][i]
					d.Proposed++
					component.Proposed++
					if accepted {
						d.Accepted++
						component.Accepted++
					}
				}
				if accepted {
					curr, proposal = proposal, curr
				}
				_, discrete := m.Variables[i].Prior.(distributions.DiscreteDistribution)
				if o.Adapt && step < o.BurnIn && !discrete {
					r.Scales[c][i] = adaptScale(r.Scales[c][i], accepted, step, o.TargetAccept)
				}
			}
		} else {
			m.proposeWithScales(curr, rng, r.Scales[c], &proposal)
			acceptedProposal := m.accept(&proposal, curr, rng)
			if step >= o.BurnIn {
				d := &r.Diagnostics.Chains[c]
				d.Proposed++
				if acceptedProposal {
					d.Accepted++
				}
			}
			if acceptedProposal {
				curr, proposal = proposal, curr
			}
			if o.Adapt && step < o.BurnIn {
				for i, scale := range r.Scales[c] {
					if _, discrete := m.Variables[i].Prior.(distributions.DiscreteDistribution); discrete {
						continue
					}
					r.Scales[c][i] = adaptScale(scale, acceptedProposal, step, o.TargetAccept)
				}
			}
		}
		if step >= o.BurnIn && (step-o.BurnIn+1)%o.Thin == 0 {
			drawIndex := (step-o.BurnIn+1)/o.Thin - 1
			for i, name := range r.Names {
				r.Draws[c][i][drawIndex] = curr.Params[name]
			}
		}
	}
	return nil
}

func adaptScale(scale float64, accepted bool, step int, target float64) float64 {
	indicator := 0.0
	if accepted {
		indicator = 1
	}
	gain := 1 / math.Sqrt(float64(step+1))
	logScale := math.Log(scale) + gain*(indicator-target)
	return math.Exp(math.Max(-20, math.Min(20, logScale)))
}
