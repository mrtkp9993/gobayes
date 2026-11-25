package gobayes

import (
	"github.com/mrtkp9993/gobayes/distributions"
	"github.com/mrtkp9993/gobayes/model"
	"github.com/mrtkp9993/gobayes/variable"
)

type (
	Model    = model.Model
	Options  = model.Options
	Result   = model.Result
	Variable = variable.Variable
)

type (
	Distribution           = distributions.Distribution
	ContinuousDistribution = distributions.ContinuousDistribution
	DiscreteDistribution   = distributions.DiscreteDistribution
	Binomial               = distributions.Binomial
	NegativeBinomial       = distributions.NegativeBinomial
	Categorical            = distributions.Categorical
	OrderedLogistic        = distributions.OrderedLogistic
	OrderedProbit          = distributions.OrderedProbit
	Exponential            = distributions.Exponential
	Gamma                  = distributions.Gamma
	Normal                 = distributions.Normal
	Uniform                = distributions.Uniform
)
