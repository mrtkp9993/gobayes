package distributions

type Distribution interface {
	LogProb(x float64) float64
	Sample() float64
}

type ContinuousDistribution interface {
	Distribution
	IsContinuous()
}

type DiscreteDistribution interface {
	Distribution
	IsDiscrete()
}

var (
	_ ContinuousDistribution = Normal{}
	_ ContinuousDistribution = Uniform{}
	_ ContinuousDistribution = Gamma{}
	_ ContinuousDistribution = Exponential{}
	_ DiscreteDistribution   = Binomial{}
	_ DiscreteDistribution   = Bernoulli{}
	_ DiscreteDistribution   = Poisson{}
	_ DiscreteDistribution   = NegativeBinomial{}
	_ DiscreteDistribution   = Categorical{}
	_ DiscreteDistribution   = OrderedLogistic{}
	_ DiscreteDistribution   = OrderedProbit{}
)
