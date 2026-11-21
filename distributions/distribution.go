package distributions

type Distribution interface {
	LogProb(x float64) float64
	Sample() float64
}

type ContinuousDistribution interface {
	Distribution
	IsContinuous()
}

var (
	_ ContinuousDistribution = Normal{}
	_ ContinuousDistribution = Uniform{}
	_ ContinuousDistribution = Gamma{}
	_ ContinuousDistribution = Exponential{}
)
