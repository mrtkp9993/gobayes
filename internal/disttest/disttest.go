package disttest

type Dist interface {
	LogProb(x float64) float64
	Sample() float64
}

const (
	NSamples = 200_000
	SETol    = 5.0
)
