package distributions

type Distribution interface {
	LogProb(x float64) float64
	Sample() float64
}
