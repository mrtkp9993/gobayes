package distributions

import (
	"math"

	"github.com/mrtkp9993/gobayes/random"
)

type Binomial struct {
	N int
	P float64
}

func (Binomial) IsDiscrete() {}

func (b Binomial) valid() bool {
	return b.N >= 0 && b.N <= 1<<53 && b.P >= 0 && b.P <= 1
}

// https://www.r-project.org/doc/reports/CLoader-dbinom-2002.pdf
func (b Binomial) LogProb(x float64) float64 {
	n := float64(b.N)
	if !b.valid() || x < 0 || x > n || math.Trunc(x) != x {
		return math.Inf(-1)
	}
	switch {
	case n == 0, b.P == 0 && x == 0, b.P == 1 && x == n:
		return 0
	case b.P == 0, b.P == 1:
		return math.Inf(-1)
	case x == 0:
		return n * math.Log1p(-b.P)
	case x == n:
		return n * math.Log(b.P)
	}

	y := n - x
	logCorrection := stirlingError(n) - stirlingError(x) - stirlingError(y)
	deviance := binomialDeviance(x, n*b.P) + binomialDeviance(y, n*(1-b.P))
	logScale := math.Log(2 * math.Pi * min(x, y) * (max(x, y) / n))
	return logCorrection - deviance - 0.5*logScale
}

func stirlingError(n float64) float64 {
	if n < 16 {
		logFactorial, _ := math.Lgamma(n + 1)
		return logFactorial - ((n+0.5)*math.Log(n) - n + 0.5*math.Log(2*math.Pi))
	}
	inverseSquare := 1 / (n * n)
	return (1.0/12 - inverseSquare*(1.0/360-inverseSquare*(1.0/1260-
		inverseSquare*(1.0/1680-inverseSquare/1188)))) / n
}

func binomialDeviance(x, mean float64) float64 {
	difference := x - mean
	if math.Abs(difference) >= 0.1*(x+mean) {
		return x*(math.Log(x)-math.Log(mean)) - difference
	}

	v := difference / (x + mean)
	sum := difference * v
	term := 2 * x * v
	for denominator := 3.0; ; denominator += 2 {
		term *= v * v
		next := sum + term/denominator
		if next == sum {
			return sum
		}
		sum = next
	}
}

func (b Binomial) Sample() float64 {
	if !b.valid() {
		panic("binomial requires 0 <= N <= 2^53 and 0 <= P <= 1")
	}
	if b.N == 0 || b.P == 0 {
		return 0
	}
	if b.P == 1 {
		return float64(b.N)
	}
	logMiss := math.Log1p(-min(b.P, 1-b.P))
	count := 0
	for remaining := b.N; remaining > 0; {
		skipped := math.Floor(math.Log(random.Float64()) / logMiss)
		if skipped >= float64(remaining) {
			break
		}
		remaining -= int(skipped) + 1
		count++
	}
	if b.P > 0.5 {
		return float64(b.N - count)
	}
	return float64(count)
}
