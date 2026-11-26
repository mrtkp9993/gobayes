package distributions

import (
	"math"

	"github.com/mrtkp9993/gobayes/internal/utils"
	"github.com/mrtkp9993/gobayes/random"
)

type Gamma struct {
	Alpha float64
	Beta  float64
}

func (Gamma) IsContinuous() {}

func (g Gamma) LogProb(x float64) float64 {
	if !(g.Alpha > 0) || math.IsInf(g.Alpha, 0) || !(g.Beta > 0) || math.IsInf(g.Beta, 0) || !(x >= 0) || math.IsInf(x, 1) {
		return math.Inf(-1)
	}
	if x == 0 {
		switch {
		case g.Alpha < 1:
			return math.Inf(1)
		case g.Alpha == 1:
			return utils.LogPositive(g.Beta)
		default:
			return math.Inf(-1)
		}
	}

	if g.Alpha >= 16 {
		base := 0.5*math.Log(g.Alpha) - utils.LogPositive(x) -
			0.5*math.Log(2*math.Pi) - stirlingError(g.Alpha)

		d := math.FMA(g.Beta, x, -g.Alpha) / g.Alpha
		if math.Abs(d) <= 0.1 {
			term := -0.5 * d * d
			sum := term
			for k := 3.0; ; k++ {
				term *= -d * (k - 1) / k
				next := sum + term
				if next == sum {
					break
				}
				sum = next
			}
			return math.FMA(g.Alpha, sum, base)
		}

		mb, eb := math.Frexp(g.Beta)
		mx, ex := math.Frexp(x)
		ma, ea := math.Frexp(g.Alpha)
		mantissa := mb * mx / ma
		exponent := eb + ex - ea

		r := math.Ldexp(mantissa, exponent)
		if math.IsInf(r, 1) {
			return math.Inf(-1)
		}

		var logR float64
		if r >= 0x1p-1022 {
			logR = math.Log(r)
		} else {
			logR = math.Log(mantissa) + float64(exponent)*math.Ln2
		}
		return math.FMA(g.Alpha, logR-(r-1), base)
	}

	var lgamma float64
	if g.Alpha < 0.5 {
		lgamma, _ = math.Lgamma(1 + g.Alpha)
		lgamma -= utils.LogPositive(g.Alpha)
	} else {
		lgamma, _ = math.Lgamma(g.Alpha)
	}

	return (g.Alpha-1)*utils.LogPositive(x) +
		g.Alpha*utils.LogPositive(g.Beta) - g.Beta*x - lgamma
}

func (g Gamma) Sample() float64 {
	alpha := g.Alpha
	beta := g.Beta
	if !(alpha > 0) || math.IsInf(alpha, 0) || !(beta > 0) || math.IsInf(beta, 0) {
		panic("gamma parameters must be finite and positive")
	}
	if alpha < 1 {
		y := Gamma{Alpha: alpha + 1, Beta: 1}.Sample()
		u := random.Float64()
		if u == 0 {
			return 0
		}

		logX := math.Log(y) + utils.LogPositive(u)/alpha -
			utils.LogPositive(beta)

		if logX < -746 {
			return 0
		}
		if logX > 710 {
			return math.Inf(1)
		}

		k := int(math.Round(logX / math.Ln2))
		remainder := math.FMA(-float64(k), math.Ln2, logX)
		return math.Ldexp(math.Exp(remainder), k)
	}

	d := alpha - 1.0/3.0
	c := 1.0 / math.Sqrt(9.0*d)
	for {
		x := random.NormFloat64()
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := random.Float64()

		if u < 1-0.0331*math.Pow(x, 4) {
			return d * v / beta
		}
		if math.Log(u) < 0.5*x*x+d*(1-v+math.Log(v)) {
			return d * v / beta
		}
	}
}
