package distributions

import (
	"fmt"
	"math"
	"testing"

	"github.com/mrtkp9993/gobayes/internal/disttest"
	"github.com/mrtkp9993/gobayes/random"
)

func TestGammaLogProb(t *testing.T) {
	t.Run("extreme finite parameters", func(t *testing.T) {
		tiny := math.SmallestNonzeroFloat64
		logTiny := -1074 * math.Ln2
		logFactorial := 0.0
		for k := 2; k <= 15; k++ {
			logFactorial += math.Log(float64(k))
		}
		for _, tc := range []struct {
			d       Gamma
			x, want float64
		}{
			{Gamma{1, tiny}, 0, logTiny},
			{Gamma{tiny, 1}, 1, logTiny - 1},
			{Gamma{16, tiny}, tiny, 31*logTiny - logFactorial},
			{Gamma{1e308, 1e308}, 2, (math.Ln2 - 1) * 1e308},
		} {
			got := tc.d.LogProb(tc.x)
			if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-tc.want) > 1e-13*math.Max(1, math.Abs(tc.want)) {
				t.Errorf("%+v x=%g: got %g, want %g", tc.d, tc.x, got, tc.want)
			}
		}
	})
	t.Run("alpha=3,beta=2", func(t *testing.T) {
		disttest.RunLogProbCases(t, Gamma{Alpha: 3, Beta: 2}, []disttest.LogProbCase{
			{Name: "x=2", X: 2, Want: -1.2274112777602189},
			{Name: "x=0", X: 0, Want: math.Inf(-1)},
			{Name: "x<0", X: -1, Want: math.Inf(-1)},
		})
	})
	t.Run("alpha=0.5,beta=1.5", func(t *testing.T) {
		disttest.RunLogProbCases(t, Gamma{Alpha: 0.5, Beta: 1.5}, []disttest.LogProbCase{
			{Name: "x=0.1", X: 0.1, Want: 0.6316601576264049},
		})
	})
	t.Run("alpha=10,beta=1", func(t *testing.T) {
		disttest.RunLogProbCases(t, Gamma{Alpha: 10, Beta: 1}, []disttest.LogProbCase{
			{Name: "x=7.5", X: 7.5, Want: -2.1677002952010884},
		})
	})
	t.Run("x=0 boundary (scipy)", func(t *testing.T) {
		disttest.AssertClose(t, Gamma{Alpha: 1, Beta: 2}.LogProb(0), 0.6931471805599453, "alpha=1")
		disttest.AssertClose(t, Gamma{Alpha: 0.5, Beta: 1.5}.LogProb(0), math.Inf(1), "alpha<1")
		disttest.AssertClose(t, Gamma{Alpha: 3, Beta: 2}.LogProb(0), math.Inf(-1), "alpha>1")
	})
	t.Run("alpha=1 matches Exponential", func(t *testing.T) {
		g, e := Gamma{Alpha: 1, Beta: 2}, Exponential{Lambda: 2}
		for _, x := range []float64{0, 0.1, 1.3, 5} {
			disttest.AssertClose(t, g.LogProb(x), e.LogProb(x), fmt.Sprintf("x=%v", x))
		}
	})
}

func TestGammaLogProbInvalidParams(t *testing.T) {
	for _, p := range []Gamma{{0, 1}, {-0.5, 1}, {2, 0}, {2, -1}} {
		t.Run(fmt.Sprintf("alpha=%v,beta=%v", p.Alpha, p.Beta), func(t *testing.T) {
			disttest.AssertClose(t, p.LogProb(1), math.Inf(-1), "LogProb")
		})
	}
}

func TestGammaLogProbLargeShapeNearMean(t *testing.T) {
	// mpmath, 100 decimal digits, exact float64 inputs.
	for _, tc := range []struct {
		alpha, beta, x, want float64
	}{
		{1e16, 1, 1e16, -19.339619277157038222},
		{1e16, 1, 1e16 - 1e8, -19.839619270490371531},
		{1e16, 1, 1e16 + 1e8, -19.839619283823704864},
		{1e16, 1.1, 9090909181818182, -19.744309114093729325},
		{16, 2, 7.2, -1.5977011045516221952},
		{16, 2, 8, -1.6172933696842276916},
		{16, 2, 8.8, -1.7876406726193550009},
	} {
		d := Gamma{Alpha: tc.alpha, Beta: tc.beta}
		got := d.LogProb(tc.x)
		if math.IsNaN(got) || math.IsInf(got, 0) || math.Abs(got-tc.want) > 1e-12 {
			t.Errorf("%+v x=%g: got %.17g, want %.17g", d, tc.x, got, tc.want)
		}
	}
}

func TestGammaSample(t *testing.T) {
	const draws = 200
	random.Seed(42)
	unit := make([]float64, draws)
	for i := range unit {
		unit[i] = (Gamma{Alpha: 0.001, Beta: 1}).Sample()
	}
	random.Seed(42)
	recovered := 0
	for _, unscaled := range unit {
		scaled := (Gamma{Alpha: 0.001, Beta: 1e-310}).Sample()
		if math.IsNaN(scaled) || scaled < 0 {
			t.Fatalf("invalid scaled draw: %g", scaled)
		}
		if unscaled == 0 && scaled > 0 && !math.IsInf(scaled, 0) {
			recovered++
		}
	}
	if recovered == 0 {
		t.Fatal("expected finite scaled draws recovered from unit-rate underflow")
	}
	cases := []struct {
		d           Gamma
		median, mad float64
	}{
		{Gamma{0.3, 1}, 0.07313113586695198, 0.07274764230770486},
		{Gamma{0.5, 2}, 0.113734105779893, 0.10477356970720476},
		{Gamma{1, 1}, 0.6931471805599455, 0.48121182505960364},
		{Gamma{2.5, 0.5}, 4.351460191095526, 1.8947227758858323},
		{Gamma{10, 3}, 3.2229048715713757, 0.691530984840641},
	}
	for _, c := range cases {
		d := c.d
		t.Run(fmt.Sprintf("alpha=%v,beta=%v", d.Alpha, d.Beta), func(t *testing.T) {
			xs := disttest.DrawSamples(t, d, disttest.NSamples, func(x float64) bool { return x >= 0 })
			a, b := d.Alpha, d.Beta
			disttest.CheckStats(t, d, xs, disttest.TheoreticalStats{
				Mean: a / b, Median: c.median, Var: a / (b * b),
				MAD:  c.mad,
				Skew: 2 / math.Sqrt(a), ExKurt: 6 / a,
			})
		})
	}
}

func TestGammaIntegratesToOne(t *testing.T) {
	for _, d := range []Gamma{{0.3, 1}, {0.5, 2}, {1, 1}, {2.5, 0.5}, {10, 3}, {100, 0.1}} {
		t.Run(fmt.Sprintf("alpha=%v,beta=%v", d.Alpha, d.Beta), func(t *testing.T) {
			disttest.CheckIntegral(t, disttest.IntegrateHalfLine(t, d, 0))
		})
	}
}

func TestGammaSampleInvalidParamsPanics(t *testing.T) {
	for _, p := range []Gamma{{0, 1}, {-1, 1}, {1, 0}, {1, -2}} {
		disttest.AssertPanics(t, func() { p.Sample() }, fmt.Sprintf("alpha=%v,beta=%v", p.Alpha, p.Beta))
	}
}
