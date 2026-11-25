package distributions

import (
	"fmt"
	"math"
	"testing"

	"github.com/mrtkp9993/gobayes/internal/disttest"
)

func TestUniformLogProb(t *testing.T) {
	t.Run("A=0,B=1", func(t *testing.T) {
		disttest.RunLogProbCases(t, Uniform{A: 0, B: 1}, []disttest.LogProbCase{
			{Name: "x=0.3", X: 0.3, Want: 0.0},
			{Name: "x<A", X: -0.1, Want: math.Inf(-1)},
			{Name: "x>B", X: 1.1, Want: math.Inf(-1)},
			{Name: "NaN observation", X: math.NaN(), Want: math.Inf(-1)},
		})
	})
	t.Run("A=-3,B=2", func(t *testing.T) {
		disttest.RunLogProbCases(t, Uniform{A: -3, B: 2}, []disttest.LogProbCase{
			{Name: "x=-1", X: -1, Want: -1.6094379124341003},
		})
	})
	t.Run("A=2,B=6 boundaries", func(t *testing.T) {
		disttest.RunLogProbCases(t, Uniform{A: 2, B: 6}, []disttest.LogProbCase{
			{Name: "x=A", X: 2, Want: -1.3862943611198906},
			{Name: "x=B", X: 6, Want: -1.3862943611198906},
		})
	})
}

func TestUniformLogProbInvalidParams(t *testing.T) {
	for _, p := range []Uniform{{1, 1}, {3, 1}, {math.NaN(), 1}, {0, math.NaN()}, {math.Inf(-1), 1}, {0, math.Inf(1)}} {
		t.Run(fmt.Sprintf("A=%v,B=%v", p.A, p.B), func(t *testing.T) {
			for _, x := range []float64{1, 2} {
				disttest.AssertClose(t, p.LogProb(x), math.Inf(-1), fmt.Sprintf("x=%v", x))
			}
		})
	}
}

func TestUniformLogProbExtremeRanges(t *testing.T) {
	for _, tc := range []struct {
		name string
		d    Uniform
		x    float64
		want float64
	}{
		{"overflowing width", Uniform{A: -1e308, B: 1e308}, 0, -308*math.Ln10 - math.Ln2},
		{"large positive bounds", Uniform{A: 5e307, B: 1e308}, 7e307, -308*math.Ln10 + math.Ln2},
		{"large negative bounds", Uniform{A: -1e308, B: -5e307}, -7e307, -308*math.Ln10 + math.Ln2},
		{"smallest width", Uniform{A: 0, B: math.SmallestNonzeroFloat64}, 0, 1074 * math.Ln2},
		{"adjacent large bounds", Uniform{A: 0x1p1023, B: 0x1.0000000000001p1023}, 0x1p1023, -971 * math.Ln2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			disttest.AssertClose(t, tc.d.LogProb(tc.x), tc.want, "LogProb")
		})
	}
}

func TestUniformSampleExtremeRanges(t *testing.T) {
	for _, d := range []Uniform{
		{A: -1e308, B: 1e308},
		{A: 5e307, B: 1e308},
		{A: -1e308, B: -5e307},
		{A: 0, B: math.SmallestNonzeroFloat64},
		{A: 0x1p1023, B: 0x1.0000000000001p1023},
	} {
		disttest.DrawSamples(t, d, 1000, func(x float64) bool {
			return x >= d.A && x <= d.B
		})
	}
	// Scaling to [-1,1] keeps the moment calculations finite.
	d := Uniform{A: -1e308, B: 1e308}
	samples := disttest.DrawSamples(t, d, disttest.NSamples, func(x float64) bool {
		return x >= d.A && x <= d.B
	})
	for i := range samples {
		samples[i] /= 1e308
	}
	disttest.CheckStats(t, Uniform{A: -1, B: 1}, samples, disttest.TheoreticalStats{
		Mean: 0, Median: 0, Var: 1.0 / 3, MAD: 0.5, Skew: 0, ExKurt: -1.2,
	})
}

func TestUniformSample(t *testing.T) {
	for _, d := range []Uniform{{0, 1}, {-3, 2}, {10, 10.5}} {
		t.Run(fmt.Sprintf("A=%v,B=%v", d.A, d.B), func(t *testing.T) {
			xs := disttest.DrawSamples(t, d, disttest.NSamples, func(x float64) bool { return x >= d.A && x <= d.B })
			w := d.B - d.A

			disttest.CheckStats(t, d, xs, disttest.TheoreticalStats{
				Mean: (d.A + d.B) / 2, Median: (d.A + d.B) / 2, Var: w * w / 12,
				MAD:  w / 4,
				Skew: 0, ExKurt: -1.2,
			})
		})
	}
}

func TestUniformIntegratesToOne(t *testing.T) {
	for _, d := range []Uniform{{0, 1}, {-3, 2}, {10, 10.5}} {
		t.Run(fmt.Sprintf("A=%v,B=%v", d.A, d.B), func(t *testing.T) {
			disttest.CheckIntegral(t, disttest.IntegrateFinite(t, d, d.A, d.B))
		})
	}
}

func TestUniformSampleInvalidParamsPanics(t *testing.T) {
	for _, p := range []Uniform{{1, 1}, {3, 1}, {math.NaN(), 1}, {0, math.NaN()}, {math.Inf(-1), 1}, {0, math.Inf(1)}} {
		disttest.AssertPanics(t, func() { p.Sample() }, fmt.Sprintf("A=%v,B=%v", p.A, p.B))
	}
}
