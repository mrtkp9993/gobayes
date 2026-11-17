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
	for _, p := range []Uniform{{1, 1}, {3, 1}} {
		t.Run(fmt.Sprintf("A=%v,B=%v", p.A, p.B), func(t *testing.T) {
			for _, x := range []float64{1, 2} {
				disttest.AssertClose(t, p.LogProb(x), math.Inf(-1), fmt.Sprintf("x=%v", x))
			}
		})
	}
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
	for _, p := range []Uniform{{1, 1}, {3, 1}} {
		disttest.AssertPanics(t, func() { p.Sample() }, fmt.Sprintf("A=%v,B=%v", p.A, p.B))
	}
}
