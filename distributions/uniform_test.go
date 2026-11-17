package distributions

import (
	"fmt"
	"math"
	"testing"
)

func TestUniformLogProb(t *testing.T) {
	t.Run("A=0,B=1", func(t *testing.T) {
		runLogProbCases(t, Uniform{A: 0, B: 1}, []logProbCase{
			{"x=0.3", 0.3, 0.0},
			{"x<A", -0.1, math.Inf(-1)},
			{"x>B", 1.1, math.Inf(-1)},
		})
	})
	t.Run("A=-3,B=2", func(t *testing.T) {
		runLogProbCases(t, Uniform{A: -3, B: 2}, []logProbCase{
			{"x=-1", -1, -1.6094379124341003},
		})
	})
	t.Run("A=2,B=6 boundaries", func(t *testing.T) {
		// scipy.stats.uniform kapalı aralık [A, B] üzerinde yoğunluk veriyor.
		runLogProbCases(t, Uniform{A: 2, B: 6}, []logProbCase{
			{"x=A", 2, -1.3862943611198906},
			{"x=B", 6, -1.3862943611198906},
		})
	})
}

func TestUniformLogProbInvalidParams(t *testing.T) {
	for _, p := range []Uniform{{1, 1}, {3, 1}} {
		t.Run(fmt.Sprintf("A=%v,B=%v", p.A, p.B), func(t *testing.T) {
			for _, x := range []float64{1, 2} {
				assertClose(t, p.LogProb(x), math.Inf(-1), fmt.Sprintf("x=%v", x))
			}
		})
	}
}

func TestUniformSample(t *testing.T) {
	for _, d := range []Uniform{{0, 1}, {-3, 2}, {10, 10.5}} {
		t.Run(fmt.Sprintf("A=%v,B=%v", d.A, d.B), func(t *testing.T) {
			xs := drawSamples(t, d, nSamples, func(x float64) bool { return x >= d.A && x <= d.B })
			w := d.B - d.A
			checkMoments(t, xs, (d.A+d.B)/2, w*w/12)
		})
	}
}

func TestUniformSampleInvalidParamsPanics(t *testing.T) {
	for _, p := range []Uniform{{1, 1}, {3, 1}} {
		assertPanics(t, func() { p.Sample() }, fmt.Sprintf("A=%v,B=%v", p.A, p.B))
	}
}
