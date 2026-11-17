package distributions

import (
	"math"
	"testing"
)

const (
	nSamples = 200_000
	seTol    = 5.0
)

type logProbCase struct {
	name string
	x    float64
	want float64
}

func assertClose(t *testing.T, got, want float64, label string) {
	t.Helper()
	if math.IsNaN(got) {
		t.Errorf("%s: got NaN, want %v", label, want)
		return
	}
	if math.IsInf(want, 0) || math.IsInf(got, 0) {
		if got != want {
			t.Errorf("%s: got %v, want %v", label, got, want)
		}
		return
	}
	const tol = 1e-12
	if diff := math.Abs(got - want); diff > tol*math.Max(1, math.Abs(want)) {
		t.Errorf("%s: got %.17g, want %.17g (diff %g)", label, got, want, diff)
	}
}

func runLogProbCases(t *testing.T, d Distribution, cases []logProbCase) {
	t.Helper()
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			assertClose(t, d.LogProb(c.x), c.want, "LogProb")
		})
	}
}

func drawSamples(t *testing.T, d Distribution, n int, inSupport func(float64) bool) []float64 {
	t.Helper()
	xs := make([]float64, n)
	bad := 0
	for i := range xs {
		x := d.Sample()
		if math.IsNaN(x) || math.IsInf(x, 0) || !inSupport(x) {
			if bad < 5 {
				t.Errorf("sample %d = %v is outside the support or not finite", i, x)
			}
			bad++
		}
		xs[i] = x
	}
	if bad > 0 {
		t.Fatalf("%d/%d samples invalid", bad, n)
	}
	return xs
}

func checkMoments(t *testing.T, xs []float64, wantMean, wantVar float64) {
	t.Helper()
	n := float64(len(xs))
	var sum float64
	for _, x := range xs {
		sum += x
	}
	mean := sum / n
	var m2, m4 float64
	for _, x := range xs {
		d := x - mean
		d2 := d * d
		m2 += d2
		m4 += d2 * d2
	}
	variance := m2 / (n - 1)
	m4 /= n

	seMean := math.Sqrt(wantVar / n)
	if diff := math.Abs(mean - wantMean); diff > seTol*seMean {
		t.Errorf("mean: got %.6g, want %.6g (|diff| %.3g > %.3g)", mean, wantMean, diff, seTol*seMean)
	}

	seVar := math.Sqrt(math.Max(m4-variance*variance, 0) / n)
	if diff := math.Abs(variance - wantVar); diff > seTol*seVar {
		t.Errorf("variance: got %.6g, want %.6g (|diff| %.3g > %.3g)", variance, wantVar, diff, seTol*seVar)
	}
}

func assertPanics(t *testing.T, f func(), label string) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: expected panic, got none", label)
		}
	}()
	f()
}
