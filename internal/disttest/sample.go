package disttest

import (
	"math"
	"slices"
	"testing"
)

func DrawSamples(t *testing.T, d Dist, n int, inSupport func(float64) bool) []float64 {
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

type TheoreticalStats struct {
	Mean, Median, Var, MAD, Skew, ExKurt float64
}

type SampleStats struct {
	Mean, Median, Var, MAD, Skew, ExKurt float64
	SEMean, SEVar, SESkew, SEExKurt      float64
}

func sdOf(n float64, f func(i int) float64, length int) float64 {
	var s, s2 float64
	for i := 0; i < length; i++ {
		v := f(i)
		s += v
		s2 += v * v
	}
	m := s / n
	return math.Sqrt(math.Max(s2/n-m*m, 0))
}

func medianSorted(xs []float64) float64 {
	k := len(xs)
	if k%2 == 1 {
		return xs[k/2]
	}
	return (xs[k/2-1] + xs[k/2]) / 2
}

func ComputeStats(xs []float64) SampleStats {
	slices.Sort(xs)
	n := float64(len(xs))
	k := len(xs)
	var st SampleStats

	var sum float64
	for _, x := range xs {
		sum += x
	}
	mean := sum / n

	var m2, m3, m4 float64
	for _, x := range xs {
		d := x - mean
		d2 := d * d
		m2 += d2
		m3 += d2 * d
		m4 += d2 * d2
	}
	st.Mean = mean
	st.Var = m2 / (n - 1)
	m2 /= n
	m3 /= n
	m4 /= n
	sd := math.Sqrt(m2)
	g1 := m3 / (m2 * sd)
	b2 := m4 / (m2 * m2)
	st.Skew = g1
	st.ExKurt = b2 - 3

	st.Median = medianSorted(xs)
	dev := make([]float64, k)
	for i, x := range xs {
		dev[i] = math.Abs(x - st.Median)
	}
	slices.Sort(dev)
	st.MAD = medianSorted(dev)

	sqrtN := math.Sqrt(n)
	z := func(i int) float64 { return (xs[i] - mean) / sd }
	st.SEMean = sd / sqrtN
	st.SEVar = sdOf(n, func(i int) float64 { d := xs[i] - mean; return d * d }, k) / sqrtN
	st.SESkew = sdOf(n, func(i int) float64 {
		zi := z(i)
		return zi*zi*zi - 3*zi - 1.5*g1*(zi*zi-1)
	}, k) / sqrtN
	st.SEExKurt = sdOf(n, func(i int) float64 {
		zi := z(i)
		z2 := zi * zi
		return z2*z2 - 4*g1*zi - 2*b2*z2
	}, k) / sqrtN
	return st
}

func checkWithin(t *testing.T, label string, got, want, se float64) {
	t.Helper()
	tol := SETol * se
	if diff := math.Abs(got - want); !(diff <= tol) {
		t.Errorf("%s: got %.6g, want %.6g (|diff| %.3g > %.3g)", label, got, want, diff, tol)
	}
}

func CheckStats(t *testing.T, d Dist, xs []float64, th TheoreticalStats) {
	t.Helper()
	st := ComputeStats(xs)
	n := float64(len(xs))

	checkWithin(t, "mean", st.Mean, th.Mean, st.SEMean)
	checkWithin(t, "variance", st.Var, th.Var, st.SEVar)
	checkWithin(t, "skewness", st.Skew, th.Skew, st.SESkew)
	checkWithin(t, "excess kurtosis", st.ExKurt, th.ExKurt, st.SEExKurt)

	seMedian := 1 / (2 * math.Exp(d.LogProb(th.Median)) * math.Sqrt(n))
	checkWithin(t, "median", st.Median, th.Median, seMedian)

	m, D := th.Median, th.MAD
	f := func(x float64) float64 { return math.Exp(d.LogProb(x)) }
	fm, fp, fn := f(m), f(m+D), f(m-D)
	sgn := func(v float64) float64 {
		switch {
		case v > 0:
			return 1
		case v < 0:
			return -1
		}
		return 0
	}
	seMAD := sdOf(n, func(i int) float64 {
		x := xs[i]
		return (sgn(math.Abs(x-m)-D) - (fp-fn)/fm*sgn(x-m)) / (2 * (fp + fn))
	}, len(xs)) / math.Sqrt(n)
	checkWithin(t, "MAD", st.MAD, th.MAD, seMAD)
}
