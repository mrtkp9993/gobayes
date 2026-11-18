package random

import (
	"math/rand/v2"
	"sync"
)

var (
	mu  sync.Mutex
	rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
)

func Seed(seed uint64) {
	mu.Lock()
	rng = rand.New(rand.NewPCG(seed, 0))
	mu.Unlock()
}

func Float64() float64 {
	mu.Lock()
	defer mu.Unlock()
	return rng.Float64()
}

func NormFloat64() float64 {
	mu.Lock()
	defer mu.Unlock()
	return rng.NormFloat64()
}
