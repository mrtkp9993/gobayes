package random

import "math/rand/v2"

type Generator struct {
	rng *rand.Rand
}

func New(seed, stream uint64) *Generator {
	return &Generator{rng: rand.New(rand.NewPCG(seed, stream))}
}

func (g *Generator) Float64() float64 { return g.rng.Float64() }

func (g *Generator) NormFloat64() float64 { return g.rng.NormFloat64() }

func Uint64() uint64 { return rand.Uint64() }
