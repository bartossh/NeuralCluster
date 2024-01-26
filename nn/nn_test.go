package nn

import (
	"fmt"
	"testing"

	"gotest.tools/assert"
)

type matrixDef struct {
	rows, cols int
}
type testCase struct {
	dst, a, b matrixDef
}

var testCases = []testCase{
	{
		dst: matrixDef{5, 5},
		a:   matrixDef{5, 10},
		b:   matrixDef{10, 5},
	},
	{
		dst: matrixDef{50, 50},
		a:   matrixDef{50, 100},
		b:   matrixDef{100, 50},
	},
	{
		dst: matrixDef{50, 50},
		a:   matrixDef{50, 151},
		b:   matrixDef{151, 50},
	},
	{
		dst: matrixDef{100, 100},
		a:   matrixDef{100, 1000},
		b:   matrixDef{1000, 100},
	},
	{
		dst: matrixDef{1000, 1000},
		a:   matrixDef{1000, 500},
		b:   matrixDef{500, 1000},
	},
}

var testCasesFailure = []testCase{
	{
		dst: matrixDef{5, 6},
		a:   matrixDef{5, 10},
		b:   matrixDef{10, 5},
	},
	{
		dst: matrixDef{50, 50},
		a:   matrixDef{50, 50},
		b:   matrixDef{100, 50},
	},
	{
		dst: matrixDef{50, 50},
		a:   matrixDef{50, 151},
		b:   matrixDef{150, 50},
	},
	{
		dst: matrixDef{100, 100},
		a:   matrixDef{1000, 1000},
		b:   matrixDef{1000, 100},
	},
	{
		dst: matrixDef{1000, 1000},
		a:   matrixDef{1000, 2000},
		b:   matrixDef{1000, 1000},
	},
}

func TestMatrixDotProductSuccess(t *testing.T) {
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("success dot for dst %v a %v b %v ", tc.dst, tc.a, tc.b), func(t *testing.T) {
			dst := NewMatrix(tc.dst.rows, tc.dst.cols)
			a := NewMatrix(tc.a.rows, tc.a.cols)
			a.Randomize()
			b := NewMatrix(tc.b.rows, tc.b.cols)
			b.Randomize()
			err := Dot(dst, a, b)
			assert.NilError(t, err)
		})
	}
}

func TestMatrixDotProductFailure(t *testing.T) {
	for _, tc := range testCasesFailure {
		t.Run(fmt.Sprintf("failure dot for dst %v a %v b %v ", tc.dst, tc.a, tc.b), func(t *testing.T) {
			dst := NewMatrix(tc.dst.rows, tc.dst.cols)
			a := NewMatrix(tc.a.rows, tc.a.cols)
			a.Randomize()
			b := NewMatrix(tc.b.rows, tc.b.cols)
			b.Randomize()
			err := Dot(dst, a, b)
			assert.ErrorContains(t, err, "wrong")
		})
	}
}

func BenchmarkMatrixDotProduct(b *testing.B) {
	for _, tc := range testCases {
		b.Run(fmt.Sprintf("dot dst %v a %v b %v ", tc.dst, tc.a, tc.b), func(b *testing.B) {
			dst := NewMatrix(tc.dst.rows, tc.dst.cols)
			a := NewMatrix(tc.a.rows, tc.a.cols)
			a.Randomize()
			c := NewMatrix(tc.b.rows, tc.b.cols)
			c.Randomize()
			b.ResetTimer()
			for n := 0; n < b.N; n++ {
				Dot(dst, a, c)
			}
		})
	}
}

func TestPrint(t *testing.T) {
	m := NewMatrix(5, 5)
	m.Print("m zero")
	m.Randomize()
	m.Print("m randomize")
	m.Activate(Sigmoid)
	m.Print("m sigmoid")
}

func TestConvolve(t *testing.T) {
	filter := NewMatrix(3, 3)
	filter.SetAt(0, 0, 1.0)
	filter.SetAt(0, 1, 0.0)
	filter.SetAt(0, 2, -1.0)
	filter.SetAt(1, 0, 1.0)
	filter.SetAt(1, 1, 0.0)
	filter.SetAt(1, 2, -1.0)
	filter.SetAt(2, 0, 1.0)
	filter.SetAt(2, 1, 0.0)
	filter.SetAt(2, 2, -1.0)
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("convolve dst %v a %v b %v ", tc.dst, tc.a, tc.b), func(t *testing.T) {
			dst := NewMatrix(tc.dst.rows, tc.dst.cols)
			a := NewMatrix(tc.a.rows, tc.a.cols)
			a.Randomize()
			b := NewMatrix(tc.b.rows, tc.b.cols)
			b.Randomize()
			err := Dot(dst, a, b)
			assert.NilError(t, err)
			err = dst.Convolve(filter)
			assert.NilError(t, err)
		})
	}
}

func BenchmarkConvolve(b *testing.B) {
	filter := NewMatrix(3, 3)
	filter.SetAt(0, 0, 1.0)
	filter.SetAt(0, 1, 0.0)
	filter.SetAt(0, 2, -1.0)
	filter.SetAt(1, 0, 1.0)
	filter.SetAt(1, 1, 0.0)
	filter.SetAt(1, 2, -1.0)
	filter.SetAt(2, 0, 1.0)
	filter.SetAt(2, 1, 0.0)
	filter.SetAt(2, 2, -1.0)
	for _, tc := range testCases {
		b.Run(fmt.Sprintf("testing dot for dst %v a %v b %v ", tc.dst, tc.a, tc.b), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				dst := NewMatrix(tc.dst.rows, tc.dst.cols)
				a := NewMatrix(tc.a.rows, tc.a.cols)
				a.Randomize()
				c := NewMatrix(tc.b.rows, tc.b.cols)
				c.Randomize()
				err := Dot(dst, a, c)
				assert.NilError(b, err)
				err = dst.Convolve(filter)
				assert.NilError(b, err)
			}
		})
	}
}

func TestNNForward(t *testing.T) {
	architecture := []Schema{
		{Size: 10, Activation: ReluActivation},
		{Size: 20, Activation: ReluActivation},
		{Size: 10, Activation: ReluActivation},
		{Size: 5, Activation: ReluActivation},
		{Size: 3, Activation: ReluActivation},
	}

	nn, err := NewNN(architecture)
	assert.NilError(t, err)
	nn.Randomize()
	err = nn.Forward()
	assert.NilError(t, err)
	for i := range architecture {
		nn.PrintActivationLayer(i)
		if i == len(architecture)-1 {
			break
		}
		nn.PrintWeightsLayer(i)
		nn.PrintBiasLayer(i)
	}
}

func BenchmarkNNForward(b *testing.B) {
	benchCase := [][]Schema{
		{
			{Size: 10, Activation: ReluActivation},
			{Size: 20, Activation: ReluActivation},
			{Size: 5, Activation: ReluActivation},
			{Size: 1, Activation: ReluActivation},
		},
		{
			{Size: 20, Activation: ReluActivation},
			{Size: 50, Activation: ReluActivation},
			{Size: 10, Activation: ReluActivation},
			{Size: 8, Activation: ReluActivation},
			{Size: 5, Activation: ReluActivation},
		},
		{
			{Size: 200, Activation: ReluActivation},
			{Size: 500, Activation: ReluActivation},
			{Size: 100, Activation: ReluActivation},
			{Size: 50, Activation: ReluActivation},
			{Size: 10, Activation: ReluActivation},
		},
	}

	for _, arch := range benchCase {
		b.Run(fmt.Sprintf("size factor %v", arch[0].Size), func(b *testing.B) {
			nn, err := NewNN(arch)
			assert.NilError(b, err)
			nn.Randomize()
			for n := 0; n < b.N; n++ {
				err = nn.Forward()
				assert.NilError(b, err)
			}
		})
	}

}
