package brain

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
		t.Run(fmt.Sprintf("testing dot for dst %v a %v b %v ", tc.dst, tc.a, tc.b), func(t *testing.T) {
			dst := MatrixNew(tc.dst.rows, tc.dst.cols)
			a := MatrixNew(tc.a.rows, tc.a.cols)
			a.Randomize()
			b := MatrixNew(tc.b.rows, tc.b.cols)
			b.Randomize()
			err := Dot(dst, a, b)
			assert.NilError(t, err)
		})
	}
}

func TestMatrixDotProductFailure(t *testing.T) {
	for _, tc := range testCasesFailure {
		t.Run(fmt.Sprintf("testing dot for dst %v a %v b %v ", tc.dst, tc.a, tc.b), func(t *testing.T) {
			dst := MatrixNew(tc.dst.rows, tc.dst.cols)
			a := MatrixNew(tc.a.rows, tc.a.cols)
			a.Randomize()
			b := MatrixNew(tc.b.rows, tc.b.cols)
			b.Randomize()
			err := Dot(dst, a, b)
			assert.ErrorContains(t, err, "wrong")
		})
	}
}

func BenchmarkMatrixDotProduct(b *testing.B) {
	for _, tc := range testCases {
		b.Run(fmt.Sprintf("bench dot for dst %v a %v b %v ", tc.dst, tc.a, tc.b), func(b *testing.B) {
			dst := MatrixNew(tc.dst.rows, tc.dst.cols)
			a := MatrixNew(tc.a.rows, tc.a.cols)
			a.Randomize()
			c := MatrixNew(tc.b.rows, tc.b.cols)
			c.Randomize()
			b.ResetTimer()
			for n := 0; n < b.N; n++ {
				Dot(dst, a, c)
			}
		})
	}
}

func TestPrint(t *testing.T) {
	m := MatrixNew(5, 5)
	m.Print("m zero")
	m.Randomize()
	m.Print("m randomize")
	m.Activate(Sigmoid)
	m.Print("m sigmoid")
}

func TestConvolve(t *testing.T) {
	filter := MatrixNew(3, 3)
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
		t.Run(fmt.Sprintf("testing dot for dst %v a %v b %v ", tc.dst, tc.a, tc.b), func(t *testing.T) {
			dst := MatrixNew(tc.dst.rows, tc.dst.cols)
			a := MatrixNew(tc.a.rows, tc.a.cols)
			a.Randomize()
			b := MatrixNew(tc.b.rows, tc.b.cols)
			b.Randomize()
			err := Dot(dst, a, b)
			assert.NilError(t, err)
			err = dst.Convolve(filter)
			assert.NilError(t, err)
		})
	}
}

func BenchmarkConvolve(b *testing.B) {
	filter := MatrixNew(3, 3)
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
				dst := MatrixNew(tc.dst.rows, tc.dst.cols)
				a := MatrixNew(tc.a.rows, tc.a.cols)
				a.Randomize()
				c := MatrixNew(tc.b.rows, tc.b.cols)
				c.Randomize()
				err := Dot(dst, a, c)
				assert.NilError(b, err)
				err = dst.Convolve(filter)
				assert.NilError(b, err)
			}
		})
	}
}
