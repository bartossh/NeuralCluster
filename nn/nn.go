package nn

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
)

type ActivationType string

const (
	SigmoidActivation   ActivationType = "sigmoid"
	TanhActivation      ActivationType = "tanh"
	ReluActivation      ActivationType = "relu"
	LeakyReluActivation ActivationType = "leaky_relu"
	EluActivation       ActivationType = "elu"
)

const alpha float64 = 0.01

// Neutral activation function does nothing to the input value.
func Neutral(a float64) float64 { return a }

// Sigmoid calculate sigmoid function for given input.
func Sigmoid(a float64) float64 { return 1.0 / (1.0 + math.Exp(-a)) }

// Than calculates hyperbolic tangent function for given input.
func Tanh(a float64) float64 { return (math.Exp(a) - math.Exp(-a)) / (math.Exp(a) + math.Exp(-a)) }

// Relu calculates rectified linear unit function for given input.
func Relu(a float64) float64 {
	if a > 0 {
		return a
	}
	return 0
}

// LeakyRelu calculates leaky rectified linear unit function for given input.
// Alpha value is constant and equal to 0.01
func LeakyRelu(a float64) float64 {
	if a > 0 {
		return a
	}
	return a * alpha
}

// Elu calculates exponential linear unit function for given input.
func Elu(a float64) float64 {
	if a > 0 {
		return a
	}
	return alpha * (math.Exp(a) - 1)
}

// Sotmax calculates softmax function on slice of float64 values.
// Returns slice with the same size as input slice.
func Softmax(x []float64) []float64 {
	var sum float64
	result := make([]float64, len(x))

	for _, val := range x {
		sum += math.Exp(val)
	}

	for i, val := range x {
		result[i] = math.Exp(val) / sum
	}

	return result
}

func selectActivationFunction(t ActivationType) ActivationFunction {
	switch t {
	case SigmoidActivation:
		return Sigmoid
	case TanhActivation:
		return Tanh
	case ReluActivation:
		return Relu
	case LeakyReluActivation:
		return LeakyRelu
	case EluActivation:
		return Elu
	default:
		return Neutral
	}
}

// ActivationFunction is an activation signature function.
// Activations functions avaliable in this package are: Sigmoid, Tanh, Relu, LeakyRelu, Elu.
type ActivationFunction func(a float64) float64

// Matrix is a 2D matrix.
type Matrix struct {
	values []float64
	rows   int
	cols   int
}

// NewMatrix creates new Matrix with given number of rows and columns.
func NewMatrix(rows, cols int) Matrix {
	return Matrix{
		values: make([]float64, rows*cols),
		rows:   rows,
		cols:   cols,
	}
}

// At returns value at given row and column or if outside of range returns an error.
func (m Matrix) At(row, col int) (float64, error) {
	if row >= m.rows || row < 0 {
		return 0.0, fmt.Errorf("wrong row index")
	}
	if col >= m.cols || col < 0 {
		return 0.0, fmt.Errorf("wrong column index")
	}
	return m.values[col+row*m.cols], nil
}

// SetAt sets value at given row and column or if outside of range returns an error.
func (m Matrix) SetAt(row, col int, v float64) error {
	if row >= m.rows || row < 0 {
		return fmt.Errorf("wrong row index")
	}
	if col >= m.cols || col < 0 {
		return fmt.Errorf("wrong column index")
	}
	m.values[col+row*m.cols] = v
	return nil
}

// Randomize rendomizes the matrix values.
func (m Matrix) Randomize() {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.SetAt(i, j, rand.Float64())
		}
	}
}

// Activate activates all vlaues in the matrix.
func (m Matrix) Activate(actf ActivationFunction) {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			v, _ := m.At(i, j)
			m.SetAt(i, j, actf(v))
		}
	}
}

// Convolve function applies filter convolution into matrix.
// Filter matrix should have the same number of rows and columns and be of smaller size than
// the convolution receiver matrix.
func (m Matrix) Convolve(filter Matrix) error {
	if filter.rows != filter.cols {
		return fmt.Errorf(
			"expected the same number of rows and columns, received rows [ %v ] cols [ %v ]",
			filter.rows, filter.cols,
		)
	}
	if filter.rows >= m.rows {
		return fmt.Errorf(
			"expected reciver matrix of bigger size than the filter, received rows in matrix [ %v ], in filter [ %v ]",
			m.rows, filter.rows,
		)
	}
	if filter.cols >= m.cols {
		return fmt.Errorf(
			"expected reciver matrix of bigger size than the filter, received columns in matrix [ %v ], in filter [ %v ]",
			m.cols, filter.cols,
		)
	}

	padding := filter.cols / 2

	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			// apply filter
			sum := 0.0
			for k := -padding; k <= padding; k++ {
				for l := -padding; l <= padding; l++ {
					r := i + k
					c := j + l
					if r >= 0 && r < m.rows && c >= 0 && c < m.cols {
						mV, err := m.At(r, c)
						if err != nil {
							// unreachable
							return fmt.Errorf("unreachable, %w", err)
						}
						mF, err := filter.At(k+padding, l+padding)
						if err != nil {
							// unreachable
							return fmt.Errorf("unreachable, %w", err)
						}
						sum += mV * mF
					}
				}
			}
			m.SetAt(i, j, sum)
		}
	}
	return nil
}

// Row returns copy of a matrix row at row index as a matrix with single row.
func (m Matrix) Row(row int) (Matrix, error) {
	var nm Matrix
	if row < 0 || row >= m.rows {
		return nm, fmt.Errorf("exceeded row number, expected row in range [ 0, %v ], received %v", m.rows, row)
	}
	nm.cols = m.cols
	nm.rows = 1
	nm.values = make([]float64, m.cols)
	copy(nm.values, m.values[row:row+m.cols])
	return nm, nil
}

// Print prints matrix into stdout.
func (m Matrix) Print(name string) {
	fmt.Printf("%s = [\n", name)
	for i := 0; i < m.rows; i++ {
		fmt.Printf("  ")
		for j := 0; j < m.cols; j++ {
			v, _ := m.At(i, j)
			fmt.Printf("%.3f, ", v)
		}
		fmt.Printf("\n")
	}
	fmt.Printf("]\n")
}

// Dot calculates dot product of two matrices and saves the product in the destination matrix.
// Input matrix a needs to have number of columns corresponding to matrix b number of rows otherwise error is returned.
// Destination matrix dst needs to have number of rows corresponding to matrix a number of rows otherwise error is returned.
// Destination matrix dst needs to have number of columns corresponding to matrix b number of columns otherwise error is returned.
func Dot(dst, a, b Matrix) error {
	if a.cols != b.rows {
		return fmt.Errorf(
			"wrong size of matrices, matrix a cols [ %v ] doesn't equal to matrix b rows [ %v ]",
			a.cols, b.rows,
		)
	}
	if dst.rows != a.rows {
		return fmt.Errorf(
			"wrong size of matrices, matrix a rows [ %v ] doesn't equal to matrix dst rows [ %v ]",
			a.rows, dst.rows,
		)
	}
	if dst.cols != b.cols {
		return fmt.Errorf(
			"wrong size of matrices, matrix a cols [ %v ] doesn't equal to matrix dst cols [ %v ]",
			b.cols, dst.cols,
		)
	}

	for i := 0; i < dst.rows; i++ {
		for j := 0; j < dst.cols; j++ {
			dst.SetAt(i, j, 0.0)
			for k := 0; k < a.cols; k++ {
				dstV, err := dst.At(i, j)
				if err != nil {
					return err
				}
				aV, err := a.At(i, k)
				if err != nil {
					// unreachable
					return fmt.Errorf("unreachable, %w", err)
				}
				bV, err := b.At(k, j)
				if err != nil {
					// unreachable
					return fmt.Errorf("unreachable, %w", err)
				}
				if err := dst.SetAt(i, j, dstV+aV*bV); err != nil {
					return fmt.Errorf("unreachable, %w", err)
				}
			}
		}
	}
	return nil
}

// Copy copies matrix from src to dst.
// Matrices shall have the same size (number of rows and columns shall match) or error is returned.
func Copy(dst, src Matrix) error {
	if dst.rows != src.rows {
		return fmt.Errorf(
			"unmatching number of rows, dst [ %v ], src [ %v]", dst.rows, src.rows)
	}
	if dst.cols != src.cols {
		return fmt.Errorf(
			"unmatching number of columns, dst [ %v ], src [ %v]", dst.cols, src.cols)
	}
	if len(dst.values) != len(src.values) {
		return fmt.Errorf(
			"unmatching underlining slice len, dst [ %v ], src [ %v]", len(dst.values), len(src.values))

	}
	copy(dst.values, src.values)
	return nil
}

// Sum sums src and dst matrices to dst.
// Matrices shall have the same size (number of rows and columns shall match) or error is returned.
func Sum(dst, src Matrix) error {
	if dst.rows != src.rows {
		return fmt.Errorf(
			"unmatching number of rows, dst [ %v ], src [ %v]", dst.rows, src.rows)
	}
	if dst.cols != src.cols {
		return fmt.Errorf(
			"unmatching number of columns, dst [ %v ], src [ %v]", dst.cols, src.cols)
	}

	for i := 0; i < src.rows; i++ {
		for j := 0; j < src.cols; j++ {
			sV, err := src.At(i, j)
			if err != nil {
				// unreachable
				return fmt.Errorf("unreachable, %w", err)
			}
			dV, err := dst.At(i, j)
			if err != nil {
				// unreachable
				return fmt.Errorf("unreachable, %w", err)
			}
			err = dst.SetAt(i, j, sV+dV)
			if err != nil {
				// unreachable
				return fmt.Errorf("unreachable, %w", err)
			}
		}
	}
	return nil
}

// Layer is a NN layer.
type layer struct {
	ws  Matrix // weights
	bs  Matrix // biases
	as  Matrix // activations
	act ActivationFunction
}

// Schema describes the layer schema and can be decoded from json or yaml file.
type Schema struct {
	Size       int            `json:"size" yaml:"size"`
	Activation ActivationType `json:"activates_type" yaml:"activates_type"`
}

// NN is a Neural Network.
type NN struct {
	arch []layer
}

// NewNN creates new NN based on the architecture.
func NewNN(architecture []Schema) (NN, error) {
	if len(architecture) < 3 {
		return NN{}, errors.New("expecting at list 3 layers, input layer, hidden layer(s), output layer")
	}
	arch := make([]layer, len(architecture))
	arch[0] = layer{as: NewMatrix(1, architecture[0].Size), act: selectActivationFunction(architecture[0].Activation)}
	for i := range architecture {
		var actf ActivationFunction = Neutral
		switch i {
		case 0:
			continue
		case len(architecture) - 1:
			actf = selectActivationFunction(architecture[i].Activation)
		default:
		}
		arch[i].act = actf
		arch[i-1].ws = NewMatrix(architecture[i-1].Size, architecture[i].Size)
		arch[i-1].bs = NewMatrix(1, architecture[i].Size)
		arch[i].as = NewMatrix(1, architecture[i].Size)
	}

	return NN{arch: arch}, nil
}

// Randomize randomizes the all layers matrices values.
func (nn NN) Randomize() {
	for _, l := range nn.arch {
		l.as.Randomize()
		l.ws.Randomize()
		l.bs.Randomize()
	}
}

// Forward applies feed forward action on neural network.
func (nn NN) Forward() error {
	for i := 0; i < len(nn.arch)-2; i++ {
		if err := Dot(nn.arch[i+1].as, nn.arch[i].as, nn.arch[i].ws); err != nil {
			return fmt.Errorf("unreachable, %w", err)
		}
		if err := Sum(nn.arch[i+1].as, nn.arch[i].bs); err != nil {
			return fmt.Errorf("unreachable, %w", err)
		}
		nn.arch[i+1].as.Activate(nn.arch[i].act)
	}
	return nil
}

// Cost function calculates the cost.
func (nn NN) Cost(in, out Matrix) (float64, error) {
	if in.cols != nn.arch[0].as.cols {
		return 0.0, fmt.Errorf(
			"expected input number of columns [ %v ], received  [ %v ]",
			nn.arch[0].as.cols,
			in.cols,
		)
	}
	if out.cols != nn.arch[len(nn.arch)-1].as.cols {
		return 0.0, fmt.Errorf(
			"expected output number of columns [ %v ], received  [ %v ]",
			nn.arch[len(nn.arch)-1].as.cols,
			in.cols,
		)
	}
	if in.rows != out.rows {
		return 0.0, fmt.Errorf(
			"expected input and output to match with number of rows, input [ %v ] rows, out [ %v ] rows",
			in.rows,
			out.rows,
		)
	}
	var cost float64

	for i := 0; i < in.rows; i++ {
		inRow, err := in.Row(i)
		if err != nil {
			return 0.0, fmt.Errorf("unreachable, %w", err)
		}
		if err := Copy(nn.arch[0].as, inRow); err != nil {
			return 0.0, fmt.Errorf("unreachable, %w", err)
		}
		if err := nn.Forward(); err != nil {
			return 0.0, err
		}
		for j := 0; j < out.cols; j++ {
			outV, err := nn.arch[len(nn.arch)-1].as.At(0, j)
			if err != nil {
				return 0.0, err
			}
			testV, err := out.At(i, j)
			if err != nil {
				return 0.0, err
			}
			d := outV - testV
			cost += d * d
		}
	}

	return cost / float64(in.rows), nil
}

// PrintActivationLayer prints activation layer from NN architecture at given index.
func (nn NN) PrintActivationLayer(idx int) {
	if idx < 0 || idx > len(nn.arch) {
		fmt.Printf("index out of architecture size, expected 0 to %v, received %v\n", len(nn.arch), idx)
	}
	nn.arch[idx].as.Print(fmt.Sprintf("activation layer %v", idx))
}

// PrintWeightsLayer prints weights layer from NN architecture at given index.
func (nn NN) PrintWeightsLayer(idx int) {
	if idx < 0 || idx > len(nn.arch)-1 {
		fmt.Printf("index out of architecture size, expected 0 to %v, received %v\n", len(nn.arch), idx)
	}
	nn.arch[idx].ws.Print(fmt.Sprintf("weights layer %v", idx))
}

// PrintBiasLayer prints bias layer from NN architecture at given index.
func (nn NN) PrintBiasLayer(idx int) {
	if idx < 0 || idx > len(nn.arch)-1 {
		fmt.Printf("index out of architecture size, expected 0 to %v, received %v\n", len(nn.arch), idx)
	}
	nn.arch[idx].bs.Print(fmt.Sprintf("bias layer %v", idx))
}
