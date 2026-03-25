// https://gist.github.com/higuma/dbcd006546eb844c01e5102b4d0bcc93
package mnist

import (
	"encoding/binary"
	"fmt"
	"os"
)

const (
	LabelFileMagic = 0x00000801
	ImageFileMagic = 0x00000801
)

func fileError(f *os.File) error {
	return fmt.Errorf("Invalid File Format: %v", f.Name())
}

func readInt32(f *os.File) (int, error) {
	buf := make([]byte, 4)
	n, err := f.Read(buf)

	if err != nil {
		return 0, err
	}

	if n != 4 {
		return 0, fileError(f)
	}
	return int(binary.BigEndian.Uint32(buf)), nil
}

type ImageData struct {
	// count
	N int
	// width
	W int
	// height
	H    int
	Data []byte
}

func ReadImageFile(path string) (*ImageData, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	magic, err := readInt32(file)
	if err != nil || magic != ImageFileMagic {
		return nil, fileError(file)
	}

	n, err := readInt32(file)
	if err != nil {
		return nil, fileError(file)
	}

	w, err := readInt32(file)
	if err != nil {
		return nil, fileError(file)
	}

	h, err := readInt32(file)
	if err != nil {
		return nil, fileError(file)
	}

	size := n * w * h
	d := &ImageData{n, w, h, make([]byte, size)}
	len, err := file.Read(d.Data)
	if err != nil || size != len {
		return nil, fileError(file)
	}

	return d, nil
}

type LabelData struct {
	N    int
	Data []byte
}

func ReadLabelFile(path string) (*LabelData, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	magic, err := readInt32(f)
	if err != nil || magic != LabelFileMagic {
		return nil, fileError(f)
	}

	n, err := readInt32(f)
	if err != nil {
		return nil, fileError(f)
	}

	d := &LabelData{n, make([]byte, n)}
	if len, err := f.Read(d.Data); err != nil || len != n {
		return nil, fileError(f)
	}
	return d, nil
}
