// https://gist.github.com/higuma/dbcd006546eb844c01e5102b4d0bcc93
package mnist

import (
	"encoding/binary"
	"fmt"
	"os"
	"slices"

	"golang.org/x/sync/errgroup"
)

const (
	LabelFileMagic = 0x00000801
	ImageFileMagic = 0x00000803
)

func fileError(f *os.File, msg string) error {
	return fmt.Errorf("Invalid File Format: %v; msg: %v", f.Name(), msg)
}

func readInt32(f *os.File) (int, error) {
	buf := make([]byte, 4)
	n, err := f.Read(buf)

	if err != nil {
		return 0, err
	}

	if n != 4 {
		return 0, fileError(f, "invalid buf size")
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
		return nil, fileError(file, "invalid magic")
	}
	n, err := readInt32(file)
	if err != nil {
		return nil, fileError(file, "invalid count")
	}

	w, err := readInt32(file)
	if err != nil {
		return nil, fileError(file, "invalid width")
	}

	h, err := readInt32(file)
	if err != nil {
		return nil, fileError(file, "invalid height")
	}

	size := n * w * h
	d := &ImageData{n, w, h, make([]byte, size)}
	len, err := file.Read(d.Data)
	if err != nil || size != len {
		return nil, fileError(file, "invalid comparison size")
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
		return nil, fileError(f, "invalid magic")
	}

	n, err := readInt32(f)
	if err != nil {
		return nil, fileError(f, "invalid count")
	}

	d := &LabelData{n, make([]byte, n)}
	if len, err := f.Read(d.Data); err != nil || len != n {
		return nil, fileError(f, "invalid size comparison")
	}
	return d, nil
}

type DigitImage struct {
	Digit int
	Image [][]byte
}

type Dataset struct {
	N    int
	W    int
	H    int
	Data []DigitImage
}

func ReadDataset(labelPath, imagePath string) (*Dataset, error) {
	lChan, iChan := make(chan *LabelData, 1), make(chan *ImageData, 1)

	var eg errgroup.Group
	// label data read
	eg.Go(func() error {
		ld, err := ReadLabelFile(labelPath)
		if err != nil {
			return err
		}
		lChan <- ld
		return nil
	})

	eg.Go(func() error {
		idt, err := ReadImageFile(imagePath)
		if err != nil {
			return err
		}
		iChan <- idt
		return nil
	})

	if err := eg.Wait(); err != nil {
		return nil, err
	}

	labelData, imgData := <-lChan, <-iChan

	if labelData.N != imgData.N {
		return nil, fmt.Errorf("data size is match label N (%d) != image N (%d)", labelData.N, imgData.N)
	}

	dataset := &Dataset{imgData.N, imgData.W, imgData.H, make([]DigitImage, imgData.N)}

	idx := 0
	// chunk by Count * Row Then Chunk again by Height to form Image
	for imgBytes := range slices.Chunk(slices.Collect(slices.Chunk(imgData.Data, imgData.N*imgData.W)), imgData.H) {
		ds := &dataset.Data[idx]
		ds.Digit = int(labelData.Data[idx])
		ds.Image = imgBytes
		idx++
	}

	return dataset, nil
}
