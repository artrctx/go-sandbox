package dogcat

import (
	"fmt"
	"os"
	"path"
)

const (
	DownloadURL   = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
	LocalZipFile  = "kagglecatsanddogs_5340.zip"
	LocalZipDir   = "PetImages"
	InvalidSubDir = "invalid"

	DownloadChecksum = "b7974bd00a84a99921f36ee4403f089853777b5ae8d151c76a86e64900334af9"
)

var (
	ImgSubDirs   = []string{"Dog", "Cat"}
	BadDogImages = map[int]bool{11233: true, 11702: true, 11912: true, 2317: true, 9500: true}
	BadCatImages = map[int]bool{10404: true, 11095: true, 12080: true, 5370: true, 6435: true, 666: true}
	BadImages    = [2]map[int]bool{BadDogImages, BadCatImages}

	MaxCount = 12500
	NumDogs  = MaxCount - len(BadDogImages)
	NumCats  = MaxCount - len(BadCatImages)
	NumValid = [2]int{NumDogs, NumCats}
	_        = NumValid
)

// // Download Dogs vs Cats Dataset to baseDir, unzips it, and checks for mal-formed files (there are a few).
// func Download(baseDir string) error {
// 	zipFilePath := path.Join(baseDir, LocalZipFile)
// 	targetZipPath := path.Join(baseDir, LocalZipDir)
// 	if err := downloader.DownloadAndUnzipIfMissing(DownloadURL, zipFilePath, baseDir, targetZipPath, DownloadChecksum); err != nil {
// 		return err
// 	}
// 	return PrefilterValidImages(baseDir)
// }

// PrefilterValidImages is like FilterValidImages, but uses pre-generated list of images known to be invalid.
func PrefilterValidImages(baseDir string) error {
	// Check if it has already been filtered.
	invalidDir := path.Join(baseDir, InvalidSubDir)
	_, err := os.Stat(invalidDir)
	if err == nil {
		// Assume they have already been filtered, return immediately.
		return nil
	} else if !os.IsNotExist(err) {
		return err
	}

	// Create subdirectories for invalid files.
	if err = os.Mkdir(invalidDir, 0755); err != nil {
		return err
	}
	for _, subDir := range ImgSubDirs {
		dir := path.Join(invalidDir, subDir)
		if err = os.Mkdir(dir, 0755); err != nil {
			return err
		}
	}

	for classIdx, subDir := range ImgSubDirs {
		dir := path.Join(baseDir, LocalZipDir, subDir)
		invalidSubDir := path.Join(invalidDir, subDir)
		var invalidList map[int]bool
		if classIdx == 0 {
			invalidList = BadDogImages
		} else {
			invalidList = BadCatImages
		}
		for imgIdx := range invalidList {
			name := fmt.Sprintf("%d.jpg", imgIdx)
			imgPath := path.Join(dir, name)
			fmt.Printf("Moving %s to %s: image known to have invalid format\n", imgPath, invalidSubDir)
			if err = os.Rename(imgPath, path.Join(invalidSubDir, name)); err != nil {
				return err
			}
		}
	}
	return nil
}
