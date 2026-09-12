package diarization

import (
	"context"
	"errors"
	"os"

	"github.com/artrctx/gossiper/internal/util/model"
	"golang.org/x/sync/errgroup"
)

type Client struct {
	embed *model.Model
	seg   *model.Model
}

func New(ctx context.Context) (*Client, error) {
	eg := errgroup.Group{}

	cwd, err := os.Getwd()
	if err != nil {
		return nil, err
	}
	var embed, seg *model.Model
	eg.Go(func() (err error) {
		// input:
		//  fbank       [B, 998, 80]
		//  speaker_mask [B, 125]
		// output: [B, 128]
		embed, err = model.New(cwd + "/models/embedding.onnx")
		return
	})

	eg.Go(func() (err error) {
		// input: [1, 1, 160000]
		// output: [1, 589, 7]
		seg, err = model.New(cwd + "/models/segmentation.onnx")
		return
	})

	if err := eg.Wait(); err != nil {
		return nil, err
	}

	return &Client{embed, seg}, nil
}

func (c *Client) Close() error {
	embedErr := c.embed.Close()
	segErr := c.seg.Close()
	if embedErr != nil || segErr != nil {
		return errors.Join(embedErr, segErr)
	}
	return nil
}
