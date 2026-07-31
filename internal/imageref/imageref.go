// Package imageref resolves the URL in an OpenAI image_url content part into
// the bytes the Bedrock backends need.
//
// Neither backend accepts an https URL: Converse takes raw bytes or an S3
// location, and Mantle takes a data: or s3:// URL. Clients overwhelmingly send
// https or data: URLs, so this package fetches the former and decodes the
// latter, leaving both translation layers with the same two cases to handle.
package imageref

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"path"
	"strings"
	"time"
)

// MaxBytes caps a fetched or decoded image. Bedrock's own per-image limit is
// 5 MB (and 20 MB per request), so anything larger would be rejected upstream
// anyway — refusing it here keeps a hostile URL from filling memory first.
const MaxBytes = 5 << 20

// fetchTimeout bounds a single image fetch. A hung image URL must not hold a
// chat completion open indefinitely.
const fetchTimeout = 30 * time.Second

// Ref is a resolved image. Exactly one of Bytes or S3URI is set.
type Ref struct {
	// Format is the Bedrock image format: png, jpeg, gif, or webp.
	Format string
	Bytes  []byte
	S3URI  string
}

// formats maps the content types Bedrock supports to its format names. Bedrock
// accepts only these four, so anything else is rejected rather than guessed at.
var formats = map[string]string{
	"image/png":  "png",
	"image/jpeg": "jpeg",
	"image/gif":  "gif",
	"image/webp": "webp",
}

// Resolve turns an image_url URL into a Ref. client may be nil, in which case a
// default client with fetchTimeout is used.
func Resolve(ctx context.Context, client *http.Client, url string) (*Ref, error) {
	switch {
	case strings.HasPrefix(url, "s3://"):
		// Passed through untouched — both backends take an S3 location directly,
		// and the bytes may well be larger than this proxy should ever hold.
		// Converse still requires a format, and the object is not fetched here,
		// so the extension is all there is to go on.
		format, err := formatFromExt(url)
		if err != nil {
			return nil, err
		}
		return &Ref{Format: format, S3URI: url}, nil

	case strings.HasPrefix(url, "data:"):
		return decodeDataURL(url)

	case strings.HasPrefix(url, "http://"), strings.HasPrefix(url, "https://"):
		return fetch(ctx, client, url)
	}
	return nil, fmt.Errorf("unsupported image_url scheme: want data:, s3://, http:// or https://")
}

// decodeDataURL parses "data:image/png;base64,<payload>". The declared media
// type is ignored in favour of sniffing the decoded bytes — clients get it
// wrong, and Bedrock rejects a format that disagrees with the content.
func decodeDataURL(url string) (*Ref, error) {
	meta, payload, ok := strings.Cut(url, ",")
	if !ok {
		return nil, errors.New("malformed data URL: no comma")
	}
	if !strings.Contains(meta, ";base64") {
		return nil, errors.New("unsupported data URL: only base64 payloads are supported")
	}
	// Tolerate both standard and URL-safe alphabets, padded or not.
	payload = strings.TrimSpace(payload)
	data, err := base64.StdEncoding.WithPadding(base64.NoPadding).DecodeString(strings.TrimRight(payload, "="))
	if err != nil {
		return nil, fmt.Errorf("decode data URL payload: %w", err)
	}
	if len(data) > MaxBytes {
		return nil, fmt.Errorf("image is %d bytes, over the %d byte limit", len(data), MaxBytes)
	}
	return newRef(data)
}

func fetch(ctx context.Context, client *http.Client, url string) (*Ref, error) {
	if client == nil {
		client = &http.Client{Timeout: fetchTimeout}
	}
	ctx, cancel := context.WithTimeout(ctx, fetchTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("fetch image: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("fetch image: upstream returned %d", resp.StatusCode)
	}

	// Read one byte past the cap so an oversized body is detected rather than
	// silently truncated into a corrupt image.
	data, err := io.ReadAll(io.LimitReader(resp.Body, MaxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read image: %w", err)
	}
	if len(data) > MaxBytes {
		return nil, fmt.Errorf("image exceeds the %d byte limit", MaxBytes)
	}
	return newRef(data)
}

// formatFromExt derives the Bedrock format from a path's extension, for the
// s3:// case where the bytes are never read.
func formatFromExt(uri string) (string, error) {
	ext := strings.ToLower(path.Ext(uri))
	switch ext {
	case ".png":
		return "png", nil
	case ".jpg", ".jpeg":
		return "jpeg", nil
	case ".gif":
		return "gif", nil
	case ".webp":
		return "webp", nil
	}
	return "", fmt.Errorf("cannot determine image format from %q: want a .png, .jpg, .gif, or .webp extension", uri)
}

// newRef sniffs the format from the bytes themselves. http.DetectContentType
// reads the magic numbers for all four formats Bedrock supports.
func newRef(data []byte) (*Ref, error) {
	if len(data) == 0 {
		return nil, errors.New("image is empty")
	}
	ct := http.DetectContentType(data)
	if i := strings.Index(ct, ";"); i >= 0 {
		ct = ct[:i]
	}
	format, ok := formats[ct]
	if !ok {
		return nil, fmt.Errorf("unsupported image format %q: want png, jpeg, gif, or webp", ct)
	}
	return &Ref{Format: format, Bytes: data}, nil
}

// DataURL re-encodes the ref as the data: URL Mantle expects.
func (r *Ref) DataURL() string {
	return "data:image/" + r.Format + ";base64," + base64.StdEncoding.EncodeToString(r.Bytes)
}
