package imageref

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// testImage encodes a 2x2 image in the named format, so the sniffing path is
// exercised against real magic numbers rather than hand-written headers.
func testImage(t *testing.T, format string) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	img.Set(0, 0, color.RGBA{R: 255, A: 255})
	var buf bytes.Buffer
	var err error
	switch format {
	case "png":
		err = png.Encode(&buf, img)
	case "jpeg":
		err = jpeg.Encode(&buf, img, nil)
	case "gif":
		err = gif.Encode(&buf, img, nil)
	default:
		t.Fatalf("unknown format %q", format)
	}
	if err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func TestResolveDataURL(t *testing.T) {
	for _, format := range []string{"png", "jpeg", "gif"} {
		t.Run(format, func(t *testing.T) {
			data := testImage(t, format)
			url := "data:image/" + format + ";base64," + base64.StdEncoding.EncodeToString(data)

			ref, err := Resolve(context.Background(), nil, url)
			if err != nil {
				t.Fatalf("Resolve: %v", err)
			}
			if ref.Format != format {
				t.Errorf("Format = %q, want %q", ref.Format, format)
			}
			if !bytes.Equal(ref.Bytes, data) {
				t.Errorf("Bytes round-trip mismatch")
			}
		})
	}

	// The declared media type is ignored in favour of the actual bytes: clients
	// mislabel images, and Bedrock rejects a format that disagrees.
	t.Run("declared type loses to sniffed type", func(t *testing.T) {
		data := testImage(t, "png")
		url := "data:image/jpeg;base64," + base64.StdEncoding.EncodeToString(data)
		ref, err := Resolve(context.Background(), nil, url)
		if err != nil {
			t.Fatalf("Resolve: %v", err)
		}
		if ref.Format != "png" {
			t.Errorf("Format = %q, want png from the sniffed bytes", ref.Format)
		}
	})

	t.Run("rejects non-image bytes", func(t *testing.T) {
		url := "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("not an image at all"))
		if _, err := Resolve(context.Background(), nil, url); err == nil {
			t.Error("expected an error for a non-image payload")
		}
	})

	t.Run("rejects non-base64 data URL", func(t *testing.T) {
		if _, err := Resolve(context.Background(), nil, "data:image/png,rawbytes"); err == nil {
			t.Error("expected an error for a non-base64 data URL")
		}
	})
}

func TestResolveHTTP(t *testing.T) {
	data := testImage(t, "png")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/ok.png":
			w.Write(data)
		case "/huge":
			// One byte over the cap must be rejected, not truncated into a
			// corrupt image.
			w.Write(bytes.Repeat([]byte{0}, MaxBytes+1))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	t.Run("fetches and sniffs", func(t *testing.T) {
		ref, err := Resolve(context.Background(), srv.Client(), srv.URL+"/ok.png")
		if err != nil {
			t.Fatalf("Resolve: %v", err)
		}
		if ref.Format != "png" || !bytes.Equal(ref.Bytes, data) {
			t.Errorf("got format %q, %d bytes", ref.Format, len(ref.Bytes))
		}
	})

	t.Run("propagates upstream failure", func(t *testing.T) {
		if _, err := Resolve(context.Background(), srv.Client(), srv.URL+"/missing"); err == nil {
			t.Error("expected an error for a 404")
		}
	})

	t.Run("enforces the size cap", func(t *testing.T) {
		_, err := Resolve(context.Background(), srv.Client(), srv.URL+"/huge")
		if err == nil || !strings.Contains(err.Error(), "limit") {
			t.Errorf("err = %v, want a size-limit error", err)
		}
	})
}

func TestResolveS3(t *testing.T) {
	// S3 objects are passed through unfetched, so the format can only come from
	// the extension.
	ref, err := Resolve(context.Background(), nil, "s3://bucket/path/pic.JPEG")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if ref.Format != "jpeg" {
		t.Errorf("Format = %q, want jpeg", ref.Format)
	}
	if ref.S3URI != "s3://bucket/path/pic.JPEG" {
		t.Errorf("S3URI = %q, want the URI unchanged", ref.S3URI)
	}
	if ref.Bytes != nil {
		t.Error("S3 refs must not carry bytes — the object is never fetched")
	}

	if _, err := Resolve(context.Background(), nil, "s3://bucket/pic.tiff"); err == nil {
		t.Error("expected an error for an unsupported extension")
	}
}

func TestResolveUnsupportedScheme(t *testing.T) {
	if _, err := Resolve(context.Background(), nil, "ftp://host/pic.png"); err == nil {
		t.Error("expected an error for an unsupported scheme")
	}
}

func TestDataURLRoundTrip(t *testing.T) {
	data := testImage(t, "png")
	ref := &Ref{Format: "png", Bytes: data}

	back, err := Resolve(context.Background(), nil, ref.DataURL())
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if !bytes.Equal(back.Bytes, data) {
		t.Error("DataURL round-trip lost bytes")
	}
}
