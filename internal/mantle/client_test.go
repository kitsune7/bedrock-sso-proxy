package mantle

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/smithy-go"
)

// testConfig points the client at srv instead of the real Mantle endpoint by
// swapping in an HTTP client that rewrites the request URL's host. Signing still
// runs against the real hostname, so the Authorization header is what production
// would send.
func testConfig(srv *httptest.Server) aws.Config {
	target := strings.TrimPrefix(srv.URL, "http://")
	return aws.Config{
		Region:      "us-east-1",
		Credentials: credentials.NewStaticCredentialsProvider("AKIAEXAMPLE", "secret", ""),
		HTTPClient:  &redirectingClient{host: target, inner: srv.Client()},
	}
}

type redirectingClient struct {
	host  string
	inner *http.Client
}

func (c *redirectingClient) Do(req *http.Request) (*http.Response, error) {
	req.URL.Scheme = "http"
	req.URL.Host = c.host
	return c.inner.Do(req)
}

func TestDo_SignsAndPostsToResponsesPath(t *testing.T) {
	var gotPath, gotAuth, gotBody, gotHost string
	var gotLength int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		gotHost = r.Host
		gotLength = r.ContentLength
		body, _ := io.ReadAll(r.Body)
		gotBody = string(body)
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"completed","output":[]}`))
	}))
	defer srv.Close()

	resp, err := Do(context.Background(), testConfig(srv), &Request{
		Model: "openai.gpt-5.5",
		Input: []InputItem{{Type: "message", Role: "user", Content: []ContentPart{{Type: "input_text", Text: "hi"}}}},
	})
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()

	// The Responses API lives at a non-standard path on Mantle.
	if gotPath != "/openai/v1/responses" {
		t.Errorf("path = %q, want /openai/v1/responses", gotPath)
	}
	// Must be signed for bedrock-mantle — a bedrock or bedrock-runtime signature
	// is rejected.
	if !strings.Contains(gotAuth, "AWS4-HMAC-SHA256") {
		t.Errorf("Authorization = %q, want a SigV4 header", gotAuth)
	}
	if !strings.Contains(gotAuth, "/us-east-1/bedrock-mantle/aws4_request") {
		t.Errorf("Authorization credential scope = %q, want service bedrock-mantle", gotAuth)
	}
	// Content-Length is part of the signature, so it must reach the wire.
	if gotLength != int64(len(gotBody)) {
		t.Errorf("ContentLength = %d, want %d", gotLength, len(gotBody))
	}
	// SigV4 signs the Host header, so it must stay the real Mantle endpoint even
	// though the test redirects the connection elsewhere.
	if gotHost != "bedrock-mantle.us-east-1.api.aws" {
		t.Errorf("Host = %q, want the signed Mantle hostname", gotHost)
	}
	if !strings.Contains(gotBody, `"model":"openai.gpt-5.5"`) {
		t.Errorf("body = %s", gotBody)
	}

	// The body must be left unread for the caller to decode or stream.
	var decoded Response
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if decoded.Status != "completed" {
		t.Errorf("status = %q", decoded.Status)
	}
}

func TestDo_StreamSetsSSEAccept(t *testing.T) {
	var accept string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		accept = r.Header.Get("Accept")
		w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	resp, err := Do(context.Background(), testConfig(srv), &Request{Model: "m", Stream: true})
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	resp.Body.Close()

	if accept != "text/event-stream" {
		t.Errorf("Accept = %q, want text/event-stream", accept)
	}
}

func TestDo_ErrorSurfacesCodeForSSODetection(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		w.Write([]byte(`{"__type":"ExpiredTokenException","message":"token expired"}`))
	}))
	defer srv.Close()

	_, err := Do(context.Background(), testConfig(srv), &Request{Model: "m"})
	if err == nil {
		t.Fatal("expected an error for a 403")
	}

	// auth.isSSOError matches on smithy.APIError, so an expired session on this
	// path has to trigger re-authentication just like an SDK call would.
	var apiErr smithy.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("err %T does not implement smithy.APIError", err)
	}
	if apiErr.ErrorCode() != "ExpiredTokenException" {
		t.Errorf("ErrorCode = %q, want ExpiredTokenException", apiErr.ErrorCode())
	}
}
