package mantle

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	v4 "github.com/aws/aws-sdk-go-v2/aws/signer/v4"
	"github.com/aws/smithy-go"
)

// signingService is the SigV4 service name for the Mantle endpoint. It is not
// "bedrock" — requests signed for bedrock or bedrock-runtime are rejected.
const signingService = "bedrock-mantle"

// responsesPath is where Mantle serves the OpenAI Responses API. Note the
// "/openai" segment: it is specific to this API on this endpoint, not the
// "/v1/responses" path OpenAI itself uses.
const responsesPath = "/openai/v1/responses"

// endpoint returns the Mantle base URL for a region.
func endpoint(region string) string {
	return fmt.Sprintf("https://bedrock-mantle.%s.api.aws", region)
}

// Do sends a Responses API request and returns the raw HTTP response with its
// body unread, so callers can either decode JSON or stream SSE from it. A
// non-2xx status is returned as an error with the body included, and the body
// is closed in that case.
//
// cfg supplies both the credentials and the signing region. Mantle models are
// in-region only, so the config's region must be one where the model exists.
func Do(ctx context.Context, cfg aws.Config, req *Request) (*http.Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("encode mantle request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint(cfg.Region)+responsesPath, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if req.Stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	} else {
		httpReq.Header.Set("Accept", "application/json")
	}
	// SigV4 signs Content-Length; without it the signature covers a header the
	// transport would otherwise add itself, and the request is rejected.
	httpReq.ContentLength = int64(len(body))

	creds, err := cfg.Credentials.Retrieve(ctx)
	if err != nil {
		return nil, fmt.Errorf("retrieve AWS credentials: %w", err)
	}

	sum := sha256.Sum256(body)
	if err := v4.NewSigner().SignHTTP(ctx, creds, httpReq, hex.EncodeToString(sum[:]), signingService, cfg.Region, time.Now()); err != nil {
		return nil, fmt.Errorf("sign mantle request: %w", err)
	}

	client := cfg.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		defer resp.Body.Close()
		// Cap the error body — a stray HTML error page should not end up in a log
		// line in full.
		detail, _ := io.ReadAll(io.LimitReader(resp.Body, 8<<10))
		return nil, &APIError{Status: resp.StatusCode, Body: string(detail)}
	}

	return resp, nil
}

// APIError is a non-2xx response from Mantle. It reports the AWS error code
// when the body carries one, so auth.isSSOError can recognize an expired
// session and trigger re-authentication.
type APIError struct {
	Status int
	Body   string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("mantle request failed with status %d: %s", e.Status, e.Body)
}

// ErrorCode implements smithy.APIError so credential errors surface through the
// same detection path as SDK calls.
func (e *APIError) ErrorCode() string {
	var parsed struct {
		Type    string `json:"__type"`
		Code    string `json:"code"`
		Message string `json:"message"`
		Error   struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(e.Body), &parsed); err != nil {
		return ""
	}
	for _, c := range []string{parsed.Type, parsed.Code, parsed.Error.Code} {
		if c != "" {
			return c
		}
	}
	return ""
}

func (e *APIError) ErrorMessage() string { return e.Body }

func (e *APIError) ErrorFault() smithy.ErrorFault {
	if e.Status >= 500 {
		return smithy.FaultServer
	}
	return smithy.FaultClient
}
