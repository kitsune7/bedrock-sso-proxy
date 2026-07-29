package mantle

import (
	"errors"
	"strings"
	"testing"
)

func TestDecodeStream(t *testing.T) {
	// Shaped like a real Responses stream: a function call announced before its
	// argument deltas, interleaved text, and a terminal event carrying usage.
	raw := strings.Join([]string{
		`event: response.output_text.delta`,
		`data: {"type":"response.output_text.delta","output_index":0,"delta":"Hel"}`,
		``,
		`data: {"type":"response.output_text.delta","output_index":0,"delta":"lo"}`,
		``,
		`: this is a comment`,
		`data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"c1","name":"f"}}`,
		``,
		`data: {"type":"response.function_call_arguments.delta","output_index":1,"delta":"{\"x\":"}`,
		``,
		`data: not valid json`,
		``,
		`data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":5,"output_tokens":7,"total_tokens":12}}}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")

	var types []string
	var text, args string
	var usage *Usage

	err := DecodeStream(strings.NewReader(raw), func(e StreamEvent) error {
		types = append(types, e.Type)
		switch e.Type {
		case EventOutputTextDelta:
			text += e.Delta
		case EventFunctionArgsDelta:
			args += e.Delta
		case EventOutputItemAdded:
			if e.Item == nil || e.Item.CallID != "c1" || e.Item.Name != "f" {
				t.Errorf("item = %+v, want c1/f", e.Item)
			}
		case EventCompleted:
			if e.Response != nil {
				usage = e.Response.Usage
			}
		}
		return nil
	})
	if err != nil {
		t.Fatalf("DecodeStream: %v", err)
	}

	// The malformed event is skipped rather than killing the stream, and the
	// "event:", comment, blank, and [DONE] lines produce no callbacks.
	want := []string{
		EventOutputTextDelta, EventOutputTextDelta,
		EventOutputItemAdded, EventFunctionArgsDelta, EventCompleted,
	}
	if len(types) != len(want) {
		t.Fatalf("event types = %v, want %v", types, want)
	}
	for i := range want {
		if types[i] != want[i] {
			t.Errorf("event %d = %q, want %q", i, types[i], want[i])
		}
	}
	if text != "Hello" {
		t.Errorf("text = %q, want %q", text, "Hello")
	}
	if args != `{"x":` {
		t.Errorf("args = %q", args)
	}
	if usage == nil || usage.TotalTokens != 12 {
		t.Errorf("usage = %+v, want 12 total", usage)
	}
}

func TestDecodeStream_CallbackErrorAborts(t *testing.T) {
	raw := "data: {\"type\":\"a\"}\n\ndata: {\"type\":\"b\"}\n\n"
	sentinel := errors.New("client gone")

	var count int
	err := DecodeStream(strings.NewReader(raw), func(StreamEvent) error {
		count++
		return sentinel
	})

	// A write failure to the client must stop the stream immediately, not keep
	// draining the upstream body.
	if !errors.Is(err, sentinel) {
		t.Errorf("err = %v, want the sentinel", err)
	}
	if count != 1 {
		t.Errorf("callback ran %d times, want 1", count)
	}
}
