package mantle

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"
)

// maxEventBytes caps a single SSE data field. The default bufio.Scanner limit
// is 64KB, which a long reasoning summary can exceed.
const maxEventBytes = 1 << 20

// DecodeStream reads Responses API SSE events from r and calls fn for each one.
// It stops at the terminal event, at EOF, or as soon as fn returns an error,
// which it returns unchanged so callers can abort a stream mid-flight.
//
// Responses SSE names its events semantically — "response.output_text.delta",
// "response.completed" — rather than sending one repeated chunk shape. Each
// event's JSON carries its own type, so the SSE "event:" line is redundant and
// only "data:" is parsed.
func DecodeStream(r io.Reader, fn func(StreamEvent) error) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64<<10), maxEventBytes)

	for scanner.Scan() {
		line := scanner.Text()
		data, ok := strings.CutPrefix(line, "data:")
		if !ok {
			// Blank lines, comments, and "event:" lines carry nothing we need.
			continue
		}
		data = strings.TrimSpace(data)
		if data == "" || data == "[DONE]" {
			continue
		}

		var event StreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			// A single malformed event should not kill an otherwise good
			// stream — the next one is likely fine.
			continue
		}
		if err := fn(event); err != nil {
			return err
		}
	}

	return scanner.Err()
}

// Event type names used by the Responses streaming API.
const (
	// EventOutputTextDelta carries an incremental chunk of assistant text.
	EventOutputTextDelta = "response.output_text.delta"
	// EventFunctionArgsDelta carries an incremental chunk of a function call's
	// JSON arguments.
	EventFunctionArgsDelta = "response.function_call_arguments.delta"
	// EventOutputItemAdded announces a new output item. For function calls it is
	// the only place the name and call_id appear — the argument deltas that
	// follow are keyed by output_index alone.
	EventOutputItemAdded = "response.output_item.added"
	// EventCompleted and EventIncomplete are terminal and carry the final
	// response object, including usage.
	EventCompleted  = "response.completed"
	EventIncomplete = "response.incomplete"
	// EventFailed is terminal and carries the error.
	EventFailed = "response.failed"
)
